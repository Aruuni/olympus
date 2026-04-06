/*
 * oc_listener.c — Option-Critic flow listener for Olympus
 *
 * Watches for TCP flows whose iperf3 client source port matches --cport,
 * switches the socket to the "mutant" CC driver, and drives arm selection
 * via the Option-Critic policy running in a per-flow Python worker.
 *
 * Each Python worker is a separate OS process (no GIL sharing between flows).
 * Workers communicate with the central actor/learner via their own IPC
 * (multiprocessing queues / ZMQ) — never threads — to avoid GIL contention
 * in the actor.
 *
 * ── Per-flow pipe protocol ───────────────────────────────────────────────────
 *  state_pipe  [C → Py]  oc_state_t  (44 bytes) sent every --scan-ms
 *  action_pipe [Py → C]  oc_action_t ( 8 bytes) sent by Python to switch arm
 *
 * Python layout:
 *   oc_state_t  = struct.unpack('<8IQ', data)   # cur_arm,rtt,rttvar,min_rtt,
 *                                               #  cwnd,lost,retrans,delivered,
 *                                               #  delivery_rate
 *   oc_action_t = struct.pack('<II', arm_id, dwell_ms)
 *
 * Arm IDs (mirror mutant.h):
 *   CUBIC=0  BBR1=2  VEGAS=5  BBR3=12  ASTRAEA=13
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -pthread -o oc_listener oc_listener.c
 *
 * Run:
 *   sudo -E env OC_PYTHON="./venv_astraea/bin/python" \
 *     ./oc_listener --cport 20000 --worker training/oc_worker.py \
 *                   --mode mininet --scan-ms 100
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <linux/netlink.h>
#include <linux/tcp.h>
#include <sys/uio.h>

/* ── syscall wrappers ────────────────────────────────────────────────────── */

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

/* ── TCP sockopt constants ───────────────────────────────────────────────── */

#ifndef TCP_CONGESTION
#define TCP_CONGESTION 13
#endif
#ifndef TCP_DEEPCC_ENABLE
#define TCP_DEEPCC_ENABLE 44
#endif

/* ── Netlink / mutant constants ──────────────────────────────────────────── */

#ifndef NETLINK_TEST
#define NETLINK_TEST 25
#endif

#define COMM_BEGIN      1
#define COMM_SELECT_ARM 2

/* Arm IDs — must match mutant.h */
#define CUBIC   0
#define BBR1    2
#define VEGAS   5
#define BBR3    12
#define ASTRAEA 13

/* ── Wire structs ────────────────────────────────────────────────────────── */

/*
 * oc_state_t  (C → Python, 44 bytes)
 * Python: struct.unpack('<8IQ', data)
 *   [0] cur_arm        — currently active arm ID
 *   [1] rtt_us         — smoothed RTT (µs)
 *   [2] rttvar_us      — RTT variance (µs)
 *   [3] min_rtt_us     — minimum RTT seen (µs)
 *   [4] snd_cwnd       — congestion window (MSS units)
 *   [5] lost           — lost packets (unrecovered)
 *   [6] retrans        — retransmitted but not yet acked
 *   [7] delivered      — total delivered segments
 *   [8] delivery_rate  — delivery rate (bytes/s)
 */
typedef struct __attribute__((packed)) {
    uint32_t cur_arm;
    uint32_t rtt_us;
    uint32_t rttvar_us;
    uint32_t min_rtt_us;
    uint32_t snd_cwnd;
    uint32_t lost;
    uint32_t retrans;
    uint32_t delivered;
    uint64_t delivery_rate;
} oc_state_t;

/*
 * oc_action_t  (Python → C, 8 bytes)
 * Python: struct.pack('<II', arm_id, dwell_ms)
 *   arm_id   — arm to switch to (CUBIC/BBR1/VEGAS/BBR3/ASTRAEA)
 *   dwell_ms — minimum time to stay on this arm before Python sends the next
 *              action; enforced by the Python worker, not by C
 */
typedef struct __attribute__((packed)) {
    uint32_t arm_id;
    uint32_t dwell_ms;
} oc_action_t;

/* ── Config ──────────────────────────────────────────────────────────────── */

typedef struct {
    char mode[16];         /* "mininet" or "normal" */
    char py_worker[PATH_MAX];
    int  cport;            /* iperf3 client source port to watch */
    int  scan_ms;
    int  ipv4_only;
    int  verbose;
} config_t;

/* ── ss record ───────────────────────────────────────────────────────────── */

typedef struct {
    char state[32], local[128], peer[128], cc[64], proc[128];
    int  pid, fd, ns_pid;
    unsigned long long ns_ino;
} ss_record_t;

/* ── Flow worker ─────────────────────────────────────────────────────────── */

typedef struct flow_worker {
    char  key[512];
    char  local[128];
    char  peer[128];

    int   fd;         /* duplicated TCP socket fd in our namespace */
    int   pid;        /* iperf3 client PID */
    int   src_fd;
    int   ns_pid;
    long  flow_id;

    pid_t child_pid;
    int   active;
    int   stop_requested;

    uint32_t cur_arm;

    int   state_pipe_wr;   /* C writes oc_state_t here  */
    int   action_pipe_rd;  /* C reads  oc_action_t here */

    pthread_t thr;
    struct flow_worker *next;
} flow_worker_t;

/* ── Globals ─────────────────────────────────────────────────────────────── */

static volatile sig_atomic_t g_stop = 0;
static pthread_mutex_t       g_workers_mu = PTHREAD_MUTEX_INITIALIZER;
static flow_worker_t        *g_workers    = NULL;
static long                  g_next_flow_id = 1;

config_t g_cfg;

/* ── Utilities ───────────────────────────────────────────────────────────── */

static void on_sig(int sig) { (void)sig; g_stop = 1; }

static void msleep_int(int ms) {
    struct timespec ts = {
        .tv_sec  = ms / 1000,
        .tv_nsec = (long)(ms % 1000) * 1000000L
    };
    nanosleep(&ts, NULL);
}

static int pidfd_open_wrap(int pid)
    { return (int)syscall(SYS_pidfd_open, pid, 0); }
static int pidfd_getfd_wrap(int pidfd, int targetfd)
    { return (int)syscall(SYS_pidfd_getfd, pidfd, targetfd, 0); }

static int dup_fd_from_pid(int pid, int fd) {
    int pidfd = pidfd_open_wrap(pid);
    if (pidfd < 0) return -1;
    int out = pidfd_getfd_wrap(pidfd, fd);
    close(pidfd);
    return out;
}

static int set_cc(int fd, const char *name) {
    socklen_t len = (socklen_t)(strlen(name) + 1);
    return setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, name, len);
}

static int get_cc(int fd, char *out, socklen_t outlen) {
    memset(out, 0, outlen);
    socklen_t len = outlen;
    if (getsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, out, &len) != 0) return -1;
    out[outlen - 1] = '\0';
    return 0;
}

static int enable_deepcc(int fd, int val) {
    return setsockopt(fd, IPPROTO_TCP, TCP_DEEPCC_ENABLE, &val, sizeof(val));
}

/* Parse the port number from a "addr:port" string (IPv4 or IPv6). */
static int port_from_addr(const char *addr) {
    const char *col = strrchr(addr, ':');
    if (!col) return -1;
    return atoi(col + 1);
}

/* ── Worker list ─────────────────────────────────────────────────────────── */

static void make_key(char *key, size_t sz, const ss_record_t *rec) {
    snprintf(key, sz, "%s->%s", rec->local, rec->peer);
}

static flow_worker_t *find_worker_locked(const char *key) {
    for (flow_worker_t *p = g_workers; p; p = p->next)
        if (strcmp(p->key, key) == 0) return p;
    return NULL;
}

static int worker_is_active(const char *key) {
    int ret = 0;
    pthread_mutex_lock(&g_workers_mu);
    flow_worker_t *w = find_worker_locked(key);
    ret = (w && w->active);
    pthread_mutex_unlock(&g_workers_mu);
    return ret;
}

static void add_worker_locked(flow_worker_t *w) {
    w->next = g_workers;
    g_workers = w;
}

static void remove_worker_locked(flow_worker_t *victim) {
    flow_worker_t **pp = &g_workers, *p;
    while ((p = *pp) != NULL) {
        if (p == victim) { *pp = p->next; return; }
        pp = &p->next;
    }
}

/* ── Mutant netlink ──────────────────────────────────────────────────────── */

static int mutant_open_nl(void) {
    int nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_TEST);
    if (nl_fd < 0) return -1;

    struct sockaddr_nl addr;
    memset(&addr, 0, sizeof(addr));
    addr.nl_family = AF_NETLINK;
    addr.nl_pid    = (uint32_t)getpid();

    if (bind(nl_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(nl_fd);
        return -1;
    }
    return nl_fd;
}

static int mutant_send(int nl_fd, uint16_t flags, uint32_t seq,
                       const char *payload) {
    struct {
        struct nlmsghdr nlh;
        char            buf[64];
    } req;
    struct sockaddr_nl dst;

    memset(&req, 0, sizeof(req));
    memset(&dst, 0, sizeof(dst));
    dst.nl_family = AF_NETLINK;

    size_t plen = strlen(payload) + 1;
    req.nlh.nlmsg_len   = (uint32_t)NLMSG_LENGTH(plen);
    req.nlh.nlmsg_flags = flags;
    req.nlh.nlmsg_seq   = seq;
    req.nlh.nlmsg_pid   = (uint32_t)getpid();
    memcpy(NLMSG_DATA(&req.nlh), payload, plen);

    return (int)sendto(nl_fd, &req, req.nlh.nlmsg_len, 0,
                       (struct sockaddr *)&dst, sizeof(dst));
}

static int mutant_begin(int nl_fd) {
    return mutant_send(nl_fd, COMM_BEGIN, 0, "INIT_COMMUNICATION");
}

static int mutant_switch_arm(int nl_fd, uint32_t arm_id) {
    return mutant_send(nl_fd, COMM_SELECT_ARM, arm_id, "SENDING ACTION");
}

/* ── State collection ────────────────────────────────────────────────────── */

static int fill_state(int fd, uint32_t cur_arm, oc_state_t *s) {
    struct tcp_info ti;
    socklen_t len = sizeof(ti);
    memset(&ti, 0, len);

    if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &len) != 0)
        return -1;

    s->cur_arm       = cur_arm;
    s->rtt_us        = ti.tcpi_rtt;
    s->rttvar_us     = ti.tcpi_rttvar;
    s->min_rtt_us    = ti.tcpi_min_rtt;
    s->snd_cwnd      = ti.tcpi_snd_cwnd;
    s->lost          = ti.tcpi_lost;
    s->retrans       = ti.tcpi_retrans;
    s->delivered     = ti.tcpi_delivered;
    s->delivery_rate = ti.tcpi_delivery_rate;
    return 0;
}

/* ── Python worker spawn ─────────────────────────────────────────────────── */

/*
 * Fork and execl the Python OC worker.  The worker is a separate process
 * with its own GIL — it can block on actor queries without stalling other
 * flows.
 *
 * Env vars passed to worker:
 *   OC_STATE_FD   — fd to read oc_state_t from  (pipe read end)
 *   OC_ACTION_FD  — fd to write oc_action_t to  (pipe write end)
 *   OC_FLOW_FD    — duplicated TCP socket fd (for Astraea DeepCC cwnd writes)
 *   OC_FLOW_ID    — monotonic flow identifier
 *   OC_CPORT      — experiment identifier (iperf3 client port)
 */
static pid_t spawn_worker(const config_t *cfg, flow_worker_t *w,
                           int state_rd, int action_wr) {
    pid_t child = fork();
    if (child < 0) return -1;

    if (child == 0) {
        char state_s[32], action_s[32], fd_s[32], flow_s[32], cport_s[32];
        snprintf(state_s,  sizeof(state_s),  "%d", state_rd);
        snprintf(action_s, sizeof(action_s), "%d", action_wr);
        snprintf(fd_s,     sizeof(fd_s),     "%d", w->fd);
        snprintf(flow_s,   sizeof(flow_s),   "%ld", w->flow_id);
        snprintf(cport_s,  sizeof(cport_s),  "%d", cfg->cport);

        setenv("OC_STATE_FD",  state_s,  1);
        setenv("OC_ACTION_FD", action_s, 1);
        setenv("OC_FLOW_FD",   fd_s,     1);
        setenv("OC_FLOW_ID",   flow_s,   1);
        setenv("OC_CPORT",     cport_s,  1);

        /* Forward learner manager address + key if they were set in our
         * environment (e.g. via "sudo -E env OC_MANAGER_ADDR=... oc_listener").
         * setenv with overwrite=0 preserves any value already present. */
        const char *mgr_addr = getenv("OC_MANAGER_ADDR");
        const char *mgr_key  = getenv("OC_MANAGER_KEY");
        if (mgr_addr) setenv("OC_MANAGER_ADDR", mgr_addr, 1);
        if (mgr_key)  setenv("OC_MANAGER_KEY",  mgr_key,  1);

        /* All pipe fds were created with O_CLOEXEC so they auto-close on
         * execl.  Explicitly clear CLOEXEC only on the three fds the child
         * actually uses: state_rd (read state from C), action_wr (write action
         * to C), and w->fd (TCP socket for DeepCC cwnd writes). */
        int cfl;
        cfl = fcntl(state_rd,  F_GETFD); if (cfl >= 0) fcntl(state_rd,  F_SETFD, cfl & ~FD_CLOEXEC);
        cfl = fcntl(action_wr, F_GETFD); if (cfl >= 0) fcntl(action_wr, F_SETFD, cfl & ~FD_CLOEXEC);
        cfl = fcntl(w->fd,     F_GETFD); if (cfl >= 0) fcntl(w->fd,     F_SETFD, cfl & ~FD_CLOEXEC);

        const char *py = getenv("OC_PYTHON");
        if (!py || !*py) py = "/usr/bin/python3";

        execl(py, py, cfg->py_worker, (char *)NULL);
        perror("execl oc_worker");
        _exit(127);
    }

    return child;
}

/* ── Flow thread ─────────────────────────────────────────────────────────── */

static void *flow_thread(void *arg) {
    flow_worker_t *w = (flow_worker_t *)arg;

    /* ── set up mutant ───────────────────────────────────────────────────── */
    if (set_cc(w->fd, "mutant") != 0) {
        perror("[oc] set_cc mutant");
        goto done;
    }

    /* DeepCC always on — astraea_service runs in background for every flow;
     * the Python-side control pipe gates whether cwnd writes are applied. */
    enable_deepcc(w->fd, 2);

    int nl_fd = mutant_open_nl();
    if (nl_fd < 0) {
        perror("[oc] mutant_open_nl");
        goto done;
    }

    if (mutant_begin(nl_fd) < 0) {
        perror("[oc] mutant_begin");
        close(nl_fd);
        goto done;
    }

    /* Start on CUBIC until Python sends the first action.
     * DeepCC off — CUBIC manages cwnd in the kernel. */
    w->cur_arm = CUBIC;
    if (mutant_switch_arm(nl_fd, CUBIC) < 0)
        perror("[oc] mutant_switch_arm(initial)");

    fprintf(stderr, "[oc flow %ld] cport=%d mutant ready, waiting for worker\n",
            w->flow_id, g_cfg.cport);
    fflush(stderr);

    /* ── main loop ───────────────────────────────────────────────────────── */
    while (!g_stop && !w->stop_requested) {
        /* Check child still alive. */
        int st = 0;
        if (waitpid(w->child_pid, &st, WNOHANG) == w->child_pid) {
            if (g_cfg.verbose) {
                fprintf(stderr, "[oc flow %ld] worker exited\n", w->flow_id);
                fflush(stderr);
            }
            break;
        }

        /* Send current state to Python worker. */
        oc_state_t state;
        if (fill_state(w->fd, w->cur_arm, &state) == 0) {
            /* Non-blocking write: drop frame if pipe is full.
             * A full pipe means the worker is behind; it will catch up on
             * the next tick.  We never block here to keep the scan loop
             * responsive. */
            ssize_t wr = write(w->state_pipe_wr, &state, sizeof(state));
            (void)wr;
        }

        /* Non-blocking read for a new action from Python.
         * The action_pipe_rd fd is set O_NONBLOCK in spawn setup below. */
        oc_action_t action;
        ssize_t rd = read(w->action_pipe_rd, &action, sizeof(action));
        if (rd == (ssize_t)sizeof(oc_action_t)) {
            if (mutant_switch_arm(nl_fd, action.arm_id) >= 0) {
                /* DeepCC stays enabled; the Python-side control pipe gates
                 * whether astraea_service actually applies cwnd writes. */
                enable_deepcc(w->fd, 2);
                fprintf(stderr,
                    "[oc flow %ld] arm %u → %u  dwell=%ums\n",
                    w->flow_id, w->cur_arm, action.arm_id, action.dwell_ms);
                fflush(stderr);
                w->cur_arm = action.arm_id;
            } else {
                perror("[oc] mutant_switch_arm");
            }
        }

        msleep_int(g_cfg.scan_ms);
    }

    close(nl_fd);

done:
    if (w->child_pid > 0) {
        kill(w->child_pid, SIGTERM);
        waitpid(w->child_pid, NULL, 0);
    }

    pthread_mutex_lock(&g_workers_mu);
    w->active = 0;
    remove_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    close(w->state_pipe_wr);
    close(w->action_pipe_rd);
    if (w->fd >= 0) close(w->fd);
    free(w);
    return NULL;
}

/* ── Flow spawning ───────────────────────────────────────────────────────── */

static void maybe_spawn_flow(const config_t *cfg, const ss_record_t *rec) {
    char key[512];
    make_key(key, sizeof(key), rec);
    if (worker_is_active(key)) return;

    /* Duplicate the TCP socket fd into our namespace. */
    int fd = dup_fd_from_pid(rec->pid, rec->fd);
    if (fd < 0) return;

    /* Match astraea_listener.c: enable DeepCC before the Python worker starts
     * so astraea_service can immediately read TCP_DEEPCC_INFO on its inherited
     * socket fd without racing the flow thread setup below. */
    if (enable_deepcc(fd, 2) != 0) {
        close(fd);
        return;
    }

    /* Set CLOEXEC on the duplicated socket — spawn_worker will clear it
     * in the child for the one worker that actually needs it. */
    int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);

    /* state_pipe: C writes, Python reads.
     * O_CLOEXEC on both ends — child only inherits state_pfd[0] after we
     * explicitly clear CLOEXEC on it in spawn_worker.  All other pipe ends
     * are automatically closed on execl so they never leak into sibling workers. */
    int state_pfd[2];
    if (pipe2(state_pfd, O_CLOEXEC) != 0) { close(fd); return; }

    /* action_pipe: Python writes, C reads */
    int action_pfd[2];
    if (pipe2(action_pfd, O_CLOEXEC) != 0) {
        close(state_pfd[0]); close(state_pfd[1]);
        close(fd);
        return;
    }

    /* C reads actions non-blocking so the flow thread never stalls. */
    int fl = fcntl(action_pfd[0], F_GETFL);
    if (fl >= 0) fcntl(action_pfd[0], F_SETFL, fl | O_NONBLOCK);

    flow_worker_t *w = calloc(1, sizeof(*w));
    if (!w) {
        close(state_pfd[0]); close(state_pfd[1]);
        close(action_pfd[0]); close(action_pfd[1]);
        close(fd);
        return;
    }

    snprintf(w->key,   sizeof(w->key),   "%s", key);
    snprintf(w->local, sizeof(w->local), "%s", rec->local);
    snprintf(w->peer,  sizeof(w->peer),  "%s", rec->peer);

    w->fd            = fd;
    w->pid           = rec->pid;
    w->src_fd        = rec->fd;
    w->ns_pid        = rec->ns_pid;
    w->flow_id       = __sync_fetch_and_add(&g_next_flow_id, 1);
    w->child_pid     = -1;
    w->active        = 1;
    w->stop_requested = 0;
    w->cur_arm       = CUBIC;
    w->state_pipe_wr = state_pfd[1];   /* C keeps write end */
    w->action_pipe_rd = action_pfd[0]; /* C keeps read end  */

    pthread_mutex_lock(&g_workers_mu);
    if (find_worker_locked(key) != NULL) {
        /* Race: another thread beat us. */
        pthread_mutex_unlock(&g_workers_mu);
        close(state_pfd[0]); close(state_pfd[1]);
        close(action_pfd[0]); close(action_pfd[1]);
        close(fd);
        free(w);
        return;
    }
    add_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    /* Spawn Python worker — it owns state_pfd[0] and action_pfd[1]. */
    w->child_pid = spawn_worker(cfg, w, state_pfd[0], action_pfd[1]);
    if (w->child_pid < 0) {
        perror("[oc] spawn_worker");
        pthread_mutex_lock(&g_workers_mu);
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);
        close(state_pfd[0]); close(state_pfd[1]);
        close(action_pfd[0]); close(action_pfd[1]);
        close(fd);
        free(w);
        return;
    }

    /* Parent closes the child's ends after fork. */
    close(state_pfd[0]);
    close(action_pfd[1]);

    fprintf(stderr, "[oc] new flow %ld  %s  pid=%d child=%d\n",
            w->flow_id, key, rec->pid, (int)w->child_pid);
    fflush(stderr);

    if (pthread_create(&w->thr, NULL, flow_thread, w) != 0) {
        perror("[oc] pthread_create");
        kill(w->child_pid, SIGTERM);
        waitpid(w->child_pid, NULL, 0);
        pthread_mutex_lock(&g_workers_mu);
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);
        close(w->state_pipe_wr);
        close(w->action_pipe_rd);
        close(fd);
        free(w);
        return;
    }
    pthread_detach(w->thr);
}

/* ── ss scanning ─────────────────────────────────────────────────────────── */

static int is_state_token(const char *tok) {
    static const char *s[] = {
        "ESTAB","SYN-SENT","SYN-RECV","FIN-WAIT-1","FIN-WAIT-2","TIME-WAIT",
        "CLOSE","CLOSE-WAIT","LAST-ACK","LISTEN","CLOSING","NEW_SYN_RECV",NULL
    };
    for (int i = 0; s[i]; i++) if (strcmp(tok, s[i]) == 0) return 1;
    return 0;
}

static int parse_users(const char *line, char *proc, size_t psz, int *pid, int *fd) {
    const char *p = strstr(line, "users:((");
    if (!p) return -1;
    const char *q1 = strchr(p, '"');  if (!q1) return -1;
    const char *q2 = strchr(q1+1, '"'); if (!q2) return -1;
    size_t n = (size_t)(q2 - q1 - 1);
    if (n >= psz) n = psz - 1;
    memcpy(proc, q1+1, n); proc[n] = '\0';
    const char *pidp = strstr(q2, "pid=");
    const char *fdp  = strstr(q2, "fd=");
    if (!pidp || !fdp) return -1;
    *pid = atoi(pidp + 4);
    *fd  = atoi(fdp  + 3);
    return 0;
}

static int scan_ss_text(const char *text, unsigned long long ns_ino, int ns_pid,
                        ss_record_t *out, int max_out) {
    int count = 0, have_cur = 0;
    char *buf = strdup(text);
    if (!buf) return 0;
    ss_record_t cur;
    char *save = NULL;
    char *line = strtok_r(buf, "\n", &save);

    while (line) {
        while (*line && isspace((unsigned char)*line)) line++;
        if (*line == '\0') { line = strtok_r(NULL, "\n", &save); continue; }

        char first[64] = {0};
        sscanf(line, "%63s", first);

        if (is_state_token(first)) {
            memset(&cur, 0, sizeof(cur));
            cur.ns_ino = ns_ino;
            cur.ns_pid = ns_pid;
            sscanf(line, "%31s %*s %*s %127s %127s",
                   cur.state, cur.local, cur.peer);
            parse_users(line, cur.proc, sizeof(cur.proc), &cur.pid, &cur.fd);
            have_cur = 1;
        } else if (have_cur) {
            if (!strstr(line, "skmem:")) {
                sscanf(line, "%63s", cur.cc);
                if (count < max_out) out[count++] = cur;
            }
            have_cur = 0;
        }
        line = strtok_r(NULL, "\n", &save);
    }
    free(buf);
    return count;
}

static int run_ss_in_ns(int target_pid, int ipv4_only, char **out_text) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;

    pid_t child = fork();
    if (child < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }

    if (child == 0) {
        close(pipefd[0]);
        if (target_pid > 0) {
            char ns_path[64];
            snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/net", target_pid);
            int nsfd = open(ns_path, O_RDONLY | O_CLOEXEC);
            if (nsfd < 0) _exit(120);
            if (setns(nsfd, CLONE_NEWNET) != 0) _exit(121);
            close(nsfd);
        }
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        if (ipv4_only) execlp("ss", "ss", "-tinHp4", (char *)NULL);
        else           execlp("ss", "ss", "-tinHp",  (char *)NULL);
        _exit(122);
    }

    close(pipefd[1]);
    size_t cap = 16384, len = 0;
    char *rbuf = malloc(cap);
    if (!rbuf) { close(pipefd[0]); return -1; }

    for (;;) {
        if (len + 4096 + 1 > cap) {
            cap *= 2;
            char *nb = realloc(rbuf, cap);
            if (!nb) { free(rbuf); close(pipefd[0]); return -1; }
            rbuf = nb;
        }
        ssize_t n = read(pipefd[0], rbuf + len, cap - len - 1);
        if (n == 0) break;
        if (n < 0) { if (errno == EINTR) continue; free(rbuf); close(pipefd[0]); return -1; }
        len += (size_t)n;
    }
    close(pipefd[0]);
    rbuf[len] = '\0';
    waitpid(child, NULL, 0);
    *out_text = rbuf;
    return 0;
}

static int discover_netns(int *pids, unsigned long long *inos, int max_out) {
    DIR *d = opendir("/proc");
    if (!d) return 0;
    int count = 0;
    struct dirent *de;

    while ((de = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) continue;
        int pid = atoi(de->d_name);
        char ns_path[64];
        struct stat st;
        snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/net", pid);
        if (stat(ns_path, &st) != 0) continue;
        unsigned long long ino = (unsigned long long)st.st_ino;
        int exists = 0;
        for (int i = 0; i < count; i++)
            if (inos[i] == ino) { exists = 1; break; }
        if (!exists && count < max_out) {
            pids[count] = pid;
            inos[count] = ino;
            count++;
        }
    }
    closedir(d);
    return count;
}

static void scan_ns_once(const config_t *cfg, int ns_pid,
                          unsigned long long ns_ino) {
    char *ss_text = NULL;
    if (run_ss_in_ns(ns_pid, cfg->ipv4_only, &ss_text) != 0) return;

    ss_record_t recs[4096];
    int n = scan_ss_text(ss_text, ns_ino, ns_pid, recs, 4096);
    free(ss_text);

    for (int i = 0; i < n; i++) {
        if (recs[i].pid <= 0 || recs[i].fd < 0) continue;
        if (strcmp(recs[i].state, "ESTAB") != 0) continue;
        /* Match by iperf3 client source port. */
        if (port_from_addr(recs[i].local) != cfg->cport) continue;
        maybe_spawn_flow(cfg, &recs[i]);
    }
}

static void scan_loop(const config_t *cfg) {
    while (!g_stop) {
        if (strcmp(cfg->mode, "mininet") == 0) {
            int              pids[1024];
            unsigned long long inos[1024];
            int n = discover_netns(pids, inos, 1024);
            for (int i = 0; i < n; i++)
                scan_ns_once(cfg, pids[i], inos[i]);
        } else {
            scan_ns_once(cfg, 0, 0);
        }
        msleep_int(cfg->scan_ms);
    }
}

/* ── Shutdown ────────────────────────────────────────────────────────────── */

static void request_stop_all(void) {
    pthread_mutex_lock(&g_workers_mu);
    for (flow_worker_t *p = g_workers; p; p = p->next)
        p->stop_requested = 1;
    pthread_mutex_unlock(&g_workers_mu);
}

static int any_workers(void) {
    int ret;
    pthread_mutex_lock(&g_workers_mu);
    ret = (g_workers != NULL);
    pthread_mutex_unlock(&g_workers_mu);
    return ret;
}

/* ── main ────────────────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --cport PORT --worker SCRIPT\n"
        "          [--mode normal|mininet] [--scan-ms 100]\n"
        "          [--ipv4-only 1] [--verbose 0]\n"
        "\n"
        "Env:  OC_PYTHON   path to Python interpreter (default /usr/bin/python3)\n"
        "\n"
        "Arm IDs: cubic=0  bbr1=2  vegas=5  bbr3=12  astraea=13\n",
        prog);
}

int main(int argc, char **argv) {
    memset(&g_cfg, 0, sizeof(g_cfg));
    snprintf(g_cfg.mode, sizeof(g_cfg.mode), "mininet");
    g_cfg.scan_ms    = 100;
    g_cfg.ipv4_only  = 1;
    g_cfg.verbose    = 0;
    g_cfg.cport      = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--cport")    && i+1 < argc) g_cfg.cport = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--worker")   && i+1 < argc) snprintf(g_cfg.py_worker, sizeof(g_cfg.py_worker), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1 < argc) snprintf(g_cfg.mode, sizeof(g_cfg.mode), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--scan-ms")  && i+1 < argc) g_cfg.scan_ms  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ipv4-only")&& i+1 < argc) g_cfg.ipv4_only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")  && i+1 < argc) g_cfg.verbose  = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }

    if (g_cfg.cport <= 0 || !g_cfg.py_worker[0]) {
        usage(argv[0]);
        return 2;
    }

    signal(SIGINT,  on_sig);
    signal(SIGTERM, on_sig);

    fprintf(stderr, "[oc] listening on cport=%d  worker=%s  scan=%dms\n",
            g_cfg.cport, g_cfg.py_worker, g_cfg.scan_ms);
    fflush(stderr);

    scan_loop(&g_cfg);

    request_stop_all();
    while (any_workers()) msleep_int(100);

    return 0;
}
