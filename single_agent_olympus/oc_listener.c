/*
 * single_agent_olympus/oc_listener.c
 *
 * Single-agent flow listener for Olympus.
 *
 * Watches TCP flows whose iperf3 client source port matches --cport and whose
 * current congestion-control name matches the configured Astraea name. The
 * target CC name is intentionally not a command-line argument; set it through
 * SAO_LISTENER_CC or OC_LISTENER_CC, defaulting to "astraea".
 *
 * The listener does not switch congestion-control algorithms. It only
 * duplicates the already-matched socket, enables DeepCC on that socket fd, and
 * forks the selected Python worker.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -pthread -o single_agent_olympus/oc_listener \
 *      single_agent_olympus/oc_listener.c
 *
 * Run:
 *   sudo -E env OC_PYTHON="./venv_training/bin/python" \
 *     single_agent_olympus/oc_listener --cport 23000 \
 *       --worker single_agent_olympus/algorithms/orca_td3/worker.py \
 *       --mode mininet --scan-ms 20 --single-flow 1 --no-state-pipe 1
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
#include <linux/tcp.h>

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
#ifndef TCP_DEEPCC_ENABLE
#define TCP_DEEPCC_ENABLE 44
#endif

typedef struct __attribute__((packed)) {
    uint32_t reserved;
    uint32_t rtt_us;
    uint32_t rttvar_us;
    uint32_t min_rtt_us;
    uint32_t snd_cwnd;
    uint32_t lost;
    uint32_t retrans;
    uint32_t delivered;
    uint64_t delivery_rate;
} oc_state_t;

typedef struct {
    char mode[16];
    char py_worker[PATH_MAX];
    char target_cc[64];
    int  cport;
    int  scan_ms;
    int  ipv4_only;
    int  verbose;
    int  single_flow;
    int  no_state_pipe;
} config_t;

typedef struct {
    char state[32], local[128], peer[128], cc[64], proc[128];
    int  pid, fd, ns_pid;
    unsigned long long ns_ino;
} ss_record_t;

typedef struct flow_worker {
    char  key[512];
    char  local[128];
    char  peer[128];

    int   fd;
    int   pid;
    int   src_fd;
    int   ns_pid;
    long  flow_id;

    pid_t child_pid;
    int   active;
    int   stop_requested;

    int   state_pipe_wr;

    pthread_t thr;
    struct flow_worker *next;
} flow_worker_t;

static volatile sig_atomic_t g_stop = 0;
static pthread_mutex_t       g_workers_mu = PTHREAD_MUTEX_INITIALIZER;
static flow_worker_t        *g_workers = NULL;
static long                  g_next_flow_id = 1;

static config_t g_cfg;

static int any_workers(void);

static void on_sig(int sig) {
    (void)sig;
    g_stop = 1;
}

static void msleep_int(int ms) {
    struct timespec ts = {
        .tv_sec = ms / 1000,
        .tv_nsec = (long)(ms % 1000) * 1000000L
    };
    nanosleep(&ts, NULL);
}

static int pidfd_open_wrap(int pid) {
    return (int)syscall(SYS_pidfd_open, pid, 0);
}

static int pidfd_getfd_wrap(int pidfd, int targetfd) {
    return (int)syscall(SYS_pidfd_getfd, pidfd, targetfd, 0);
}

static int dup_fd_from_pid(int pid, int fd) {
    int pidfd = pidfd_open_wrap(pid);
    if (pidfd < 0) return -1;
    int out = pidfd_getfd_wrap(pidfd, fd);
    close(pidfd);
    return out;
}

static int enable_deepcc(int fd, int val) {
    return setsockopt(fd, IPPROTO_TCP, TCP_DEEPCC_ENABLE, &val, sizeof(val));
}

static int port_from_addr(const char *addr) {
    const char *col = strrchr(addr, ':');
    if (!col) return -1;
    return atoi(col + 1);
}

static void make_key(char *key, size_t sz, const ss_record_t *rec) {
    snprintf(key, sz, "%s->%s", rec->local, rec->peer);
}

static flow_worker_t *find_worker_locked(const char *key) {
    for (flow_worker_t *p = g_workers; p; p = p->next) {
        if (strcmp(p->key, key) == 0) return p;
    }
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
    flow_worker_t **pp = &g_workers;
    flow_worker_t *p;
    while ((p = *pp) != NULL) {
        if (p == victim) {
            *pp = p->next;
            return;
        }
        pp = &p->next;
    }
}

static int fill_state(int fd, oc_state_t *s) {
    struct tcp_info ti;
    socklen_t len = sizeof(ti);
    memset(&ti, 0, len);

    if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &ti, &len) != 0)
        return -1;

    s->reserved      = 0;
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

static pid_t spawn_worker(const config_t *cfg, flow_worker_t *w, int state_rd) {
    pid_t child = fork();
    if (child < 0) return -1;

    if (child == 0) {
        char state_s[32], action_s[32], fd_s[32], flow_s[32], cport_s[32];
        snprintf(state_s, sizeof(state_s), "%d", state_rd);
        snprintf(action_s, sizeof(action_s), "%d", -1);
        snprintf(fd_s, sizeof(fd_s), "%d", w->fd);
        snprintf(flow_s, sizeof(flow_s), "%ld", w->flow_id);
        snprintf(cport_s, sizeof(cport_s), "%d", cfg->cport);

        setenv("OC_STATE_FD", state_s, 1);
        setenv("OC_ACTION_FD", action_s, 1);
        setenv("OC_FLOW_FD", fd_s, 1);
        setenv("OC_FLOW_ID", flow_s, 1);
        setenv("OC_CPORT", cport_s, 1);

        const char *mgr_addr = getenv("OC_MANAGER_ADDR");
        const char *mgr_key = getenv("OC_MANAGER_KEY");
        if (mgr_addr) setenv("OC_MANAGER_ADDR", mgr_addr, 1);
        if (mgr_key) setenv("OC_MANAGER_KEY", mgr_key, 1);

        int cfl;
        if (state_rd >= 0) {
            cfl = fcntl(state_rd, F_GETFD);
            if (cfl >= 0) fcntl(state_rd, F_SETFD, cfl & ~FD_CLOEXEC);
        }
        cfl = fcntl(w->fd, F_GETFD);
        if (cfl >= 0) fcntl(w->fd, F_SETFD, cfl & ~FD_CLOEXEC);

        const char *py = getenv("OC_PYTHON");
        if (!py || !*py) py = "/usr/bin/python3";

        execl(py, py, cfg->py_worker, (char *)NULL);
        perror("execl worker");
        _exit(127);
    }

    return child;
}

static void *flow_thread(void *arg) {
    flow_worker_t *w = (flow_worker_t *)arg;

    enable_deepcc(w->fd, 2);
    fprintf(stderr, "[sao-listener flow %ld] cport=%d cc=%s worker active\n",
            w->flow_id, g_cfg.cport, g_cfg.target_cc);
    fflush(stderr);

    while (!g_stop && !w->stop_requested) {
        int st = 0;
        if (waitpid(w->child_pid, &st, WNOHANG) == w->child_pid) {
            if (g_cfg.verbose) {
                fprintf(stderr, "[sao-listener flow %ld] worker exited\n",
                        w->flow_id);
                fflush(stderr);
            }
            break;
        }

        if (w->state_pipe_wr >= 0) {
            oc_state_t state;
            if (fill_state(w->fd, &state) == 0) {
                ssize_t wr = write(w->state_pipe_wr, &state, sizeof(state));
                (void)wr;
            }
        }

        msleep_int(g_cfg.scan_ms);
    }

    if (w->child_pid > 0) {
        kill(w->child_pid, SIGTERM);
        waitpid(w->child_pid, NULL, 0);
    }

    pthread_mutex_lock(&g_workers_mu);
    w->active = 0;
    remove_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    if (w->state_pipe_wr >= 0) close(w->state_pipe_wr);
    if (w->fd >= 0) close(w->fd);
    free(w);
    return NULL;
}

static int maybe_spawn_flow(const config_t *cfg, const ss_record_t *rec) {
    char key[512];
    make_key(key, sizeof(key), rec);
    if (worker_is_active(key)) return 0;

    int fd = dup_fd_from_pid(rec->pid, rec->fd);
    if (fd < 0) return 0;

    if (enable_deepcc(fd, 2) != 0) {
        close(fd);
        return 0;
    }

    int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);

    int state_pfd[2] = {-1, -1};
    if (!cfg->no_state_pipe) {
        if (pipe2(state_pfd, O_CLOEXEC) != 0) {
            close(fd);
            return 0;
        }
    }

    flow_worker_t *w = calloc(1, sizeof(*w));
    if (!w) {
        if (state_pfd[0] >= 0) close(state_pfd[0]);
        if (state_pfd[1] >= 0) close(state_pfd[1]);
        close(fd);
        return 0;
    }

    snprintf(w->key, sizeof(w->key), "%s", key);
    snprintf(w->local, sizeof(w->local), "%s", rec->local);
    snprintf(w->peer, sizeof(w->peer), "%s", rec->peer);
    w->fd = fd;
    w->pid = rec->pid;
    w->src_fd = rec->fd;
    w->ns_pid = rec->ns_pid;
    w->flow_id = __sync_fetch_and_add(&g_next_flow_id, 1);
    w->child_pid = -1;
    w->active = 1;
    w->stop_requested = 0;
    w->state_pipe_wr = state_pfd[1];

    pthread_mutex_lock(&g_workers_mu);
    if (find_worker_locked(key) != NULL) {
        pthread_mutex_unlock(&g_workers_mu);
        if (state_pfd[0] >= 0) close(state_pfd[0]);
        if (state_pfd[1] >= 0) close(state_pfd[1]);
        close(fd);
        free(w);
        return 0;
    }
    add_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    w->child_pid = spawn_worker(cfg, w, state_pfd[0]);
    if (w->child_pid < 0) {
        perror("[sao-listener] spawn_worker");
        pthread_mutex_lock(&g_workers_mu);
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);
        if (state_pfd[0] >= 0) close(state_pfd[0]);
        if (state_pfd[1] >= 0) close(state_pfd[1]);
        close(fd);
        free(w);
        return 0;
    }

    if (state_pfd[0] >= 0) close(state_pfd[0]);

    fprintf(stderr, "[sao-listener] new flow %ld %s cc=%s pid=%d child=%d\n",
            w->flow_id, key, rec->cc, rec->pid, (int)w->child_pid);
    fflush(stderr);

    if (pthread_create(&w->thr, NULL, flow_thread, w) != 0) {
        perror("[sao-listener] pthread_create");
        kill(w->child_pid, SIGTERM);
        waitpid(w->child_pid, NULL, 0);
        pthread_mutex_lock(&g_workers_mu);
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);
        if (w->state_pipe_wr >= 0) close(w->state_pipe_wr);
        close(fd);
        free(w);
        return 0;
    }
    pthread_detach(w->thr);
    return 1;
}

static int is_state_token(const char *tok) {
    static const char *s[] = {
        "ESTAB", "SYN-SENT", "SYN-RECV", "FIN-WAIT-1", "FIN-WAIT-2",
        "TIME-WAIT", "CLOSE", "CLOSE-WAIT", "LAST-ACK", "LISTEN",
        "CLOSING", "NEW_SYN_RECV", NULL
    };
    for (int i = 0; s[i]; i++) {
        if (strcmp(tok, s[i]) == 0) return 1;
    }
    return 0;
}

static int parse_users(const char *line, char *proc, size_t psz, int *pid, int *fd) {
    const char *p = strstr(line, "users:((");
    if (!p) return -1;
    const char *q1 = strchr(p, '"');
    if (!q1) return -1;
    const char *q2 = strchr(q1 + 1, '"');
    if (!q2) return -1;
    size_t n = (size_t)(q2 - q1 - 1);
    if (n >= psz) n = psz - 1;
    memcpy(proc, q1 + 1, n);
    proc[n] = '\0';
    const char *pidp = strstr(q2, "pid=");
    const char *fdp = strstr(q2, "fd=");
    if (!pidp || !fdp) return -1;
    *pid = atoi(pidp + 4);
    *fd = atoi(fdp + 3);
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
        if (*line == '\0') {
            line = strtok_r(NULL, "\n", &save);
            continue;
        }

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
    if (child < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

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
        else execlp("ss", "ss", "-tinHp", (char *)NULL);
        _exit(122);
    }

    close(pipefd[1]);
    size_t cap = 16384, len = 0;
    char *rbuf = malloc(cap);
    if (!rbuf) {
        close(pipefd[0]);
        return -1;
    }

    for (;;) {
        if (len + 4096 + 1 > cap) {
            cap *= 2;
            char *nb = realloc(rbuf, cap);
            if (!nb) {
                free(rbuf);
                close(pipefd[0]);
                return -1;
            }
            rbuf = nb;
        }
        ssize_t n = read(pipefd[0], rbuf + len, cap - len - 1);
        if (n == 0) break;
        if (n < 0) {
            if (errno == EINTR) continue;
            free(rbuf);
            close(pipefd[0]);
            return -1;
        }
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
        for (int i = 0; i < count; i++) {
            if (inos[i] == ino) {
                exists = 1;
                break;
            }
        }
        if (!exists && count < max_out) {
            pids[count] = pid;
            inos[count] = ino;
            count++;
        }
    }
    closedir(d);
    return count;
}

static int scan_ns_once(const config_t *cfg, int ns_pid,
                        unsigned long long ns_ino) {
    char *ss_text = NULL;
    if (run_ss_in_ns(ns_pid, cfg->ipv4_only, &ss_text) != 0) return 0;

    ss_record_t recs[4096];
    int n = scan_ss_text(ss_text, ns_ino, ns_pid, recs, 4096);
    free(ss_text);

    int spawned = 0;
    for (int i = 0; i < n; i++) {
        if (recs[i].pid <= 0 || recs[i].fd < 0) continue;
        if (strcmp(recs[i].state, "ESTAB") != 0) continue;
        if (port_from_addr(recs[i].local) != cfg->cport) continue;
        if (strcmp(recs[i].cc, cfg->target_cc) != 0) continue;
        spawned += maybe_spawn_flow(cfg, &recs[i]);
        if (cfg->single_flow && spawned > 0) break;
    }
    return spawned;
}

static void scan_loop(const config_t *cfg) {
    while (!g_stop) {
        int spawned = 0;
        if (strcmp(cfg->mode, "mininet") == 0) {
            int pids[1024];
            unsigned long long inos[1024];
            int n = discover_netns(pids, inos, 1024);
            for (int i = 0; i < n; i++) {
                spawned += scan_ns_once(cfg, pids[i], inos[i]);
                if (cfg->single_flow && spawned > 0) break;
            }
        } else {
            spawned += scan_ns_once(cfg, 0, 0);
        }

        if (cfg->single_flow && spawned > 0) {
            fprintf(stderr, "[sao-listener] single-flow mode: attached worker; stopping ss scans\n");
            fflush(stderr);
            while (!g_stop && any_workers()) msleep_int(100);
            break;
        }

        msleep_int(cfg->scan_ms);
    }
}

static void request_stop_all(void) {
    pthread_mutex_lock(&g_workers_mu);
    for (flow_worker_t *p = g_workers; p; p = p->next) {
        p->stop_requested = 1;
    }
    pthread_mutex_unlock(&g_workers_mu);
}

static int any_workers(void) {
    int ret;
    pthread_mutex_lock(&g_workers_mu);
    ret = (g_workers != NULL);
    pthread_mutex_unlock(&g_workers_mu);
    return ret;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --cport PORT --worker SCRIPT\n"
        "          [--mode normal|mininet] [--scan-ms 100]\n"
        "          [--single-flow 0] [--no-state-pipe 0]\n"
        "          [--ipv4-only 1] [--verbose 0]\n"
        "\n"
        "Env:  OC_PYTHON        path to Python interpreter (default /usr/bin/python3)\n"
        "      SAO_LISTENER_CC  target TCP CC name to scan for (default astraea)\n"
        "      OC_LISTENER_CC   fallback target TCP CC name\n",
        prog);
}

int main(int argc, char **argv) {
    memset(&g_cfg, 0, sizeof(g_cfg));
    snprintf(g_cfg.mode, sizeof(g_cfg.mode), "mininet");
    snprintf(g_cfg.target_cc, sizeof(g_cfg.target_cc), "astraea");
    g_cfg.scan_ms = 100;
    g_cfg.ipv4_only = 1;
    g_cfg.verbose = 0;
    g_cfg.single_flow = 0;
    g_cfg.no_state_pipe = 0;
    g_cfg.cport = 0;

    const char *env_cc = getenv("SAO_LISTENER_CC");
    if (!env_cc || !*env_cc) env_cc = getenv("OC_LISTENER_CC");
    if (env_cc && *env_cc) {
        snprintf(g_cfg.target_cc, sizeof(g_cfg.target_cc), "%s", env_cc);
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cport") && i + 1 < argc) {
            g_cfg.cport = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--worker") && i + 1 < argc) {
            snprintf(g_cfg.py_worker, sizeof(g_cfg.py_worker), "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            snprintf(g_cfg.mode, sizeof(g_cfg.mode), "%s", argv[++i]);
        } else if (!strcmp(argv[i], "--scan-ms") && i + 1 < argc) {
            g_cfg.scan_ms = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--single-flow") && i + 1 < argc) {
            g_cfg.single_flow = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--once") && i + 1 < argc) {
            g_cfg.single_flow = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-state-pipe") && i + 1 < argc) {
            g_cfg.no_state_pipe = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ipv4-only") && i + 1 < argc) {
            g_cfg.ipv4_only = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--verbose") && i + 1 < argc) {
            g_cfg.verbose = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (g_cfg.cport <= 0 || !g_cfg.py_worker[0]) {
        usage(argv[0]);
        return 2;
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    fprintf(stderr,
            "[sao-listener] cport=%d worker=%s scan=%dms target_cc=%s "
            "single_flow=%d no_state_pipe=%d\n",
            g_cfg.cport, g_cfg.py_worker, g_cfg.scan_ms, g_cfg.target_cc,
            g_cfg.single_flow, g_cfg.no_state_pipe);
    fflush(stderr);

    scan_loop(&g_cfg);

    request_stop_all();
    while (any_workers()) msleep_int(100);

    return 0;
}
