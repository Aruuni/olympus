/*
 * single_agent_olympus/oc_bridge.c
 *
 * Single-agent flow listener for Olympus.
 *
 * Watches TCP flows whose iperf3 client source port matches --cport and whose
 * current congestion-control name matches the configured Astraea name. The
 * target CC name is intentionally not a command-line argument; set it through
 * SAO_LISTENER_CC or OC_LISTENER_CC, defaulting to "astraea".
 *
 * The listener does not switch congestion-control algorithms. Once it finds a
 * matching socket it duplicates the fd, enables DeepCC on it, and then
 * exec()s into the selected Python worker -- the listener process *becomes*
 * the worker. The inherited socket fd and OC_* env vars carry across the exec;
 * the worker reads/writes TCP state directly via getsockopt/setsockopt.
 *
 * Because exec() replaces the process image, the listener handles exactly one
 * flow: the scan loop runs until the first match, then hands off for good.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -o single_agent_olympus/oc_bridge \
 *      single_agent_olympus/oc_bridge.c
 *
 * Run:
 *   sudo -E env OC_PYTHON="./venv_training/bin/python" \
 *     single_agent_olympus/oc_bridge --cport 23000 \
 *       --worker single_agent_olympus/algorithms/orca_td3/worker.py \
 *       --mode mininet --scan-ms 20
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
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

#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
#ifndef TCP_DEEPCC_ENABLE
#define TCP_DEEPCC_ENABLE 44
#endif

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

static volatile sig_atomic_t g_stop = 0;
static long                  g_next_flow_id = 1;

static config_t g_cfg;

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

/*
 * Replace the listener process image with the Python worker. The duplicated
 * socket fd is kept open across the exec (FD_CLOEXEC cleared) and its number,
 * along with the other OC_* values, is passed through the environment. Does
 * not return on success.
 */
static void exec_into_worker(const config_t *cfg, int flow_fd,
                             const ss_record_t *rec) {
    char fd_s[32], flow_s[32], cport_s[32];
    long flow_id = __sync_fetch_and_add(&g_next_flow_id, 1);

    snprintf(fd_s, sizeof(fd_s), "%d", flow_fd);
    snprintf(flow_s, sizeof(flow_s), "%ld", flow_id);
    snprintf(cport_s, sizeof(cport_s), "%d", cfg->cport);

    /* No state pipe and no action pipe: the worker reads state directly. */
    setenv("OC_STATE_FD", "-1", 1);
    setenv("OC_ACTION_FD", "-1", 1);
    setenv("OC_FLOW_FD", fd_s, 1);
    setenv("OC_FLOW_ID", flow_s, 1);
    setenv("OC_CPORT", cport_s, 1);

    /* deepcc already enabled by the caller; just keep the fd across exec. */
    int cfl = fcntl(flow_fd, F_GETFD);
    if (cfl >= 0) fcntl(flow_fd, F_SETFD, cfl & ~FD_CLOEXEC);

    fprintf(stderr,
            "[sao-listener] exec-into-worker flow %ld %s->%s cc=%s fd=%d\n",
            flow_id, rec->local, rec->peer, rec->cc, flow_fd);
    fflush(stderr);

    const char *py = getenv("OC_PYTHON");
    if (!py || !*py) py = "/usr/bin/python3";

    execl(py, py, cfg->py_worker, (char *)NULL);
    perror("execl worker");
    _exit(127);
}

static int maybe_spawn_flow(const config_t *cfg, const ss_record_t *rec) {
    int fd = dup_fd_from_pid(rec->pid, rec->fd);
    if (fd < 0) return 0;

    if (enable_deepcc(fd, 2) != 0) {
        close(fd);
        return 0;
    }

    exec_into_worker(cfg, fd, rec);  /* replaces this process; never returns */

    /* Only reached if execl failed. */
    close(fd);
    return 0;
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

    for (int i = 0; i < n; i++) {
        if (recs[i].pid <= 0 || recs[i].fd < 0) continue;
        if (strcmp(recs[i].state, "ESTAB") != 0) continue;
        if (port_from_addr(recs[i].local) != cfg->cport) continue;
        if (strcmp(recs[i].cc, cfg->target_cc) != 0) continue;
        /* On success this exec()s into the worker and never returns. */
        maybe_spawn_flow(cfg, &recs[i]);
    }
    return 0;
}

static void scan_loop(const config_t *cfg) {
    while (!g_stop) {
        if (strcmp(cfg->mode, "mininet") == 0) {
            int pids[1024];
            unsigned long long inos[1024];
            int n = discover_netns(pids, inos, 1024);
            for (int i = 0; i < n; i++) {
                scan_ns_once(cfg, pids[i], inos[i]);
            }
        } else {
            scan_ns_once(cfg, 0, 0);
        }

        msleep_int(cfg->scan_ms);
    }
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
            /* Accepted for compatibility; exec-into-worker is always
             * single-flow because exec() ends the scan loop. */
            g_cfg.single_flow = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--once") && i + 1 < argc) {
            g_cfg.single_flow = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--no-state-pipe") && i + 1 < argc) {
            /* Accepted for compatibility; the exec path never uses a pipe. */
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
            "[sao-listener] cport=%d worker=%s scan=%dms target_cc=%s\n",
            g_cfg.cport, g_cfg.py_worker, g_cfg.scan_ms, g_cfg.target_cc);
    fflush(stderr);

    scan_loop(&g_cfg);

    return 0;
}
