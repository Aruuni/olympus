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
#include <sys/uio.h>

#ifndef TCP_CONGESTION
#define TCP_CONGESTION 13
#endif
#ifndef TCP_DEEPCC_ENABLE
#define TCP_DEEPCC_ENABLE 44
#endif
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif



#ifndef NETLINK_TEST
#define NETLINK_TEST 25
#endif

#ifndef COMM_SELECT_ARM
#define COMM_SELECT_ARM 2
#endif

#define COMM_BEGIN 1

#define CUBIC      0
#define HYBLA      1
#define BBR1       2
#define WESTWOOD   3
#define VENO       4
#define VEGAS      5
#define YEAH       6
#define BIC        7
#define HTCP       8
#define HIGHSPEED  9
#define ILLINOIS   10
#define CDG        11
#define BBR3       12
#define ASTRAEA    13


typedef struct {
    char state[32], local[128], peer[128], cc[64], proc[128];
    int pid, fd, ns_pid;
    unsigned long long ns_ino;
} ss_record_t;

typedef struct {
    char mode[16];
    char cc_name[64];
    char py_script[PATH_MAX];
    char py_config[PATH_MAX];
    char py_model[PATH_MAX];
    int scan_ms, ipv4_only, include_listen, verbose;
} config_t;

typedef struct flow_worker {
    char key[512];
    char local[128];
    char peer[128];

    int fd;
    int pid;
    int src_fd;
    int ns_pid;
    long flow_id;

    pid_t child_pid;
    int active;
    int stop_requested;

    int control_enabled;
    int control_sent;

    int ctrl_pipe_rd;
    int ctrl_pipe_wr;

    pthread_t thr;
    struct flow_worker* next;
} flow_worker_t;

static volatile sig_atomic_t g_stop = 0;
static pthread_mutex_t g_workers_mu = PTHREAD_MUTEX_INITIALIZER;
static flow_worker_t* g_workers = NULL;
static long g_next_flow_id = 1;

static void on_sig(int sig) { (void)sig; g_stop = 1; }

static void msleep_int(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static int pidfd_open_wrap(int pid) { return (int)syscall(SYS_pidfd_open, pid, 0); }
static int pidfd_getfd_wrap(int pidfd, int targetfd) { return (int)syscall(SYS_pidfd_getfd, pidfd, targetfd, 0); }

static int set_cc_algorithm_fd(int fd, const char *cc_name) {
    if (fd < 0 || cc_name == NULL || *cc_name == '\0') {
        errno = EINVAL;
        return -1;
    }

    /* Linux expects the congestion control name as a NUL-terminated string. */
    socklen_t optlen = (socklen_t)(strlen(cc_name) + 1);

    if (setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, cc_name, optlen) != 0) {
        return -1;
    }

    return 0;
}

static int mutant_send_switch_cmd(int nl_fd, uint32_t proto_id) {
    struct {
        struct nlmsghdr nlh;
        char payload[32];
    } req;

    struct sockaddr_nl dst;
    memset(&req, 0, sizeof(req));
    memset(&dst, 0, sizeof(dst));

    dst.nl_family = AF_NETLINK;

    req.nlh.nlmsg_len   = NLMSG_LENGTH((uint32_t)(strlen("SENDING ACTION") + 1));
    req.nlh.nlmsg_type  = 0;
    req.nlh.nlmsg_flags = COMM_SELECT_ARM;   /* 2 */
    req.nlh.nlmsg_seq   = proto_id;          /* CUBIC=0, VEGAS=5 */
    req.nlh.nlmsg_pid   = (uint32_t)getpid();

    memcpy(NLMSG_DATA(&req.nlh), "SENDING ACTION", strlen("SENDING ACTION") + 1);
    fprintf(stderr, "Sending switch command proto_id=%u\n", proto_id);
    if (sendto(nl_fd,
               &req,
               req.nlh.nlmsg_len,
               0,
               (struct sockaddr *)&dst,
               sizeof(dst)) < 0) {
        return -1;
    }

    return 0;
}

static int mutant_send_begin(int nl_fd) {
    struct {
        struct nlmsghdr nlh;
        char payload[32];
    } req;

    struct sockaddr_nl dst;
    memset(&req, 0, sizeof(req));
    memset(&dst, 0, sizeof(dst));

    dst.nl_family = AF_NETLINK;

    req.nlh.nlmsg_len   = NLMSG_LENGTH((uint32_t)(strlen("INIT_COMMUNICATION") + 1));
    req.nlh.nlmsg_type  = 0;
    req.nlh.nlmsg_flags = COMM_BEGIN;   /* 1 */
    req.nlh.nlmsg_seq   = 0;
    req.nlh.nlmsg_pid   = (uint32_t)syscall(SYS_gettid);

    memcpy(NLMSG_DATA(&req.nlh), "INIT_COMMUNICATION",
           strlen("INIT_COMMUNICATION") + 1);

    if (sendto(nl_fd, &req, req.nlh.nlmsg_len, 0,
               (struct sockaddr *)&dst, sizeof(dst)) < 0) {
        return -1;
    }

    return 0;
}

static int dup_fd_from_pid(int pid, int fd) {
    int pidfd = pidfd_open_wrap(pid);
    if (pidfd < 0) return -1;
    int out = pidfd_getfd_wrap(pidfd, fd);
    close(pidfd);
    return out;
}

static int get_cc_name(int fd, char* out, socklen_t outlen) {
    memset(out, 0, outlen);
    socklen_t len = outlen;
    if (getsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, out, &len) != 0) return -1;
    out[outlen - 1] = '\0';
    return 0;
}

static int enable_deepcc_fd(int fd, int val) {
    return setsockopt(fd, IPPROTO_TCP, TCP_DEEPCC_ENABLE, &val, sizeof(val));
}

static void make_flow_key(char* key, size_t key_sz, const ss_record_t* rec) {
    snprintf(key, key_sz, "%s->%s", rec->local, rec->peer);
}

static flow_worker_t* find_worker_locked(const char* key) {
    for (flow_worker_t* p = g_workers; p; p = p->next) {
        if (strcmp(p->key, key) == 0) return p;
    }
    return NULL;
}

static int worker_is_active(const char* key) {
    int ret = 0;
    pthread_mutex_lock(&g_workers_mu);
    flow_worker_t* w = find_worker_locked(key);
    ret = (w && w->active);
    pthread_mutex_unlock(&g_workers_mu);
    return ret;
}

static void add_worker_locked(flow_worker_t* w) {
    w->next = g_workers;
    g_workers = w;
}

static void remove_worker_locked(flow_worker_t* victim) {
    flow_worker_t **pp = &g_workers, *p;
    while ((p = *pp) != NULL) {
        if (p == victim) {
            *pp = p->next;
            return;
        }
        pp = &p->next;
    }
}

static pid_t spawn_python_child(const config_t* cfg, flow_worker_t* w) {
    pid_t child = fork();
    if (child < 0) return -1;

    if (child == 0) {
        char fd_s[32], flow_s[32], ctrl_s[32];
        snprintf(fd_s, sizeof(fd_s), "%d", w->fd);
        snprintf(flow_s, sizeof(flow_s), "%ld", w->flow_id);
        snprintf(ctrl_s, sizeof(ctrl_s), "%d", w->ctrl_pipe_rd);

        setenv("ASTRAEA_FLOW_FD", fd_s, 1);
        setenv("ASTRAEA_FLOW_ID", flow_s, 1);
        setenv("ASTRAEA_CONFIG", cfg->py_config, 1);
        setenv("ASTRAEA_MODEL", cfg->py_model, 1);
        setenv("ASTRAEA_CONTROL_FD", ctrl_s, 1);

        close(w->ctrl_pipe_wr);

        const char* py = getenv("ASTRAEA_PYTHON");
        if (!py || !*py) py = "/usr/bin/python3";

        execl(py, py, cfg->py_script, (char*)NULL);
        perror("execl python");
        _exit(127);
    }

    close(w->ctrl_pipe_rd);
    w->ctrl_pipe_rd = -1;
    return child;
}

static void* flow_thread(void* arg) {
    flow_worker_t* w = (flow_worker_t*)arg;
    extern config_t g_cfg;

    w->child_pid = spawn_python_child(&g_cfg, w);
    if (w->child_pid < 0) {
        fprintf(stderr, "[flow %ld] failed to spawn python child\n", w->flow_id);
        fflush(stderr);

        pthread_mutex_lock(&g_workers_mu);
        w->active = 0;
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);

        if (w->ctrl_pipe_rd >= 0) close(w->ctrl_pipe_rd);
        if (w->ctrl_pipe_wr >= 0) close(w->ctrl_pipe_wr);
        if (w->fd >= 0) close(w->fd);
        free(w);
        return NULL;
    }


    fprintf(stderr, "[attach] %s ns_pid=%d flow_id=%ld child=%d cc=%s\n",
            w->key, w->ns_pid, w->flow_id, (int)w->child_pid, g_cfg.cc_name);
    fflush(stderr);
    int nl_fd = -1;

    
    int counter = 0;
    int mutant_on = 0;
    int cur_idx = 0;

    /* adjust these enum names to match your mutant command IDs */
    static const int proto_ids[] = { ASTRAEA, CUBIC, VEGAS };
    static const char *proto_names[] = { "astraea", "cubic", "vegas" };
    static const int proto_count = sizeof(proto_ids) / sizeof(proto_ids[0]);

    char cc[64];

    if (get_cc_name(w->fd, cc, sizeof(cc)) == 0) {
        fprintf(stderr, "[flow %ld] socket cc is now: %s\n", w->flow_id, cc);
        fflush(stderr);
    }

    while (!g_stop && !w->stop_requested) {
        int st = 0;
        pid_t rc = waitpid(w->child_pid, &st, WNOHANG);
        if (rc == w->child_pid) {
            if (g_cfg.verbose) {
                fprintf(stderr, "[flow %ld] child exited pid=%d\n",
                        w->flow_id, (int)w->child_pid);
                fflush(stderr);
            }
            break;
        }
        // set_cc_algorithm_fd(w->fd, "astraea");
        // w->control_enabled = 1;
        /* switch every 5 seconds: 100 ms sleep * 50 = 5 s */
        if (!mutant_on) {
            struct sockaddr_nl addr;
            memset(&addr, 0, sizeof(addr));

            if (set_cc_algorithm_fd(w->fd, "mutant") != 0) {
                perror("set_cc_algorithm_fd(mutant)");
            } else {
                if (get_cc_name(w->fd, cc, sizeof(cc)) == 0) {
                    fprintf(stderr, "[flow %ld] socket cc is now: %s\n", w->flow_id, cc);
                    fflush(stderr);
                }
            }

            nl_fd = socket(AF_NETLINK, SOCK_RAW, NETLINK_TEST);
            if (nl_fd < 0) {
                perror("socket(AF_NETLINK)");
            } else {
                addr.nl_family = AF_NETLINK;
                addr.nl_pid = (uint32_t)getpid();

                if (bind(nl_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
                    perror("bind(AF_NETLINK)");
                    close(nl_fd);
                    nl_fd = -1;
                } else {
                    if (mutant_send_begin(nl_fd) != 0) {
                        perror("mutant_send_begin");
                    } else {
                        mutant_on = 1;

                        /* set initial protocol immediately */
                        if (mutant_send_switch_cmd(nl_fd, proto_ids[cur_idx]) != 0) {
                            perror("mutant_send_switch_cmd(initial)");
                        } else {
                            fprintf(stderr, "[flow %ld] switched to %s\n",
                                    w->flow_id, proto_names[cur_idx]);
                            fflush(stderr);
                        }
                        enable_deepcc_fd(w->fd, 2); 
                        /* python controls only Astraea */
                        w->control_enabled = (proto_ids[cur_idx] == ASTRAEA);
                    }
                }
            }

            counter = 0;
        }

        if (mutant_on && counter >= 100) {
            cur_idx = (cur_idx + 1) % proto_count;

            if (mutant_send_switch_cmd(nl_fd, proto_ids[cur_idx]) != 0) {
                perror("mutant_send_switch_cmd");
            } else {
                fprintf(stderr, "[flow %ld] switched to %s\n",
                        w->flow_id, proto_names[cur_idx]);
                fflush(stderr);
            }

            /* give control to python only for Astraea */
            w->control_enabled = (proto_ids[cur_idx] == ASTRAEA);

            counter = 0;
        }
        


        
        if (w->control_enabled != w->control_sent) {
            char c = w->control_enabled ? '1' : '0';

            if (write(w->ctrl_pipe_wr, &c, 1) == 1) {
                w->control_sent = w->control_enabled;
                fprintf(stderr, "[flow %ld] sent control=%d to python\n",
                        w->flow_id, w->control_enabled);
                fflush(stderr);
            } else {
                perror("write control pipe");
            }
        }
        msleep_int(100);
        counter++;
    }

    if (nl_fd >= 0) close(nl_fd);

    if (w->child_pid > 0) {
        kill(w->child_pid, SIGTERM);
        waitpid(w->child_pid, NULL, 0);
    }

    pthread_mutex_lock(&g_workers_mu);
    w->active = 0;
    remove_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    if (w->ctrl_pipe_rd >= 0) close(w->ctrl_pipe_rd);
    if (w->ctrl_pipe_wr >= 0) close(w->ctrl_pipe_wr);
    if (w->fd >= 0) close(w->fd);
    free(w);
    return NULL;
}




static int parse_users_blob(const char* line, char* proc, size_t proc_sz, int* pid, int* fd) {
    const char* p = strstr(line, "users:((");
    if (!p) return -1;
    const char* q1 = strchr(p, '"'); if (!q1) return -1;
    const char* q2 = strchr(q1 + 1, '"'); if (!q2) return -1;
    size_t n = (size_t)(q2 - q1 - 1); if (n >= proc_sz) n = proc_sz - 1;
    memcpy(proc, q1 + 1, n); proc[n] = '\0';
    const char* pidp = strstr(q2, "pid=");
    const char* fdp  = strstr(q2, "fd=");
    if (!pidp || !fdp) return -1;
    *pid = atoi(pidp + 4);
    *fd  = atoi(fdp + 3);
    return 0;
}

static int is_state_token(const char* tok) {
    static const char* s[] = {
        "ESTAB","SYN-SENT","SYN-RECV","FIN-WAIT-1","FIN-WAIT-2","TIME-WAIT",
        "CLOSE","CLOSE-WAIT","LAST-ACK","LISTEN","CLOSING","NEW_SYN_RECV",NULL
    };
    for (int i = 0; s[i]; i++) if (strcmp(tok, s[i]) == 0) return 1;
    return 0;
}

static int scan_ss_text(const char* text, unsigned long long ns_ino, int ns_pid,
                        ss_record_t* out, int max_out) {
    int count = 0, have_cur = 0;
    char* buf = strdup(text);
    if (!buf) return 0;
    ss_record_t cur; memset(&cur, 0, sizeof(cur));
    char *save = NULL, *line = strtok_r(buf, "\n", &save);

    while (line) {
        while (*line && isspace((unsigned char)*line)) line++;
        if (*line == '\0') { line = strtok_r(NULL, "\n", &save); continue; }

        char first[64] = {0};
        sscanf(line, "%63s", first);

        if (is_state_token(first)) {
            memset(&cur, 0, sizeof(cur));
            cur.ns_ino = ns_ino;
            cur.ns_pid = ns_pid;
            sscanf(line, "%31s %*s %*s %127s %127s", cur.state, cur.local, cur.peer);
            parse_users_blob(line, cur.proc, sizeof(cur.proc), &cur.pid, &cur.fd);
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

static int run_ss_in_ns_pid(int target_pid, int ipv4_only, char** out_text) {
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
        if (ipv4_only) execlp("ss", "ss", "-tinHp4", (char*)NULL);
        else execlp("ss", "ss", "-tinHp", (char*)NULL);
        _exit(122);
    }

    close(pipefd[1]);
    size_t cap = 16384, len = 0;
    char* buf = malloc(cap);
    if (!buf) { close(pipefd[0]); return -1; }

    for (;;) {
        if (len + 4096 + 1 > cap) {
            cap *= 2;
            char* nb = realloc(buf, cap);
            if (!nb) { free(buf); close(pipefd[0]); return -1; }
            buf = nb;
        }
        ssize_t n = read(pipefd[0], buf + len, cap - len - 1);
        if (n == 0) break;
        if (n < 0) { if (errno == EINTR) continue; free(buf); close(pipefd[0]); return -1; }
        len += (size_t)n;
    }
    close(pipefd[0]);
    buf[len] = '\0';
    waitpid(child, NULL, 0);
    *out_text = buf;
    return 0;
}

static int discover_netns(int* pids, unsigned long long* inos, int max_out) {
    DIR* d = opendir("/proc");
    if (!d) return 0;
    int count = 0;
    struct dirent* de;

    while ((de = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) continue;
        int pid = atoi(de->d_name);
        char ns_path[64];
        struct stat st;
        snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/net", pid);
        if (stat(ns_path, &st) != 0) continue;
        unsigned long long ino = (unsigned long long)st.st_ino;
        int exists = 0;
        for (int i = 0; i < count; i++) if (inos[i] == ino) { exists = 1; break; }
        if (!exists && count < max_out) {
            pids[count] = pid;
            inos[count] = ino;
            count++;
        }
    }

    closedir(d);
    return count;
}

config_t g_cfg;

static void spawn_worker(const config_t* cfg, const ss_record_t* rec) {
    char key[512];
    make_flow_key(key, sizeof(key), rec);
    if (worker_is_active(key)) return;

    int fd = dup_fd_from_pid(rec->pid, rec->fd);
    if (fd < 0) return;

    char cc[64];
    if (get_cc_name(fd, cc, sizeof(cc)) != 0 || strcmp(cc, cfg->cc_name) != 0) {
        close(fd);
        return;
    }

    if (enable_deepcc_fd(fd, 2) != 0) {
        close(fd);
        return;
    }

    int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC);

    int pfd[2];
    if (pipe(pfd) != 0) {
        close(fd);
        return;
    }

    flow_worker_t* w = calloc(1, sizeof(*w));
    if (!w) {
        close(pfd[0]);
        close(pfd[1]);
        close(fd);
        return;
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

    w->control_enabled = 0;
    w->control_sent = -1;

    w->ctrl_pipe_rd = pfd[0];
    w->ctrl_pipe_wr = pfd[1];

    pthread_mutex_lock(&g_workers_mu);
    if (find_worker_locked(key) != NULL) {
        pthread_mutex_unlock(&g_workers_mu);
        close(w->ctrl_pipe_rd);
        close(w->ctrl_pipe_wr);
        close(fd);
        free(w);
        return;
    }
    add_worker_locked(w);
    pthread_mutex_unlock(&g_workers_mu);

    if (pthread_create(&w->thr, NULL, flow_thread, w) != 0) {
        pthread_mutex_lock(&g_workers_mu);
        remove_worker_locked(w);
        pthread_mutex_unlock(&g_workers_mu);

        close(w->ctrl_pipe_rd);
        close(w->ctrl_pipe_wr);
        close(fd);
        free(w);
        return;
    }

    pthread_detach(w->thr);
}

static void scan_namespace_once(const config_t* cfg, int ns_pid, unsigned long long ns_ino) {
    char* ss_text = NULL;
    if (run_ss_in_ns_pid(ns_pid, cfg->ipv4_only, &ss_text) != 0) return;

    ss_record_t recs[4096];
    int n = scan_ss_text(ss_text, ns_ino, ns_pid, recs, 4096);
    free(ss_text);

    for (int i = 0; i < n; i++) {
        if (recs[i].pid <= 0 || recs[i].fd < 0) continue;
        if (strcmp(recs[i].cc, cfg->cc_name) != 0) continue;
        if (!cfg->include_listen && strcmp(recs[i].state, "LISTEN") == 0) continue;
        spawn_worker(cfg, &recs[i]);
    }
}

static void scan_loop(const config_t* cfg) {
    while (!g_stop) {
        if (strcmp(cfg->mode, "mininet") == 0) {
            int pids[1024];
            unsigned long long inos[1024];
            int n = discover_netns(pids, inos, 1024);
            for (int i = 0; i < n; i++) scan_namespace_once(cfg, pids[i], inos[i]);
        } else {
            scan_namespace_once(cfg, 0, 0);
        }
        msleep_int(cfg->scan_ms);
    }
}

static void request_stop_all_workers(void) {
    pthread_mutex_lock(&g_workers_mu);
    for (flow_worker_t* p = g_workers; p; p = p->next) {
        p->stop_requested = 1;
    }
    pthread_mutex_unlock(&g_workers_mu);
}

static int any_workers_left(void) {
    int ret;
    pthread_mutex_lock(&g_workers_mu);
    ret = (g_workers != NULL);
    pthread_mutex_unlock(&g_workers_mu);
    return ret;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --script PATH --config PATH --model PATH [--mode normal|mininet]\n"
        "          [--cc-name astraea] [--scan-ms 250] [--ipv4-only 1]\n"
        "          [--include-listen 0] [--verbose 0]\n", prog);
}

int main(int argc, char** argv) {
    memset(&g_cfg, 0, sizeof(g_cfg));
    snprintf(g_cfg.mode, sizeof(g_cfg.mode), "normal");
    snprintf(g_cfg.cc_name, sizeof(g_cfg.cc_name), "astraea");
    g_cfg.scan_ms = 250;
    g_cfg.ipv4_only = 1;
    g_cfg.include_listen = 0;
    g_cfg.verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) snprintf(g_cfg.mode, sizeof(g_cfg.mode), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--cc-name") && i + 1 < argc) snprintf(g_cfg.cc_name, sizeof(g_cfg.cc_name), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--script") && i + 1 < argc) snprintf(g_cfg.py_script, sizeof(g_cfg.py_script), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) snprintf(g_cfg.py_config, sizeof(g_cfg.py_config), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) snprintf(g_cfg.py_model, sizeof(g_cfg.py_model), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--scan-ms") && i + 1 < argc) g_cfg.scan_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ipv4-only") && i + 1 < argc) g_cfg.ipv4_only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--include-listen") && i + 1 < argc) g_cfg.include_listen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose") && i + 1 < argc) g_cfg.verbose = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }

    if (!g_cfg.py_script[0] || !g_cfg.py_config[0] || !g_cfg.py_model[0]) {
        usage(argv[0]);
        return 2;
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    scan_loop(&g_cfg);

    request_stop_all_workers();
    while (any_workers_left()) {
        msleep_int(100);
    }

    return 0;
}