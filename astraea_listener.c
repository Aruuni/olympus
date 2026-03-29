#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
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

typedef struct {
    char state[32], local[128], peer[128], cc[64], proc[128];
    int pid, fd, ns_pid;
    unsigned long long ns_ino;
} ss_record_t;

typedef struct worker {
    char key[512];
    pid_t child_pid;
    struct worker* next;
} worker_t;

typedef struct flow_record {
    char key[512];
    pid_t child_pid;
    int active;
    struct flow_record* next;
} flow_record_t;

typedef struct {
    char mode[16];
    char cc_name[64];
    char py_script[PATH_MAX];
    char py_config[PATH_MAX];
    char py_model[PATH_MAX];
    int scan_ms, ipv4_only, include_listen, verbose;
} config_t;

static volatile sig_atomic_t g_stop = 0;
static worker_t* g_workers = NULL;
static flow_record_t* g_records = NULL;
static long g_next_flow_id = 1;

static void on_sig(int sig) { (void)sig; g_stop = 1; }

static void msleep_int(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static int pidfd_open_wrap(int pid) { return (int)syscall(SYS_pidfd_open, pid, 0); }
static int pidfd_getfd_wrap(int pidfd, int targetfd) { return (int)syscall(SYS_pidfd_getfd, pidfd, targetfd, 0); }

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

static flow_record_t* find_record(const char* key) {
    for (flow_record_t* p = g_records; p; p = p->next) {
        if (strcmp(p->key, key) == 0) return p;
    }
    return NULL;
}

static int record_is_active(const char* key) {
    flow_record_t* r = find_record(key);
    return r && r->active;
}

static int add_record(const char* key, pid_t child_pid) {
    flow_record_t* r = calloc(1, sizeof(*r));
    if (!r) return -1;
    snprintf(r->key, sizeof(r->key), "%s", key);
    r->child_pid = child_pid;
    r->active = 1;
    r->next = g_records;
    g_records = r;
    return 0;
}

static void mark_record_inactive_by_pid(pid_t child_pid) {
    for (flow_record_t* r = g_records; r; r = r->next) {
        if (r->child_pid == child_pid) {
            r->active = 0;
            return;
        }
    }
}

static void add_worker(const char* key, pid_t child_pid) {
    worker_t* w = calloc(1, sizeof(*w));
    if (!w) return;
    snprintf(w->key, sizeof(w->key), "%s", key);
    w->child_pid = child_pid;
    w->next = g_workers;
    g_workers = w;
}

static void reap_workers(void) {
    worker_t **pp = &g_workers, *p;
    while ((p = *pp) != NULL) {
        int st = 0;
        pid_t rc = waitpid(p->child_pid, &st, WNOHANG);
        if (rc == 0) {
            pp = &p->next;
            continue;
        }
        mark_record_inactive_by_pid(p->child_pid);
        *pp = p->next;
        free(p);
    }
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

static void spawn_worker(const config_t* cfg, const ss_record_t* rec) {
    char key[512];
    make_flow_key(key, sizeof(key), rec);
    if (record_is_active(key)) return;

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

    long flow_id = __sync_fetch_and_add(&g_next_flow_id, 1);
    pid_t child = fork();
    if (child < 0) {
        close(fd);
        return;
    }

    if (child == 0) {
        char fd_s[32], flow_s[32];
        snprintf(fd_s, sizeof(fd_s), "%d", fd);
        snprintf(flow_s, sizeof(flow_s), "%ld", flow_id);

        setenv("ASTRAEA_FLOW_FD", fd_s, 1);
        setenv("ASTRAEA_FLOW_ID", flow_s, 1);
        setenv("ASTRAEA_CONFIG", cfg->py_config, 1);
        setenv("ASTRAEA_MODEL", cfg->py_model, 1);

        const char* py = getenv("ASTRAEA_PYTHON");
        if (!py || !*py) py = "/usr/bin/python3";

        execl(py, py, cfg->py_script, (char*)NULL);
        perror("execl python");
        _exit(127);
    }

    close(fd);
    add_worker(key, child);
    add_record(key, child);

    fprintf(stderr, "[attach] %s ns_pid=%d flow_id=%ld child=%d cc=%s\n",
            key, rec->ns_pid, flow_id, (int)child, cc);
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
        reap_workers();
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

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --script PATH --config PATH --model PATH [--mode normal|mininet]\n"
        "          [--cc-name astraea] [--scan-ms 250] [--ipv4-only 1]\n"
        "          [--include-listen 0] [--verbose 0]\n", prog);
}

int main(int argc, char** argv) {
    config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    snprintf(cfg.mode, sizeof(cfg.mode), "normal");
    snprintf(cfg.cc_name, sizeof(cfg.cc_name), "astraea");
    cfg.scan_ms = 250;
    cfg.ipv4_only = 1;
    cfg.include_listen = 0;
    cfg.verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) snprintf(cfg.mode, sizeof(cfg.mode), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--cc-name") && i + 1 < argc) snprintf(cfg.cc_name, sizeof(cfg.cc_name), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--script") && i + 1 < argc) snprintf(cfg.py_script, sizeof(cfg.py_script), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) snprintf(cfg.py_config, sizeof(cfg.py_config), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) snprintf(cfg.py_model, sizeof(cfg.py_model), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--scan-ms") && i + 1 < argc) cfg.scan_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ipv4-only") && i + 1 < argc) cfg.ipv4_only = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--include-listen") && i + 1 < argc) cfg.include_listen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose") && i + 1 < argc) cfg.verbose = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }

    if (!cfg.py_script[0] || !cfg.py_config[0] || !cfg.py_model[0]) {
        usage(argv[0]);
        return 2;
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    scan_loop(&cfg);

    while (g_workers) {
        reap_workers();
        msleep_int(100);
    }
    return 0;
}