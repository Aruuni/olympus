#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <linux/tcp.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifndef TCP_CWND
#define TCP_CWND 43
#endif

#ifndef TCP_DEEPCC_ENABLE
#define TCP_DEEPCC_ENABLE 44
#endif

#ifndef TCP_DEEPCC_INFO
#define TCP_DEEPCC_INFO 46
#endif

#ifndef SO_MAX_PACING_RATE
#define SO_MAX_PACING_RATE 47
#endif

typedef uint32_t u32;
typedef uint64_t u64;

struct TCPDeepCCInfo {
    u32 min_rtt;
    u32 avg_urtt;
    u32 cnt;
    u64 avg_thr;
    u32 thr_cnt;
    u32 cwnd;
    u32 pacing_rate;
    u32 lost_bytes;
    u32 srtt_us;
    u32 snd_ssthresh;
    u32 packets_out;
    u32 retrans_out;
    u32 max_packets_out;
    u32 mss;
};

static int dict_set_u64(PyObject* d, const char* k, uint64_t v) {
    PyObject* obj = PyLong_FromUnsignedLongLong((unsigned long long)v);
    if (!obj) return -1;
    int rc = PyDict_SetItemString(d, k, obj);
    Py_DECREF(obj);
    return rc;
}

static int dict_set_u32(PyObject* d, const char* k, uint32_t v) {
    PyObject* obj = PyLong_FromUnsignedLong((unsigned long)v);
    if (!obj) return -1;
    int rc = PyDict_SetItemString(d, k, obj);
    Py_DECREF(obj);
    return rc;
}

static int pydeepcc_to_dict(PyObject* d, const struct TCPDeepCCInfo* info) {
    if (dict_set_u32(d, "struct_size", (uint32_t)sizeof(*info)) < 0) return -1;

    if (dict_set_u32(d, "min_rtt", info->min_rtt) < 0) return -1;
    if (dict_set_u32(d, "avg_urtt", info->avg_urtt) < 0) return -1;
    if (dict_set_u32(d, "cnt", info->cnt) < 0) return -1;
    if (dict_set_u64(d, "avg_thr", info->avg_thr) < 0) return -1;
    if (dict_set_u32(d, "thr_cnt", info->thr_cnt) < 0) return -1;
    if (dict_set_u32(d, "cwnd", info->cwnd) < 0) return -1;
    if (dict_set_u32(d, "pacing_rate", info->pacing_rate) < 0) return -1;
    if (dict_set_u32(d, "loss_bytes", info->lost_bytes) < 0) return -1;
    if (dict_set_u32(d, "srtt_us", info->srtt_us) < 0) return -1;
    if (dict_set_u32(d, "snd_ssthresh", info->snd_ssthresh) < 0) return -1;
    if (dict_set_u32(d, "packets_out", info->packets_out) < 0) return -1;
    if (dict_set_u32(d, "retrans_out", info->retrans_out) < 0) return -1;
    if (dict_set_u32(d, "max_packets_out", info->max_packets_out) < 0) return -1;
    if (dict_set_u32(d, "mss_cache", info->mss) < 0) return -1;

    return 0;
}

static PyObject* py_enable_deepcc(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* sock_or_fd = NULL;
    int val = 2;

    if (!PyArg_ParseTuple(args, "O|i", &sock_or_fd, &val)) return NULL;

    int fd = PyObject_AsFileDescriptor(sock_or_fd);
    if (fd < 0) return NULL;

    if (setsockopt(fd, IPPROTO_TCP, TCP_DEEPCC_ENABLE, &val, (socklen_t)sizeof(val)) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* py_get_tcp_deepcc_info(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* sock_or_fd = NULL;

    if (!PyArg_ParseTuple(args, "O", &sock_or_fd)) return NULL;

    int fd = PyObject_AsFileDescriptor(sock_or_fd);
    if (fd < 0) return NULL;

    struct TCPDeepCCInfo info;
    memset(&info, 0, sizeof(info));
    socklen_t len = (socklen_t)sizeof(info);

    if (getsockopt(fd, IPPROTO_TCP, TCP_DEEPCC_INFO, &info, &len) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    PyObject* d = PyDict_New();
    if (!d) return NULL;

    if (dict_set_u32(d, "info_len", (uint32_t)len) < 0) {
        Py_DECREF(d);
        return NULL;
    }

    if (pydeepcc_to_dict(d, &info) != 0) {
        Py_DECREF(d);
        return NULL;
    }

    return d;
}

static PyObject* py_set_cwnd(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* sock_or_fd = NULL;
    unsigned long long cwnd_ull = 0;

    if (!PyArg_ParseTuple(args, "OK", &sock_or_fd, &cwnd_ull)) return NULL;

    int fd = PyObject_AsFileDescriptor(sock_or_fd);
    if (fd < 0) return NULL;

    if (cwnd_ull > 0xFFFFFFFFULL) {
        PyErr_SetString(PyExc_ValueError, "cwnd must fit in uint32");
        return NULL;
    }

    uint32_t cwnd = (uint32_t)cwnd_ull;

    if (setsockopt(fd, IPPROTO_TCP, TCP_CWND, &cwnd, (socklen_t)sizeof(cwnd)) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* py_set_pacing_rate(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* sock_or_fd = NULL;
    unsigned long long rate = 0;

    if (!PyArg_ParseTuple(args, "OK", &sock_or_fd, &rate)) return NULL;

    int fd = PyObject_AsFileDescriptor(sock_or_fd);
    if (fd < 0) return NULL;

    uint64_t v = (uint64_t)rate;
    if (setsockopt(fd, SOL_SOCKET, SO_MAX_PACING_RATE, &v, (socklen_t)sizeof(v)) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* py_get_tcp_getsockopt_info(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* sock_or_fd = NULL;

    if (!PyArg_ParseTuple(args, "O", &sock_or_fd)) return NULL;

    int fd = PyObject_AsFileDescriptor(sock_or_fd);
    if (fd < 0) return NULL;

    struct tcp_info info;
    memset(&info, 0, sizeof(info));
    socklen_t len = (socklen_t)sizeof(info);

    if (getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &len) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    PyObject* d = PyDict_New();
    if (!d) return NULL;

    if (dict_set_u32(d, "info_len", (uint32_t)len) < 0) goto error;
    if (dict_set_u32(d, "struct_size", (uint32_t)sizeof(info)) < 0) goto error;

    if (dict_set_u32(d, "state", info.tcpi_state) < 0) goto error;
    if (dict_set_u32(d, "ca_state", info.tcpi_ca_state) < 0) goto error;
    if (dict_set_u32(d, "rtt", info.tcpi_rtt) < 0) goto error;
    if (dict_set_u32(d, "rttvar", info.tcpi_rttvar) < 0) goto error;
    if (dict_set_u32(d, "rto", info.tcpi_rto) < 0) goto error;
    if (dict_set_u32(d, "ato", info.tcpi_ato) < 0) goto error;
#ifdef tcpi_snd_cwnd
    if (dict_set_u32(d, "snd_cwnd", info.tcpi_snd_cwnd) < 0) goto error;
#endif
#ifdef tcpi_unacked
    if (dict_set_u32(d, "unacked", info.tcpi_unacked) < 0) goto error;
#endif
#ifdef tcpi_lost
    if (dict_set_u32(d, "lost", info.tcpi_lost) < 0) goto error;
#endif
#ifdef tcpi_delivered
    if (dict_set_u32(d, "delivered", info.tcpi_delivered) < 0) goto error;
#endif
#ifdef tcpi_bytes_acked
    if (dict_set_u64(d, "bytes_acked", info.tcpi_bytes_acked) < 0) goto error;
#endif
#ifdef tcpi_bytes_sent
    if (dict_set_u64(d, "bytes_sent", info.tcpi_bytes_sent) < 0) goto error;
#endif

    return d;

error:
    Py_DECREF(d);
    return NULL;
}

static PyMethodDef Methods[] = {
    {"enable_deepcc", py_enable_deepcc, METH_VARARGS,
     "enable_deepcc(sock_or_fd, val=2) -> None"},
    {"get_tcp_deepcc_info", py_get_tcp_deepcc_info, METH_VARARGS,
     "get_tcp_deepcc_info(sock_or_fd) -> dict"},
    {"get_tcp_getsockopt_info", py_get_tcp_getsockopt_info, METH_VARARGS,
     "get_tcp_getsockopt_info(sock_or_fd) -> dict"},
    {"set_cwnd", py_set_cwnd, METH_VARARGS,
     "set_cwnd(sock_or_fd, cwnd_u32) -> None"},
    {"set_pacing_rate", py_set_pacing_rate, METH_VARARGS,
     "set_pacing_rate(sock_or_fd, bytes_per_sec_u64) -> None"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "tcp_sockopt",
    "DeepCC/TCP sockopts bindings",
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_tcp_sockopt(void) {
    return PyModule_Create(&module);
}