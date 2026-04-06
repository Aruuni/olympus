#ifndef MUTANT_H
#define MUTANT_H

#include <linux/netlink.h>
#include <net/tcp.h>

#define MAX_PAYLOAD 256

#define COMM_END 0
#define COMM_BEGIN 1
#define COMM_SELECT_ARM 2
#define TEST 3

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

struct mutant_info {
    u32 now;
    u32 snd_cwnd;
    u32 rtt_us;
    u32 srtt_us;
    u32 mdev_us;
    u32 min_rtt;
    u32 advmss;
    u32 delivered;
    u32 lost_out;
    u32 packets_out;
    u32 retrans_out;
    u64 rate;
    u32 prev_proto_id;
    u32 selected_proto_id;
    u64 thruput;
    u32 loss_rate;
};

/* Cubic */
struct bictcp {
    u32 cnt;
    u32 last_max_cwnd;
    u32 last_cwnd;
    u32 last_time;
    u32 bic_origin_point;
    u32 bic_K;
    u32 delay_min;
    u32 epoch_start;
    u32 ack_cnt;
    u32 tcp_cwnd;
    u16 unused;
    u8 sample_cnt;
    u8 found;
    u32 round_start;
    u32 end_seq;
    u32 last_ack;
    u32 curr_rtt;
};

/* Hybla */
struct hybla {
    bool hybla_en;
    u32 snd_cwnd_cents;
    u32 rho;
    u32 rho2;
    u32 rho_3ls;
    u32 rho2_7ls;
    u32 minrtt_us;
};

/* BBR */
struct bbr1 {
    u32 min_rtt_us;
    u32 min_rtt_stamp;
    u32 probe_rtt_done_stamp;
    struct minmax bw;
    u32 rtt_cnt;
    u32 next_rtt_delivered;
    u64 cycle_mstamp;
    u32 mode:3,
        prev_ca_state:3,
        packet_conservation:1,
        round_start:1,
        idle_restart:1,
        probe_rtt_round_done:1,
        unused:13,
        lt_is_sampling:1,
        lt_rtt_cnt:7,
        lt_use_bw:1;
    u32 lt_bw;
    u32 lt_last_delivered;
    u32 lt_last_stamp;
    u32 lt_last_lost;
    u32 pacing_gain:10,
        cwnd_gain:10,
        full_bw_reached:1,
        full_bw_cnt:2,
        cycle_idx:3,
        has_seen_rtt:1,
        unused_b:5;
    u32 prior_cwnd;
    u32 full_bw;
    u64 ack_epoch_mstamp;
    u16 extra_acked[2];
    u32 ack_epoch_acked:20,
        extra_acked_win_rtts:5,
        extra_acked_win_idx:1,
        unused_c:6;
};

struct bbr3 {
    u32 min_rtt_us;
    u32 min_rtt_stamp;
    u32 probe_rtt_done_stamp;
    u32 probe_rtt_min_us;
    u32 probe_rtt_min_stamp;
    u32 next_rtt_delivered;
    u64 cycle_mstamp;
    u32 mode:2,
        prev_ca_state:3,
        round_start:1,
        ce_state:1,
        bw_probe_up_rounds:5,
        try_fast_path:1,
        idle_restart:1,
        probe_rtt_round_done:1,
        init_cwnd:7,
        unused_1:10;
    u32 pacing_gain:10,
        cwnd_gain:10,
        full_bw_reached:1,
        full_bw_cnt:2,
        cycle_idx:2,
        has_seen_rtt:1,
        unused_2:6;
    u32 prior_cwnd;
    u32 full_bw;
    u64 ack_epoch_mstamp;
    u16 extra_acked[2];
    u32 ack_epoch_acked:20,
        extra_acked_win_rtts:5,
        extra_acked_win_idx:1,
        full_bw_now:1,
        startup_ecn_rounds:2,
        loss_in_cycle:1,
        ecn_in_cycle:1,
        unused_3:1;
    u32 loss_round_delivered;
    u32 undo_bw_lo;
    u32 undo_inflight_lo;
    u32 undo_inflight_hi;
    u32 bw_latest;
    u32 bw_lo;
    u32 bw_hi[2];
    u32 inflight_latest;
    u32 inflight_lo;
    u32 inflight_hi;
    u32 bw_probe_up_cnt;
    u32 bw_probe_up_acks;
    u32 probe_wait_us;
    u32 prior_rcv_nxt;
    u32 ecn_eligible:1,
        ecn_alpha:9,
        bw_probe_samples:1,
        prev_probe_too_high:1,
        stopped_risky_probe:1,
        rounds_since_probe:8,
        loss_round_start:1,
        loss_in_round:1,
        ecn_in_round:1,
        ack_phase:3,
        loss_events_in_round:4,
        initialized:1;
    u32 alpha_last_delivered;
    u32 alpha_last_delivered_ce;
    u8  unused_4;
    struct tcp_plb_state plb;
};

struct westwood {
    u32 bw_ns_est;
    u32 bw_est;
    u32 rtt_win_sx;
    u32 bk;
    u32 snd_una;
    u32 cumul_ack;
    u32 accounted;
    u32 rtt;
    u32 rtt_min;
    u8 first_ack;
    u8 reset_rtt_min;
};

struct veno {
    u8 doing_veno_now;
    u16 cntrtt;
    u32 minrtt;
    u32 basertt;
    u32 inc;
    u32 diff;
};

struct vegas {
    u32 beg_snd_nxt;
    u32 beg_snd_una;
    u32 beg_snd_cwnd;
    u8 doing_vegas_now;
    u16 cntRTT;
    u32 minRTT;
    u32 baseRTT;
    u32 cwnd;
};

struct yeah {
    struct vegas vegas;
    u32 lastQ;
    u32 doing_reno_now;
    u32 reno_count;
    u32 fast_count;
    u32 pkts_acked;
};

struct cdg_minmax {
    union {
        struct {
            s32 min;
            s32 max;
        };
        u64 v64;
    };
};

struct cdg {
    struct cdg_minmax rtt;
    struct cdg_minmax rtt_prev;
    struct cdg_minmax *gradients;
    struct cdg_minmax gsum;
    bool gfilled;
    u8 tail;
    u8 state;
    u8 delack;
    u32 rtt_seq;
    u32 shadow_wnd;
    u16 backoff_cnt;
    u16 sample_cnt;
    s32 delay_min;
    u32 last_ack;
    u32 round_start;
};

struct bic {
    u32 cnt;
    u32 last_max_cwnd;
    u32 last_cwnd;
    u32 last_time;
    u32 epoch_start;
#define ACK_RATIO_SHIFT 4
    u32 delayed_ack;
    u32 cwnd;
};

struct htcp {
    u32 alpha;
    u8 beta;
    u8 modeswitch;
    u16 pkts_acked;
    u32 packetcount;
    u32 minRTT;
    u32 maxRTT;
    u32 last_cong;
    u32 undo_last_cong;
    u32 undo_maxRTT;
    u32 undo_old_maxB;
    u32 minB;
    u32 maxB;
    u32 old_maxB;
    u32 Bi;
    u32 lasttime;
};

struct hstcp {
    u32 ai;
};

struct illinois {
    u64 sum_rtt;
    u16 cnt_rtt;
    u32 base_rtt;
    u32 max_rtt;
    u32 end_seq;
    u32 alpha;
    u32 beta;
    u16 acked;
    u8 rtt_above;
    u8 rtt_low;
};

/* Astraea */
struct astraea {
    u32 prev_ca_state : 3;
    u32 prior_cwnd;
};

/* prototypes */
static void send_msg(char *message, int socketId);
static void start_connection(struct nlmsghdr *nlh);
static void end_connection(struct nlmsghdr *nlh);
static void receive_msg(struct sk_buff *skb);
static int netlink_init(void);
static void netlink_exit(void);

static void save_state(struct sock *sk);
static void load_state(struct sock *sk);
static void init_saved_states(void);
static void print_bictcp(struct bictcp *cubic);
static void print_hybla(struct hybla *hybla);
static void print_bbr(struct bbr *bbr);
static void print_mutant_state(struct sock *sk);

static void mutant_switch_congestion_control(void);
static void send_net_params(struct tcp_sock *tp, struct sock *sk, int socketId);
static void mutant_tcp_init(struct sock *sk);
static void mutant_tcp_cong_avoid(struct sock *sk, u32 ack, u32 acked);
static u32 mutant_tcp_ssthresh(struct sock *sk);
static void mutant_tcp_set_state(struct sock *sk, u8 new_state);
static u32 mutant_tcp_undo_cwnd(struct sock *sk);
static void mutant_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event);
static void mutant_tcp_pkts_acked(struct sock *sk, const struct ack_sample *sample);
static u32 mutant_tcp_cong_control(struct sock *sk, const struct rate_sample *rs, u32 ack, u32 acked, int flag);
static u32 mutant_tcp_sndbuf_expand(struct sock *sk);
static u32 mutant_tcp_min_tso_segs(struct sock *sk, unsigned int mss);
static size_t mutant_tcp_get_info(struct sock *sk, u32 ext, int *attr, union tcp_cc_info *info);
static void mutant_tcp_release(struct sock *sk);
static void send_info(struct mutant_info *info);

#endif