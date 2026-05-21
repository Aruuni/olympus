#include <asm-generic/errno.h>
#include <asm-generic/errno-base.h>
#include <linux/hashtable.h>
#include <linux/spinlock.h>
#include "mutant.h"

#define NETLINK_USER 25
#define MAX_PAYLOAD 256

#define COMM_END   0
#define COMM_BEGIN 1

/* ── Netlink reporting channel ───────────────────────────────────────────── */
struct sock *nl_sk = NULL;
static u32 nl_peer_pid = (u32)-1;

static u64 thruput   = 0;
static u64 loss_rate = 0;
static struct mutant_info info;

/* ── Exported CC ops ─────────────────────────────────────────────────────── */
extern struct tcp_congestion_ops cubictcp;
extern struct tcp_congestion_ops tcp_hybla;
extern struct tcp_congestion_ops tcp_bbr1_cong_ops;
extern struct tcp_congestion_ops tcp_bbr3_cong_ops;
extern struct tcp_congestion_ops tcp_westwood;
extern struct tcp_congestion_ops tcp_veno;
extern struct tcp_congestion_ops tcp_vegas;
extern struct tcp_congestion_ops tcp_yeah;
extern struct tcp_congestion_ops tcp_cdg;
extern struct tcp_congestion_ops bic;
extern struct tcp_congestion_ops htcp;
extern struct tcp_congestion_ops tcp_highspeed;
extern struct tcp_congestion_ops tcp_illinois;
extern struct tcp_congestion_ops tcp_astraea_ops;

/* ── Per-socket saved algorithm states ───────────────────────────────────── */
struct mutant_state {
    struct bictcp   *cubic_state;
    struct hybla    *hybla_state;
    struct bbr1     *bbr1_state;
    struct bbr3     *bbr3_state;
    struct westwood *westwood_state;
    struct veno     *veno_state;
    struct vegas    *vegas_state;
    struct yeah     *yeah_state;
    struct cdg      *cdg_state;
    struct bic      *bic_state;
    struct htcp     *htcp_state;
    struct hstcp    *highspeed_state;
    struct illinois *illinois_state;
    struct astraea  *astraea_state;
};

/*
 * mutant_per_sock — per-socket CC control state.
 *
 * icsk_ca_priv is left ENTIRELY for the delegated algorithm's live state,
 * exactly as in the original design.  Our bookkeeping lives on the heap,
 * indexed by sk pointer in mps_hash below.
 */
struct mutant_per_sock {
    struct tcp_congestion_ops *current_ops;
    u32                        selected_proto_id;
    u32                        prev_proto_id;
    struct mutant_state       *saved_states;
};

/* ── Per-socket hash table ───────────────────────────────────────────────── */

#define MPS_HASH_BITS 6   /* 64 buckets; plenty for experiment-scale flows */

struct mps_entry {
    struct hlist_node     node;
    struct sock          *sk;
    struct mutant_per_sock mps;
};

static DEFINE_HASHTABLE(mps_hash, MPS_HASH_BITS);
static DEFINE_SPINLOCK(mps_lock);

/*
 * get_mps — look up the per-socket struct for sk.
 *
 * Safe to call from softirq (pkts_acked) as well as process context.
 * TCP socket callbacks for the same socket are serialised by the socket
 * lock, so the pointer returned remains valid for the duration of the
 * calling callback.
 */
static struct mutant_per_sock *get_mps(struct sock *sk)
{
    struct mps_entry *e;
    u32 bkt = hash_ptr(sk, MPS_HASH_BITS);

    spin_lock_bh(&mps_lock);
    hlist_for_each_entry(e, &mps_hash[bkt], node) {
        if (e->sk == sk) {
            spin_unlock_bh(&mps_lock);
            return &e->mps;
        }
    }
    spin_unlock_bh(&mps_lock);
    return NULL;
}

/* ── Saved-state helpers ─────────────────────────────────────────────────── */

static struct mutant_state *alloc_saved_states(void)
{
    struct mutant_state *s = kmalloc(sizeof(*s), GFP_KERNEL);
    if (s)
        memset(s, 0, sizeof(*s));
    return s;
}

static void free_saved_states(struct mutant_state *s)
{
    if (!s)
        return;
    kfree(s->cubic_state);
    kfree(s->hybla_state);
    kfree(s->bbr1_state);
    kfree(s->bbr3_state);
    kfree(s->westwood_state);
    kfree(s->veno_state);
    kfree(s->vegas_state);
    kfree(s->yeah_state);
    kfree(s->cdg_state);
    kfree(s->bic_state);
    kfree(s->htcp_state);
    kfree(s->highspeed_state);
    kfree(s->illinois_state);
    kfree(s->astraea_state);
    kfree(s);
}

/* ── Netlink messaging ───────────────────────────────────────────────────── */

static void send_msg(char *message, u32 pid)
{
    int messageSize;
    int rc;
    struct sk_buff *skb;
    struct nlmsghdr *reply_nlh;

    if (pid == (u32)-1 || nl_sk == NULL)
        return;

    messageSize = strlen(message);
    skb = nlmsg_new(messageSize, 0);
    if (!skb)
        return;

    reply_nlh = nlmsg_put(skb, 0, 0, NLMSG_DONE, messageSize, 0);
    NETLINK_CB(skb).dst_group = 0;
    strncpy(NLMSG_DATA(reply_nlh), message, messageSize);

    rc = nlmsg_unicast(nl_sk, skb, pid);
    (void)rc;
}

static void start_connection(struct nlmsghdr *nlh)
{
    char msg[MAX_PAYLOAD - 1];
    nl_peer_pid = nlh->nlmsg_pid;
    snprintf(msg, sizeof(msg), "%u", 0);
    send_msg(msg, nl_peer_pid);
    printk(KERN_INFO "[mutant] reporting channel open  peer_pid=%u\n", nl_peer_pid);
}

static void end_connection(struct nlmsghdr *nlh)
{
    char msg[MAX_PAYLOAD - 1];
    (void)nlh;
    snprintf(msg, sizeof(msg), "%u", (u32)-1);
    send_msg(msg, nl_peer_pid);
    nl_peer_pid = (u32)-1;
    printk(KERN_INFO "[mutant] reporting channel closed\n");
}

static void receive_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    if (!skb)
        return;
    nlh = (struct nlmsghdr *)skb->data;
    switch (nlh->nlmsg_flags) {
    case COMM_END:   end_connection(nlh);   break;
    case COMM_BEGIN: start_connection(nlh); break;
    default:
        /* COMM_SELECT_ARM retired — arm switching via TCP_MUTANT_ARM sockopt */
        break;
    }
}

static int __init netlink_init(void)
{
    struct netlink_kernel_cfg cfg = { .input = receive_msg };
    nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &cfg);
    if (!nl_sk) {
        printk(KERN_ALERT "[mutant] netlink socket create failed\n");
        return -10;
    }
    return 0;
}

static void __exit netlink_exit(void)
{
    netlink_kernel_release(nl_sk);
}

/* ── CC ops selector ─────────────────────────────────────────────────────── */

static struct tcp_congestion_ops *ops_for_arm(u32 arm_id)
{
    switch (arm_id) {
    case CUBIC:     return &cubictcp;
    case HYBLA:     return &tcp_hybla;
    case BBR1:      return &tcp_bbr1_cong_ops;
    case BBR3:      return &tcp_bbr3_cong_ops;
    case WESTWOOD:  return &tcp_westwood;
    case VENO:      return &tcp_veno;
    case VEGAS:     return &tcp_vegas;
    case YEAH:      return &tcp_yeah;
    case CDG:       return &tcp_cdg;
    case BIC:       return &bic;
    case HTCP:      return &htcp;
    case HIGHSPEED: return &tcp_highspeed;
    case ILLINOIS:  return &tcp_illinois;
    case ASTRAEA:   return &tcp_astraea_ops;
    default:        return &cubictcp;
    }
}

/* ── Per-socket save / load (operate on inet_csk_ca(sk) at offset 0) ─────── */

static void save_state(struct sock *sk, struct mutant_per_sock *mps)
{
    struct mutant_state *s = mps->saved_states;

    if (!s)
        return;

#define SAVE_ARM(arm_const, field, type)                              \
    case arm_const:                                                    \
        kfree(s->field);                                               \
        s->field = kmalloc(sizeof(type), GFP_ATOMIC);                  \
        if (s->field)                                                  \
            memcpy(s->field, inet_csk_ca(sk), sizeof(type));           \
        break;

    switch (mps->prev_proto_id) {
    SAVE_ARM(CUBIC,     cubic_state,     struct bictcp)
    SAVE_ARM(HYBLA,     hybla_state,     struct hybla)
    SAVE_ARM(BBR1,      bbr1_state,      struct bbr1)
    SAVE_ARM(BBR3,      bbr3_state,      struct bbr3)
    SAVE_ARM(WESTWOOD,  westwood_state,  struct westwood)
    SAVE_ARM(VENO,      veno_state,      struct veno)
    SAVE_ARM(VEGAS,     vegas_state,     struct vegas)
    SAVE_ARM(YEAH,      yeah_state,      struct yeah)
    SAVE_ARM(CDG,       cdg_state,       struct cdg)
    SAVE_ARM(BIC,       bic_state,       struct bic)
    SAVE_ARM(HTCP,      htcp_state,      struct htcp)
    SAVE_ARM(HIGHSPEED, highspeed_state, struct hstcp)
    SAVE_ARM(ILLINOIS,  illinois_state,  struct illinois)
    SAVE_ARM(ASTRAEA,   astraea_state,   struct astraea)
    default: break;
    }
#undef SAVE_ARM
}

static void load_state(struct sock *sk, struct mutant_per_sock *mps)
{
    struct mutant_state *s = mps->saved_states;
    struct tcp_congestion_ops *ops = mps->current_ops;

    if (!s)
        return;

#define LOAD_OR_INIT(arm_const, field, type)                          \
    case arm_const:                                                    \
        if (s->field) {                                               \
            memcpy(inet_csk_ca(sk), s->field, sizeof(type));          \
        } else {                                                      \
            s->field = kmalloc(sizeof(type), GFP_ATOMIC);             \
            if (!s->field) break;                                     \
            memset(inet_csk_ca(sk), 0, sizeof(type));                 \
            if (ops->init) ops->init(sk);                             \
            memcpy(s->field, inet_csk_ca(sk), sizeof(type));          \
        }                                                             \
        break;

    switch (mps->selected_proto_id) {
    LOAD_OR_INIT(CUBIC,     cubic_state,     struct bictcp)
    LOAD_OR_INIT(HYBLA,     hybla_state,     struct hybla)
    LOAD_OR_INIT(BBR1,      bbr1_state,      struct bbr1)
    LOAD_OR_INIT(BBR3,      bbr3_state,      struct bbr3)
    LOAD_OR_INIT(WESTWOOD,  westwood_state,  struct westwood)
    LOAD_OR_INIT(VENO,      veno_state,      struct veno)
    LOAD_OR_INIT(VEGAS,     vegas_state,     struct vegas)
    LOAD_OR_INIT(YEAH,      yeah_state,      struct yeah)
    LOAD_OR_INIT(CDG,       cdg_state,       struct cdg)
    LOAD_OR_INIT(BIC,       bic_state,       struct bic)
    LOAD_OR_INIT(HTCP,      htcp_state,      struct htcp)
    LOAD_OR_INIT(HIGHSPEED, highspeed_state, struct hstcp)
    LOAD_OR_INIT(ILLINOIS,  illinois_state,  struct illinois)
    LOAD_OR_INIT(ASTRAEA,   astraea_state,   struct astraea)
    default:
        printk(KERN_WARNING "[mutant] load_state: unknown arm %u\n",
               mps->selected_proto_id);
        break;
    }
#undef LOAD_OR_INIT
}

/* ── Arm switch on a specific socket ─────────────────────────────────────── */

static void do_arm_switch(struct sock *sk, struct mutant_per_sock *mps, u32 new_arm)
{
    if (new_arm == mps->selected_proto_id)
        return;

    mps->prev_proto_id     = mps->selected_proto_id;
    save_state(sk, mps);

    mps->selected_proto_id = new_arm;
    mps->current_ops       = ops_for_arm(new_arm);
    load_state(sk, mps);

    printk(KERN_INFO "[mutant] sk=%p  arm %u->%u  (%s)\n",
           sk, mps->prev_proto_id, mps->selected_proto_id,
           mps->current_ops->name);
}

/* ── Metrics reporting ───────────────────────────────────────────────────── */

static void send_info(struct mutant_info *mi)
{
    char msg[MAX_PAYLOAD - 1];
    snprintf(msg, sizeof(msg),
        "%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%llu;%u;%u;%llu;%u",
        mi->now, mi->snd_cwnd, mi->rtt_us, mi->srtt_us,
        mi->mdev_us, mi->min_rtt, mi->advmss, mi->delivered,
        mi->lost_out, mi->packets_out, mi->retrans_out,
        mi->rate, mi->prev_proto_id, mi->selected_proto_id,
        mi->thruput, mi->loss_rate);
    send_msg(msg, nl_peer_pid);
}

static void send_net_params(struct tcp_sock *tp, struct sock *sk,
                             struct mutant_per_sock *mps)
{
    u32 rate = READ_ONCE(tp->rate_delivered);
    u32 intv = READ_ONCE(tp->rate_interval_us);

    if (tp->packets_out + tp->retrans_out > 0)
        loss_rate = ((u64)tp->lost_out * 100) /
                    (tp->packets_out + tp->retrans_out);

    if (rate && intv) {
        thruput = (u64)rate * tp->mss_cache * USEC_PER_SEC * 8;
        do_div(thruput, intv);
    } else {
        thruput = 0;
    }

    info.now               = tcp_jiffies32;
    info.snd_cwnd          = tp->snd_cwnd;
    info.rtt_us            = tp->rack.rtt_us;
    info.srtt_us           = tp->srtt_us;
    info.mdev_us           = tp->mdev_us;
    info.min_rtt           = tcp_min_rtt(tp);
    info.advmss            = tp->advmss;
    info.delivered         = tp->delivered;
    info.lost_out          = tp->lost_out;
    info.packets_out       = tp->packets_out;
    info.retrans_out       = tp->retrans_out;
    info.rate              = rate;
    info.prev_proto_id     = mps->prev_proto_id;
    info.selected_proto_id = mps->selected_proto_id;
    info.thruput           = thruput;
    info.loss_rate         = loss_rate;

    send_info(&info);
    (void)sk;
}

/* ── TCP congestion control callbacks ────────────────────────────────────── */

static void mutant_tcp_init(struct sock *sk)
{
    struct tcp_sock   *tp  = tcp_sk(sk);
    struct mps_entry  *e;
    struct mutant_per_sock *mps;
    u32 bkt = hash_ptr(sk, MPS_HASH_BITS);

    e = kzalloc(sizeof(*e), GFP_KERNEL);
    if (!e) {
        pr_err("[mutant] init: OOM for sk=%p\n", sk);
        return;
    }

    e->sk                  = sk;
    e->mps.current_ops     = &cubictcp;
    e->mps.selected_proto_id = CUBIC;
    e->mps.prev_proto_id   = CUBIC;
    e->mps.saved_states    = alloc_saved_states();

    spin_lock_bh(&mps_lock);
    hlist_add_head(&e->node, &mps_hash[bkt]);
    spin_unlock_bh(&mps_lock);

    mps = &e->mps;

    /* Initialise the sockopt request fields */
    WRITE_ONCE(tp->mutant_arm, CUBIC);
    WRITE_ONCE(tp->mutant_arm_pending, 0);

    /* Delegate init — writes bictcp to inet_csk_ca(sk) */
    if (cubictcp.init)
        cubictcp.init(sk);

    printk(KERN_INFO "[mutant] init  sk=%p\n", sk);
    (void)mps;
}

static void mutant_tcp_release(struct sock *sk)
{
    struct mps_entry *e;
    u32 bkt = hash_ptr(sk, MPS_HASH_BITS);

    spin_lock_bh(&mps_lock);
    hlist_for_each_entry(e, &mps_hash[bkt], node) {
        if (e->sk == sk) {
            /* Run the delegated release before we remove the entry */
            if (e->mps.current_ops && e->mps.current_ops->release)
                e->mps.current_ops->release(sk);

            hlist_del(&e->node);
            spin_unlock_bh(&mps_lock);

            free_saved_states(e->mps.saved_states);
            kfree(e);
            printk(KERN_INFO "[mutant] release  sk=%p\n", sk);
            return;
        }
    }
    spin_unlock_bh(&mps_lock);
}

static void mutant_tcp_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->cong_avoid)
        mps->current_ops->cong_avoid(sk, ack, acked);
}

static u32 mutant_tcp_ssthresh(struct sock *sk)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->ssthresh)
        return mps->current_ops->ssthresh(sk);
    return TCP_INFINITE_SSTHRESH;
}

static void mutant_tcp_set_state(struct sock *sk, u8 new_state)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->set_state)
        mps->current_ops->set_state(sk, new_state);
}

static u32 mutant_tcp_undo_cwnd(struct sock *sk)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->undo_cwnd)
        return mps->current_ops->undo_cwnd(sk);
    return tcp_sk(sk)->snd_cwnd;
}

static void mutant_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->cwnd_event)
        mps->current_ops->cwnd_event(sk, event);
}

/*
 * pkts_acked — hot path.
 *
 * Checks tp->mutant_arm_pending (set atomically by TCP_MUTANT_ARM setsockopt)
 * and performs the per-socket arm switch before delegating.
 */
static void mutant_tcp_pkts_acked(struct sock *sk, const struct ack_sample *sample)
{
    struct tcp_sock        *tp  = tcp_sk(sk);
    struct mutant_per_sock *mps = get_mps(sk);

    if (unlikely(!mps))
        return;

    if (unlikely(READ_ONCE(tp->mutant_arm_pending))) {
        u32 new_arm = READ_ONCE(tp->mutant_arm);
        WRITE_ONCE(tp->mutant_arm_pending, 0);
        do_arm_switch(sk, mps, new_arm);
    }

    if (mps->current_ops && mps->current_ops->pkts_acked)
        mps->current_ops->pkts_acked(sk, sample);

    if (nl_peer_pid != (u32)-1)
        send_net_params(tp, sk, mps);
}

static void mutant_tcp_ack_event(struct sock *sk, u32 flags)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->in_ack_event)
        mps->current_ops->in_ack_event(sk, flags);
}

static u32 mutant_tcp_cong_control(struct sock *sk, const struct rate_sample *rs,
                                   u32 ack, u32 acked, int flag)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->cong_control) {
        mps->current_ops->cong_control(sk, ack, flag, rs);
        return 0;
    }
    return -1;
}

static u32 mutant_tcp_sndbuf_expand(struct sock *sk)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->sndbuf_expand)
        return mps->current_ops->sndbuf_expand(sk);
    return 2;
}

static u32 mutant_tcp_min_tso_segs(struct sock *sk, unsigned int mss)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->tso_segs)
        return mps->current_ops->tso_segs(sk, mss);
    return 2;
}

static size_t mutant_tcp_get_info(struct sock *sk, u32 ext, int *attr,
                                  union tcp_cc_info *info)
{
    struct mutant_per_sock *mps = get_mps(sk);
    if (mps && mps->current_ops && mps->current_ops->get_info)
        return mps->current_ops->get_info(sk, ext, attr, info);
    return 0;
}

/* ── Registration ────────────────────────────────────────────────────────── */

static struct tcp_congestion_ops mutant_cong_ops __read_mostly = {
    .flags                   = TCP_CONG_NON_RESTRICTED,
    .init                    = mutant_tcp_init,
    .release                 = mutant_tcp_release,
    .ssthresh                = mutant_tcp_ssthresh,
    .cong_avoid              = mutant_tcp_cong_avoid,
    .set_state               = mutant_tcp_set_state,
    .undo_cwnd               = mutant_tcp_undo_cwnd,
    .cwnd_event              = mutant_tcp_cwnd_event,
    .pkts_acked              = mutant_tcp_pkts_acked,
    .in_ack_event            = mutant_tcp_ack_event,
    .sndbuf_expand           = mutant_tcp_sndbuf_expand,
    .tso_segs                = mutant_tcp_min_tso_segs,
    .get_info                = mutant_tcp_get_info,
    .mutant_tcp_cong_control = mutant_tcp_cong_control,
    .owner                   = THIS_MODULE,
    .name                    = "mutant",
};

static int __init mutant_tcp_module_init(void)
{
    hash_init(mps_hash);

    if (netlink_init() < 0) {
        pr_err("[mutant] netlink init failed\n");
        return -EINVAL;
    }

    if (tcp_register_congestion_control(&mutant_cong_ops) < 0) {
        pr_err("[mutant] CC registration failed\n");
        netlink_exit();
        return -EINVAL;
    }

    printk(KERN_INFO "[mutant] loaded — per-socket via hash table, arm switch via TCP_MUTANT_ARM\n");
    return 0;
}

static void __exit mutant_tcp_module_exit(void)
{
    tcp_unregister_congestion_control(&mutant_cong_ops);
    netlink_exit();
}

module_init(mutant_tcp_module_init);
module_exit(mutant_tcp_module_exit);
MODULE_AUTHOR("Lorenzo Pappone");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Mutant — per-socket CC arm switcher");
MODULE_VERSION("2.0");
