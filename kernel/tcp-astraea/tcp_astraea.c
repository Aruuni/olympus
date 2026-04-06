#include <linux/module.h>
#include <linux/random.h>
#include <net/tcp.h>

#define THR_SCALE 24
#define THR_UNIT (1 << THR_SCALE)

const char* prefix = "[Astraea-sat-dbg12]";

struct astraea {
  /* CA state on previous ACK */
  u32 prev_ca_state : 3;
  /* prior cwnd upon entering loss recovery */
  u32 prior_cwnd;
};

static void astraea_init(struct sock* sk) {
  // struct tcp_sock* tp = tcp_sk(sk);
  struct astraea* astraea = inet_csk_ca(sk);
  astraea->prev_ca_state = TCP_CA_Open;
  astraea->prior_cwnd = 0;

  cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}

/* Initialize cwnd to support current pacing rate (but not less then 4 packets)
 */
static void astraea_set_cwnd(struct sock* sk) {
  // struct tcp_sock* tp = tcp_sk(sk);
}

static void astraea_save_cwnd(struct sock* sk) {
  struct tcp_sock* tp = tcp_sk(sk);
  struct astraea* astraea = inet_csk_ca(sk);

  if (astraea->prev_ca_state < TCP_CA_Recovery)
    astraea->prior_cwnd = tp->snd_cwnd; /* this cwnd is good enough */
  else /* loss recovery or BBR_PROBE_RTT have temporarily cut cwnd */
    astraea->prior_cwnd = max(astraea->prior_cwnd, tp->snd_cwnd);
}

static void astraea_update_pacing_rate(struct sock* sk) {
  const struct tcp_sock* tp = tcp_sk(sk);
  u64 rate;
  cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);

  rate = tcp_mss_to_mtu(sk, tcp_sk(sk)->mss_cache);  //

  rate *= USEC_PER_SEC;

  rate *= max(tp->snd_cwnd, tp->packets_out);

  if (likely(tp->srtt_us >> 3)) do_div(rate, tp->srtt_us >> 3);

  /* WRITE_ONCE() is needed because sch_fq fetches sk_pacing_rate
   * without any lock. We want to make sure compiler wont store
   * intermediate values in this location.
   */
  WRITE_ONCE(sk->sk_pacing_rate, min_t(u64, rate, sk->sk_max_pacing_rate));
}

void astraea_update_cwnd(struct sock* sk) {
  struct tcp_sock* tp = tcp_sk(sk);
  tp->snd_cwnd = max(tp->snd_cwnd, tp->cwnd_min);
  tp->snd_cwnd = min(tp->snd_cwnd, tp->snd_cwnd_clamp);
  if (tp->deepcc_enable > 1) astraea_update_pacing_rate(sk);
}

/**
 * @brief We implement this function to bypass the CWND reduction in loss &
 * recovery state Note that: Updating of CWND and pacing rate should be done
 * here as this function will end utilimate congestion control logic.
 *
 * @param sk pointer to current socket
 * @param rs pointer to rate sample
 */
static void astraea_cong_control(struct sock* sk, const struct rate_sample* rs) {
  struct tcp_sock* tp = tcp_sk(sk);
  struct astraea* astraea = inet_csk_ca(sk);
  // print rate sample
  u64 bw;

  // we believe cwnd has been modified by user-space RL-agent
  u32 cwnd = max(tp->prior_cwnd, astraea->prior_cwnd);
  tp->snd_cwnd = max(tp->snd_cwnd, cwnd);
  astraea_update_cwnd(sk);

  if (rs->delivered < 0 || rs->interval_us <= 0) {
    bw = 0;
  } else {
    // we first need to enlarge bw thus avoiding get zero
    bw = (u64)rs->delivered * THR_UNIT;
    do_div(bw, rs->interval_us);
    // deliverd is num of packers, we translate it to bytes, bw in bytes per
    // second
    bw = bw * tp->mss_cache * USEC_PER_SEC >> THR_SCALE;
  }

  // printk(KERN_INFO
  //        "%s: snd_cwnd: %u, rcv_wnd: %u, current_state: %u, pacing_rate: %lu, "
  //        "sampled_rate: %llu, deliverd: %u, interval_us:%lu, packet_out:%u",
  //        prefix, tp->snd_cwnd, tp->rcv_wnd, inet_csk(sk)->icsk_ca_state,
  //        sk->sk_pacing_rate, bw, rs->delivered, rs->interval_us,
  //        tp->packets_out);
}

/**
 * Astraea actually does not need this function as we don't always
 * want to reduce CWND in losses.
 */
static u32 astraea_undo_cwnd(struct sock* sk) { return tcp_sk(sk)->snd_cwnd; }

/* save current cwnd for quick ramp up */
static u32 astraea_ssthresh(struct sock* sk) {
  const struct tcp_sock* tp = tcp_sk(sk);
  // we want RL to take more efficient control
  astraea_save_cwnd(sk);
  return max(tp->snd_cwnd, 10U);
}

static void astraea_cong_avoid(struct sock* sk, u32 ack, u32 acked) {
  // printk(KERN_INFO "[TCP Astraea] Nothing done in Astraea CC");
  // struct tcp_sock* tp = tcp_sk(sk);
  // if (tcp_in_cwnd_reduction(sk)) {
  //   // prior_cwnd is the cwnd right before starting loss recovery
  //   if (tp->prior_cwnd > tp->snd_cwnd) tp->snd_cwnd = tp->prior_cwnd;
  // }
}

static void astraea_pkts_acked(struct sock* sk, const struct ack_sample* acks) {
  struct tcp_sock* tp = tcp_sk(sk);
  s32 rtt = max(acks->rtt_us, 0);
  // printk(KERN_INFO "%s: cwnd: %u, current_state: %u, sampled_rtt: %u", prefix, tp->snd_cwnd, inet_csk(sk)->icsk_ca_state, rtt);
}

static void astraea_ack_event(struct sock* sk, u32 flags) {}

static void astraea_cwnd_event(struct sock* sk, enum tcp_ca_event event) {
  if (event == CA_EVENT_LOSS) {
    // printk(KERN_INFO "%s packet loss: cwnd: %u, current_state: %u", prefix, tcp_sk(sk)->snd_cwnd, inet_csk(sk)->icsk_ca_state);
  }
}

static void astraea_set_state(struct sock* sk, u8 new_state) {
  struct astraea* astraea = inet_csk_ca(sk);

  if (new_state == TCP_CA_Loss) {
    astraea->prev_ca_state = TCP_CA_Loss;
  } else if (new_state == TCP_CA_Recovery) {
    //printk(KERN_INFO "%s recovery: cwnd: %u, current_state: %u", prefix, tcp_sk(sk)->snd_cwnd, inet_csk(sk)->icsk_ca_state);
  }
}

struct tcp_congestion_ops tcp_astraea_ops __read_mostly = {
  .flags = TCP_CONG_NON_RESTRICTED,
  .name = "astraea",
  .owner = THIS_MODULE,
  .init = astraea_init,
  // .cong_control = astraea_cong_control,
  .undo_cwnd = astraea_undo_cwnd,
  .ssthresh = astraea_ssthresh,
  .set_state = astraea_set_state,
  .cong_avoid = astraea_cong_avoid,
  .pkts_acked = astraea_pkts_acked,
  // .in_ack_event = astraea_ack_event,
  .cwnd_event = astraea_cwnd_event,
};

EXPORT_SYMBOL_GPL(tcp_astraea_ops);

/* Kernel module section */
static int __init astraea_register(void) {
  BUILD_BUG_ON(sizeof(struct astraea) > ICSK_CA_PRIV_SIZE);
  printk(KERN_INFO
         "[TCP Astraea] Astraea init clean tcp congestion control logic\n");
  return tcp_register_congestion_control(&tcp_astraea_ops);
}

static void __exit astraea_unregister(void) {
  printk(KERN_INFO "[TCP Astraea] Astraea unregistered");
  tcp_unregister_congestion_control(&tcp_astraea_ops);
}

module_init(astraea_register);
module_exit(astraea_unregister);

MODULE_AUTHOR("Xudong Liao <stephenxudong@gmail.com>");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("TCP Astraea (Clean-version TCP Congestion Control)");
