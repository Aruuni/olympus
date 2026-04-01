#include <asm-generic/errno.h>
#include <asm-generic/errno-base.h>
#include "mutant.h"

#define NETLINK_USER 25
#define MAX_PAYLOAD 256

#define COMM_END 0
#define COMM_BEGIN 1
#define COMM_SELECT_ARM 2
#define TEST 3

struct sock *nl_sk = NULL;
static u32 socketId = -1;
static u32 selected_proto_id = CUBIC;
static u32 prev_proto_id = CUBIC;
static bool switching_flag = false;
struct mutant_info info;

u64 fail_cnt = 0;
u64 success_cnt = 0;
u64 thruput = 0;
u64 loss_rate = 0.0;

struct tcp_mutant_wrapper {
    struct tcp_congestion_ops *current_ops;
};

static struct tcp_mutant_wrapper mutant_wrapper;

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

static struct mutant_state *saved_states;

static void send_msg(char *message, int socketId)
{
    int messageSize;
    int messageSentReponseCode;
    struct sk_buff *socketMessage;
    struct nlmsghdr *reply_nlh = NULL;

    if (socketId == -1) {
        printk(KERN_INFO "Message not sent: socket not initialized (-1)");
        return;
    }

    messageSize = strlen(message);
    socketMessage = nlmsg_new(messageSize, 0);
    if (!socketMessage) {
        printk(KERN_ERR "Mutant | %s: Failed to allocate new skb | (PID = %d)\n",
               __FUNCTION__, socketId);
        return;
    }

    reply_nlh = nlmsg_put(socketMessage, 0, 0, NLMSG_DONE, messageSize, 0);
    NETLINK_CB(socketMessage).dst_group = 0;
    strncpy(NLMSG_DATA(reply_nlh), message, messageSize);

    if (nl_sk == NULL) {
        printk(KERN_ERR "Mutant | %s: nl_sk is NULL | (PID = %d)\n",
               __FUNCTION__, socketId);
        return;
    }

    messageSentReponseCode = nlmsg_unicast(nl_sk, socketMessage, socketId);
    (void)messageSentReponseCode;
}

static void free_saved_states(void)
{
    if (!saved_states)
        return;

    kfree(saved_states->cubic_state);
    kfree(saved_states->hybla_state);
    kfree(saved_states->bbr1_state);
    kfree(saved_states->westwood_state);
    kfree(saved_states->veno_state);
    kfree(saved_states->vegas_state);
    kfree(saved_states->yeah_state);
    kfree(saved_states->cdg_state);
    kfree(saved_states->bic_state);
    kfree(saved_states->htcp_state);
    kfree(saved_states->highspeed_state);
    kfree(saved_states->illinois_state);
    kfree(saved_states->astraea_state);
    kfree(saved_states);
    saved_states = NULL;
}

static void init_saved_states(void)
{
    if (saved_states)
        return;

    saved_states = kmalloc(sizeof(*saved_states), GFP_KERNEL);
    if (!saved_states) {
        pr_err("Failed to allocate memory for saved_states\n");
        return;
    }
    memset(saved_states, 0, sizeof(*saved_states));
}

static void start_connection(struct nlmsghdr *nlh)
{
    char message[MAX_PAYLOAD - 1];

    fail_cnt = 0;
    prev_proto_id = CUBIC;
    selected_proto_id = CUBIC;
    switching_flag = false;
    mutant_wrapper.current_ops = &cubictcp;

    free_saved_states();
    init_saved_states();

    printk(KERN_INFO "User-kernel communication initialized");

    socketId = nlh->nlmsg_pid;
    snprintf(message, MAX_PAYLOAD - 1, "%u", 0);
    send_msg(message, socketId);
}

static void end_connection(struct nlmsghdr *nlh)
{
    char message[MAX_PAYLOAD - 1];
    (void)nlh;

    printk(KERN_INFO "User-kernel communication ended");

    snprintf(message, MAX_PAYLOAD - 1, "%u", -1);
    send_msg(message, socketId);

    free_saved_states();

    mutant_wrapper.current_ops = &cubictcp;
    prev_proto_id = CUBIC;
    selected_proto_id = CUBIC;
    switching_flag = false;
    socketId = -1;
}

static void receive_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = NULL;

    if (!skb) {
        printk(KERN_ERR "Mutant | %s: skb is NULL\n", __FUNCTION__);
        return;
    }

    nlh = (struct nlmsghdr *)skb->data;

    switch (nlh->nlmsg_flags) {
    case COMM_END:
        printk(KERN_INFO "%s: End connection signal received.", __FUNCTION__);
        end_connection(nlh);
        break;

    case COMM_BEGIN:
        printk(KERN_INFO "%s: Start connection signal received.", __FUNCTION__);
        start_connection(nlh);
        break;

    case COMM_SELECT_ARM:
        printk(KERN_INFO "%s: Select ARM signal received.", __FUNCTION__);
        if (nlh->nlmsg_seq != selected_proto_id) {
            switching_flag = true;
            prev_proto_id = selected_proto_id;
            selected_proto_id = nlh->nlmsg_seq;
            printk(KERN_INFO "%s: switching signal (id: %d->%d)",
                   __FUNCTION__, prev_proto_id, selected_proto_id);
        }
        
        break;

    default:
        printk("Test message received!");
        break;
    }
}

static int __init netlink_init(void)
{
    struct netlink_kernel_cfg cfg = {
        .input = receive_msg,
    };

    nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &cfg);
    if (!nl_sk) {
        printk(KERN_ALERT "Error creating netlink socket.\n");
        return -10;
    }

    printk(KERN_INFO "Netlink socket created successfully.\n");
    return 0;
}

static void __exit netlink_exit(void)
{
    netlink_kernel_release(nl_sk);
    printk(KERN_INFO "Netlink socket released.\n");
}

static void print_bictcp(struct bictcp *cubic)
{
    printk("BiC-TCP State:\n");
    printk("[DEBUG] cnt: %d\n", cubic->cnt);
    printk("[DEBUG] last_max_cwnd: %d\n", cubic->last_max_cwnd);
    printk("[DEBUG] last_cwnd: %d\n", cubic->last_cwnd);
    printk("[DEBUG] last_time: %d\n", cubic->last_time);
    printk("[DEBUG] bic_origin_point: %d\n", cubic->bic_origin_point);
    printk("[DEBUG] bic_K: %d\n", cubic->bic_K);
    printk("[DEBUG] delay_min: %d\n", cubic->delay_min);
    printk("[DEBUG] ack_cnt: %d\n", cubic->ack_cnt);
    printk("[DEBUG] tcp_cwnd: %d\n", cubic->tcp_cwnd);
    printk("[DEBUG] found: %d\n", cubic->found);
}

static void print_hybla(struct hybla *hybla)
{
    printk("Hybla State:\n");
    printk("[DEBUG] hybla_en: %d\n", hybla->hybla_en);
    printk("[DEBUG] snd_cwnd_cents: %d\n", hybla->snd_cwnd_cents);
    printk("[DEBUG] rho: %d\n", hybla->rho);
    printk("[DEBUG] rho2: %d\n", hybla->rho2);
    printk("[DEBUG] rho_3ls: %d\n", hybla->rho_3ls);
    printk("[DEBUG] rho2_7ls: %d\n", hybla->rho2_7ls);
    printk("[DEBUG] minrtt_us: %d\n", hybla->minrtt_us);
}


static void print_mutant_state(struct sock *sk)
{
    if (selected_proto_id == CUBIC && saved_states && saved_states->cubic_state) {
        memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
        print_bictcp(saved_states->cubic_state);
    } else if (selected_proto_id == HYBLA && saved_states && saved_states->hybla_state) {
        memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
        print_hybla(saved_states->hybla_state);
    }
}

static void save_state(struct sock *sk)
{
    if (!saved_states) {
        pr_err("saved_states not initialized\n");
        return;
    }

    switch (prev_proto_id) {
    case CUBIC:
        kfree(saved_states->cubic_state);
        saved_states->cubic_state = kmalloc(sizeof(struct bictcp), GFP_KERNEL);
        if (saved_states->cubic_state)
            memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
        break;

    case HYBLA:
        kfree(saved_states->hybla_state);
        saved_states->hybla_state = kmalloc(sizeof(struct hybla), GFP_KERNEL);
        if (saved_states->hybla_state)
            memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
        break;

    case BBR1:
        kfree(saved_states->bbr1_state);
        saved_states->bbr1_state = kmalloc(sizeof(struct bbr1), GFP_KERNEL);
        if (saved_states->bbr1_state)
            memcpy(saved_states->bbr1_state, inet_csk_ca(sk), sizeof(struct bbr1));
        break;

    case BBR3:
        kfree(saved_states->bbr3_state);
        saved_states->bbr3_state = kmalloc(sizeof(struct bbr3), GFP_KERNEL);
        if (saved_states->bbr3_state)
            memcpy(saved_states->bbr3_state, inet_csk_ca(sk), sizeof(struct bbr3));
        break;

    case WESTWOOD:
        kfree(saved_states->westwood_state);
        saved_states->westwood_state = kmalloc(sizeof(struct westwood), GFP_KERNEL);
        if (saved_states->westwood_state)
            memcpy(saved_states->westwood_state, inet_csk_ca(sk), sizeof(struct westwood));
        break;

    case VENO:
        kfree(saved_states->veno_state);
        saved_states->veno_state = kmalloc(sizeof(struct veno), GFP_KERNEL);
        if (saved_states->veno_state)
            memcpy(saved_states->veno_state, inet_csk_ca(sk), sizeof(struct veno));
        break;

    case VEGAS:
        kfree(saved_states->vegas_state);
        saved_states->vegas_state = kmalloc(sizeof(struct vegas), GFP_KERNEL);
        if (saved_states->vegas_state)
            memcpy(saved_states->vegas_state, inet_csk_ca(sk), sizeof(struct vegas));
        break;

    case YEAH:
        kfree(saved_states->yeah_state);
        saved_states->yeah_state = kmalloc(sizeof(struct yeah), GFP_KERNEL);
        if (saved_states->yeah_state)
            memcpy(saved_states->yeah_state, inet_csk_ca(sk), sizeof(struct yeah));
        break;

    case CDG:
        kfree(saved_states->cdg_state);
        saved_states->cdg_state = kmalloc(sizeof(struct cdg), GFP_KERNEL);
        if (saved_states->cdg_state)
            memcpy(saved_states->cdg_state, inet_csk_ca(sk), sizeof(struct cdg));
        break;

    case BIC:
        kfree(saved_states->bic_state);
        saved_states->bic_state = kmalloc(sizeof(struct bic), GFP_KERNEL);
        if (saved_states->bic_state)
            memcpy(saved_states->bic_state, inet_csk_ca(sk), sizeof(struct bic));
        break;

    case HTCP:
        kfree(saved_states->htcp_state);
        saved_states->htcp_state = kmalloc(sizeof(struct htcp), GFP_KERNEL);
        if (saved_states->htcp_state)
            memcpy(saved_states->htcp_state, inet_csk_ca(sk), sizeof(struct htcp));
        break;

    case HIGHSPEED:
        kfree(saved_states->highspeed_state);
        saved_states->highspeed_state = kmalloc(sizeof(struct hstcp), GFP_KERNEL);
        if (saved_states->highspeed_state)
            memcpy(saved_states->highspeed_state, inet_csk_ca(sk), sizeof(struct hstcp));
        break;

    case ILLINOIS:
        kfree(saved_states->illinois_state);
        saved_states->illinois_state = kmalloc(sizeof(struct illinois), GFP_KERNEL);
        if (saved_states->illinois_state)
            memcpy(saved_states->illinois_state, inet_csk_ca(sk), sizeof(struct illinois));
        break;

    case ASTRAEA:
        kfree(saved_states->astraea_state);
        saved_states->astraea_state = kmalloc(sizeof(struct astraea), GFP_KERNEL);
        if (saved_states->astraea_state)
            memcpy(saved_states->astraea_state, inet_csk_ca(sk), sizeof(struct astraea));
        break;

    default:
        break;
    }
}

static void load_state(struct sock *sk)
{
    struct tcp_congestion_ops *cubic     = &cubictcp;
    struct tcp_congestion_ops *hybla     = &tcp_hybla;
    struct tcp_congestion_ops *bbr1      = &tcp_bbr1_cong_ops;
    struct tcp_congestion_ops *bbr3      = &tcp_bbr3_cong_ops;
    struct tcp_congestion_ops *westwood  = &tcp_westwood;
    struct tcp_congestion_ops *veno      = &tcp_veno;
    struct tcp_congestion_ops *vegas     = &tcp_vegas;
    struct tcp_congestion_ops *yeah      = &tcp_yeah;
    struct tcp_congestion_ops *cdg       = &tcp_cdg;
    struct tcp_congestion_ops *tcp_bic   = &bic;
    struct tcp_congestion_ops *tcp_htcp  = &htcp;
    struct tcp_congestion_ops *highspeed = &tcp_highspeed;
    struct tcp_congestion_ops *illinois  = &tcp_illinois;
    struct tcp_congestion_ops *astraea   = &tcp_astraea_ops;

    if (!saved_states) {
        pr_err("saved_states not initialized\n");
        return;
    }

    switch (selected_proto_id) {
    case CUBIC:
        if (saved_states->cubic_state) {
            printk("%s: Loading Cubic state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->cubic_state, sizeof(struct bictcp));
        } else {
            printk("%s: Initializing Cubic state.\n", __FUNCTION__);
            saved_states->cubic_state = kmalloc(sizeof(struct bictcp), GFP_KERNEL);
            if (!saved_states->cubic_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct bictcp));
            if (cubic->init)
                cubic->init(sk);
            memcpy(saved_states->cubic_state, inet_csk_ca(sk), sizeof(struct bictcp));
        }
        break;

    case HYBLA:
        if (saved_states->hybla_state) {
            printk("%s: Loading Hybla state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->hybla_state, sizeof(struct hybla));
        } else {
            printk("%s: Initializing Hybla state.\n", __FUNCTION__);
            saved_states->hybla_state = kmalloc(sizeof(struct hybla), GFP_KERNEL);
            if (!saved_states->hybla_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct hybla));
            if (hybla->init)
                hybla->init(sk);
            memcpy(saved_states->hybla_state, inet_csk_ca(sk), sizeof(struct hybla));
        }
        break;

    case BBR1:
        if (saved_states->bbr1_state) {
            printk("%s: Loading BBR1 state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->bbr1_state, sizeof(struct bbr1));
        } else {
            printk("%s: Initializing BBR1 state.\n", __FUNCTION__);
            saved_states->bbr1_state = kmalloc(sizeof(struct bbr1), GFP_KERNEL);
            if (!saved_states->bbr1_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct bbr1));
            if (bbr1->init)
                bbr1->init(sk);
            memcpy(saved_states->bbr1_state, inet_csk_ca(sk), sizeof(struct bbr1));
        }
        break;

    case BBR3:
        if (saved_states->bbr3_state) {
            printk("%s: Loading BBR3 state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->bbr3_state, sizeof(struct bbr3));
        } else {
            printk("%s: Initializing BBR3 state.\n", __FUNCTION__);
            saved_states->bbr3_state = kmalloc(sizeof(struct bbr3), GFP_KERNEL);
            if (!saved_states->bbr3_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct bbr3));
            if (tcp_bbr3_cong_ops.init)
                tcp_bbr3_cong_ops.init(sk);
            memcpy(saved_states->bbr3_state, inet_csk_ca(sk), sizeof(struct bbr3));
        }
        break;

    case WESTWOOD:
        if (saved_states->westwood_state) {
            printk("%s: Loading Westwood state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->westwood_state, sizeof(struct westwood));
        } else {
            printk("%s: Initializing Westwood state.\n", __FUNCTION__);
            saved_states->westwood_state = kmalloc(sizeof(struct westwood), GFP_KERNEL);
            if (!saved_states->westwood_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct westwood));
            if (westwood->init)
                westwood->init(sk);
            memcpy(saved_states->westwood_state, inet_csk_ca(sk), sizeof(struct westwood));
        }
        break;

    case VENO:
        if (saved_states->veno_state) {
            printk("%s: Loading Veno state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->veno_state, sizeof(struct veno));
        } else {
            printk("%s: Initializing Veno state.\n", __FUNCTION__);
            saved_states->veno_state = kmalloc(sizeof(struct veno), GFP_KERNEL);
            if (!saved_states->veno_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct veno));
            if (veno->init)
                veno->init(sk);
            memcpy(saved_states->veno_state, inet_csk_ca(sk), sizeof(struct veno));
        }
        break;

    case VEGAS:
        if (saved_states->vegas_state) {
            printk("%s: Loading Vegas state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->vegas_state, sizeof(struct vegas));
        } else {
            printk("%s: Initializing Vegas state.\n", __FUNCTION__);
            saved_states->vegas_state = kmalloc(sizeof(struct vegas), GFP_KERNEL);
            if (!saved_states->vegas_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct vegas));
            if (vegas->init)
                vegas->init(sk);
            memcpy(saved_states->vegas_state, inet_csk_ca(sk), sizeof(struct vegas));
        }
        break;

    case YEAH:
        if (saved_states->yeah_state) {
            printk("%s: Loading Yeah state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->yeah_state, sizeof(struct yeah));
        } else {
            printk("%s: Initializing Yeah state.\n", __FUNCTION__);
            saved_states->yeah_state = kmalloc(sizeof(struct yeah), GFP_KERNEL);
            if (!saved_states->yeah_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct yeah));
            if (yeah->init)
                yeah->init(sk);
            memcpy(saved_states->yeah_state, inet_csk_ca(sk), sizeof(struct yeah));
        }
        break;

    case CDG:
        if (saved_states->cdg_state) {
            printk("%s: Loading CDG state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->cdg_state, sizeof(struct cdg));
        } else {
            printk("%s: Initializing CDG state.\n", __FUNCTION__);
            saved_states->cdg_state = kmalloc(sizeof(struct cdg), GFP_KERNEL);
            if (!saved_states->cdg_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct cdg));
            if (cdg->init)
                cdg->init(sk);
            memcpy(saved_states->cdg_state, inet_csk_ca(sk), sizeof(struct cdg));
        }
        break;

    case BIC:
        if (saved_states->bic_state) {
            printk("%s: Loading BIC state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->bic_state, sizeof(struct bic));
        } else {
            printk("%s: Initializing BIC state.\n", __FUNCTION__);
            saved_states->bic_state = kmalloc(sizeof(struct bic), GFP_KERNEL);
            if (!saved_states->bic_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct bic));
            if (tcp_bic->init)
                tcp_bic->init(sk);
            memcpy(saved_states->bic_state, inet_csk_ca(sk), sizeof(struct bic));
        }
        break;

    case HTCP:
        if (saved_states->htcp_state) {
            printk("%s: Loading HTCP state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->htcp_state, sizeof(struct htcp));
        } else {
            printk("%s: Initializing HTCP state.\n", __FUNCTION__);
            saved_states->htcp_state = kmalloc(sizeof(struct htcp), GFP_KERNEL);
            if (!saved_states->htcp_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct htcp));
            if (tcp_htcp->init)
                tcp_htcp->init(sk);
            memcpy(saved_states->htcp_state, inet_csk_ca(sk), sizeof(struct htcp));
        }
        break;

    case HIGHSPEED:
        if (saved_states->highspeed_state) {
            printk("%s: Loading Highspeed state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->highspeed_state, sizeof(struct hstcp));
        } else {
            printk("%s: Initializing Highspeed state.\n", __FUNCTION__);
            saved_states->highspeed_state = kmalloc(sizeof(struct hstcp), GFP_KERNEL);
            if (!saved_states->highspeed_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct hstcp));
            if (highspeed->init)
                highspeed->init(sk);
            memcpy(saved_states->highspeed_state, inet_csk_ca(sk), sizeof(struct hstcp));
        }
        break;

    case ILLINOIS:
        if (saved_states->illinois_state) {
            printk("%s: Loading Illinois state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->illinois_state, sizeof(struct illinois));
        } else {
            printk("%s: Initializing Illinois state.\n", __FUNCTION__);
            saved_states->illinois_state = kmalloc(sizeof(struct illinois), GFP_KERNEL);
            if (!saved_states->illinois_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct illinois));
            if (illinois->init)
                illinois->init(sk);
            memcpy(saved_states->illinois_state, inet_csk_ca(sk), sizeof(struct illinois));
        }
        break;

    case ASTRAEA:
        if (saved_states->astraea_state) {
            printk("%s: Loading Astraea state.\n", __FUNCTION__);
            memcpy(inet_csk_ca(sk), saved_states->astraea_state, sizeof(struct astraea));
        } else {
            printk("%s: Initializing Astraea state.\n", __FUNCTION__);
            saved_states->astraea_state = kmalloc(sizeof(struct astraea), GFP_KERNEL);
            if (!saved_states->astraea_state)
                return;
            memset(inet_csk_ca(sk), 0, sizeof(struct astraea));
            if (astraea->init)
                astraea->init(sk);
            memcpy(saved_states->astraea_state, inet_csk_ca(sk), sizeof(struct astraea));
        }
        break;

    default:
        printk("%s: unknown proto id=%u\n", __FUNCTION__, selected_proto_id);
        break;
    }
}

static void mutant_switch_congestion_control(void)
{
    switch (selected_proto_id) {
    case CUBIC:
        printk(KERN_INFO "Switching to Cubic (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &cubictcp;
        break;
    case HYBLA:
        printk(KERN_INFO "Switching to Hybla (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_hybla;
        break;
    case BBR1:
        printk(KERN_INFO "Switching to BBR1 (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_bbr1_cong_ops;
        break;
    case BBR3:
        printk(KERN_INFO "Switching to BBR3 (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_bbr3_cong_ops;
        break;
    case WESTWOOD:
        printk(KERN_INFO "Switching to Westwood (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_westwood;
        break;
    case VENO:
        printk(KERN_INFO "Switching to Veno (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_veno;
        break;
    case VEGAS:
        printk(KERN_INFO "Switching to Vegas (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_vegas;
        break;
    case YEAH:
        printk(KERN_INFO "Switching to Yeah (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_yeah;
        break;
    case CDG:
        printk(KERN_INFO "Switching to CDG (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_cdg;
        break;
    case BIC:
        printk(KERN_INFO "Switching to BIC (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &bic;
        break;
    case HTCP:
        printk(KERN_INFO "Switching to HTCP (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &htcp;
        break;
    case HIGHSPEED:
        printk(KERN_INFO "Switching to Highspeed (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_highspeed;
        break;
    case ILLINOIS:
        printk(KERN_INFO "Switching to Illinois (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_illinois;
        break;
    case ASTRAEA:
        printk(KERN_INFO "Switching to Astraea (ID: %d)", selected_proto_id);
        mutant_wrapper.current_ops = &tcp_astraea_ops;
        break;
    default:
        mutant_wrapper.current_ops = &cubictcp;
        break;
    }
}

static void send_info(struct mutant_info *info)
{
    char msg[MAX_PAYLOAD - 1];

    snprintf(msg, MAX_PAYLOAD - 1,
        "%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%llu;%u;%u;%llu;%u",
        info->now, info->snd_cwnd, info->rtt_us, info->srtt_us,
        info->mdev_us, info->min_rtt, info->advmss, info->delivered,
        info->lost_out, info->packets_out, info->retrans_out,
        info->rate, info->prev_proto_id, info->selected_proto_id,
        info->thruput, info->loss_rate);

    send_msg(msg, socketId);
}

static void send_net_params(struct tcp_sock *tp, struct sock *sk, int socketId)
{
    u32 rate = READ_ONCE(tp->rate_delivered);
    u32 intv = READ_ONCE(tp->rate_interval_us);

    if (tp->packets_out + tp->retrans_out > 0)
        loss_rate = ((u64)tp->lost_out * 100) / (tp->packets_out + tp->retrans_out);

    if (rate && intv) {
        thruput = (u64)rate * tp->mss_cache * USEC_PER_SEC * 8;
        do_div(thruput, intv);
    } else {
        thruput = 0;
    }

    info.now = tcp_jiffies32;
    info.snd_cwnd = tp->snd_cwnd;
    info.rtt_us = tp->rack.rtt_us;
    info.srtt_us = tp->srtt_us;
    info.mdev_us = tp->mdev_us;
    info.min_rtt = tcp_min_rtt(tp);
    info.advmss = tp->advmss;
    info.delivered = tp->delivered;
    info.lost_out = tp->lost_out;
    info.packets_out = tp->packets_out;
    info.retrans_out = tp->retrans_out;
    info.rate = rate;
    info.prev_proto_id = prev_proto_id;
    info.selected_proto_id = selected_proto_id;
    info.thruput = thruput;
    info.loss_rate = loss_rate;

    send_info(&info);
    (void)sk;
    (void)socketId;
}

static void mutant_tcp_init(struct sock *sk)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->init) {
        mutant_wrapper.current_ops->init(sk);
        printk("Mutant %s: init %s", __FUNCTION__, mutant_wrapper.current_ops->name);
    }
    printk("[mutant] %s: init %s", __FUNCTION__, mutant_wrapper.current_ops->name);
}

static void mutant_tcp_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
    struct tcp_sock *tp = tcp_sk(sk);
    u32 before = tp->snd_cwnd;

    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cong_avoid) {
        mutant_wrapper.current_ops->cong_avoid(sk, ack, acked);
        //printk(KERN_INFO "[mutant] cong_avoid algo=%s cwnd %u -> %u", mutant_wrapper.current_ops->name, before, tp->snd_cwnd);
    }
}

static u32 mutant_tcp_ssthresh(struct sock *sk)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->ssthresh)
        return mutant_wrapper.current_ops->ssthresh(sk);
    return TCP_INFINITE_SSTHRESH;
}

static void mutant_tcp_set_state(struct sock *sk, u8 new_state)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->set_state)
        mutant_wrapper.current_ops->set_state(sk, new_state);
}

static u32 mutant_tcp_undo_cwnd(struct sock *sk)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->undo_cwnd)
        return mutant_wrapper.current_ops->undo_cwnd(sk);
    return tcp_sk(sk)->snd_cwnd;
}

static void mutant_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cwnd_event)
        mutant_wrapper.current_ops->cwnd_event(sk, event);
}

static void mutant_tcp_pkts_acked(struct sock *sk, const struct ack_sample *sample)
{
    struct tcp_sock *tp = tcp_sk(sk);

    if (switching_flag) {
        save_state(sk);
        mutant_switch_congestion_control();
        load_state(sk);
        switching_flag = false;
    }

    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->pkts_acked)
        mutant_wrapper.current_ops->pkts_acked(sk, sample);

    //printk(KERN_INFO "[mutant] pkts_acked algo=%s cwnd=%u ssthresh=%u", mutant_wrapper.current_ops ? mutant_wrapper.current_ops->name : "none", tp->snd_cwnd, tp->snd_ssthresh);

    if (socketId != -1)
        send_net_params(tp, sk, socketId);
}

static void mutant_tcp_ack_event(struct sock *sk, u32 flags)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->in_ack_event)
        mutant_wrapper.current_ops->in_ack_event(sk, flags);
}

static u32 mutant_tcp_cong_control(struct sock *sk, const struct rate_sample *rs,
                                   u32 ack, u32 acked, int flag)
{
    struct tcp_sock *tp = tcp_sk(sk);

    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->cong_control) {
        mutant_wrapper.current_ops->cong_control(sk, ack, flag, rs);
        //printk(KERN_INFO "[mutant] cong_control handled by %s cwnd=%u ack=%u acked=%u flag=%d", mutant_wrapper.current_ops->name, tp->snd_cwnd, ack, acked, flag);
        return 0;
    }

    //printk(KERN_INFO "[mutant] cong_control not handled by %s", mutant_wrapper.current_ops ? mutant_wrapper.current_ops->name : "none");
    return -1;
}

static u32 mutant_tcp_sndbuf_expand(struct sock *sk)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->sndbuf_expand)
        return mutant_wrapper.current_ops->sndbuf_expand(sk);
    return 2;
}

static u32 mutant_tcp_min_tso_segs(struct sock *sk, unsigned int mss)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->tso_segs)
        return mutant_wrapper.current_ops->tso_segs(sk, mss);
    return 2;
}

static size_t mutant_tcp_get_info(struct sock *sk, u32 ext, int *attr,
                                  union tcp_cc_info *info)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->get_info)
        return mutant_wrapper.current_ops->get_info(sk, ext, attr, info);
    return 0;
}

static void mutant_tcp_release(struct sock *sk)
{
    if (mutant_wrapper.current_ops && mutant_wrapper.current_ops->release)
        mutant_wrapper.current_ops->release(sk);
}

static struct tcp_congestion_ops mutant_cong_ops __read_mostly = {
    .flags                   = TCP_CONG_NON_RESTRICTED,
    .init                    = mutant_tcp_init,
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
    .release                 = mutant_tcp_release,
    .owner                   = THIS_MODULE,
    .name                    = "mutant",
};

static int __init mutant_tcp_module_init(void)
{
    if (netlink_init() < 0) {
        pr_err("Netlink could not be initialized\n");
        return -EINVAL;
    }

    init_saved_states();
    mutant_wrapper.current_ops = &cubictcp;

    if (tcp_register_congestion_control(&mutant_cong_ops) < 0) {
        pr_err("Mutant congestion control could not be registered\n");
        return -EINVAL;
    }

    return 0;
}

static void __exit mutant_tcp_module_exit(void)
{
    netlink_exit();
    free_saved_states();
    tcp_unregister_congestion_control(&mutant_cong_ops);
}

module_init(mutant_tcp_module_init);
module_exit(mutant_tcp_module_exit);
MODULE_AUTHOR("Lorenzo Pappone");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Mutant");
MODULE_VERSION("1.0");