import argparse
import subprocess
import sys
import time

from mininet.net import Mininet
from mininet.node import Switch, Host
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.log import setLogLevel


class LinuxBridgeSwitch(Switch):
    """
    Simple Linux kernel bridge — no OVS, no daemon, cannot hang.
    Uses 'ip link ... type bridge' for L2 forwarding.
    """
    def start(self, controllers):
        self.cmd(f'ip link add name {self.name} type bridge stp_state 0 forward_delay 0')
        self.cmd(f'ip link set {self.name} up')
        for intf in self.intfList():
            if intf.name != 'lo':
                self.cmd(f'ip link set {intf.name} master {self.name}')
                self.cmd(f'ip link set {intf.name} up')

    def stop(self, deleteIntfs=True):
        self.cmd(f'ip link set {self.name} down 2>/dev/null; '
                 f'ip link del {self.name} 2>/dev/null; true')
        super().stop(deleteIntfs)


class DumbbellTopo(Topo):
    def build(self, n=1):
        s1 = self.addSwitch('s1', cls=LinuxBridgeSwitch)
        s2 = self.addSwitch('s2', cls=LinuxBridgeSwitch)
        s3 = self.addSwitch('s3', cls=LinuxBridgeSwitch)

        self.addLink(s1, s2)
        self.addLink(s2, s3)

        for i in range(1, n + 1):
            c = self.addHost(f'c{i}', cls=Host)
            self.addLink(c, s1)
            x = self.addHost(f'x{i}', cls=Host)
            self.addLink(x, s3)


def _configure_link(node, intf_name, bw=None, delay=None, qsize=1000, loss=None):
    """Apply tc qdiscs to an interface on a Mininet node."""
    if delay and not bw:
        cmd = f'tc qdisc add dev {intf_name} root handle 3:0 netem delay {delay}ms limit 100000'
        if loss and float(loss) > 0:
            cmd += f' loss {loss}%'
        node.cmd(cmd)
    elif bw and not delay:
        burst = int(10 * bw * (2 ** 20) / 250 / 8)
        node.cmd(
            f'tc qdisc add dev {intf_name} root handle 1:0 tbf'
            f' rate {bw}mbit burst {burst} limit {qsize}'
        )
    elif bw and delay:
        burst = int(10 * bw * (2 ** 20) / 250 / 8)
        netem = f'netem delay {delay}ms limit 100000'
        if loss and float(loss) > 0:
            netem += f' loss {loss}%'
        node.cmd(f'tc qdisc add dev {intf_name} root handle 1:0 {netem}')
        node.cmd(
            f'tc qdisc add dev {intf_name} parent 1:1 handle 10:0 tbf'
            f' rate {bw}mbit burst {burst} limit {qsize}'
        )


def _change_delay(node, intf_name, delay, loss=None):
    """Live-update netem delay on an interface (handle 3:0, delay-only path)."""
    cmd = f'tc qdisc change dev {intf_name} root handle 3:0 netem delay {delay}ms limit 100000'
    if loss and float(loss) > 0:
        cmd += f' loss {loss}%'
    node.cmd(cmd)


def _change_bw(node, intf_name, bw, qsize):
    """Live-update tbf rate on an interface (handle 1:0, bw-only path)."""
    burst = int(10 * bw * (2 ** 20) / 250 / 8)
    node.cmd(
        f'tc qdisc change dev {intf_name} root handle 1:0 tbf'
        f' rate {bw}mbit burst {burst} limit {qsize}'
    )


def _peer_intf(intf):
    link = intf.link
    return link.intf1 if link.intf2 is intf else link.intf2


class MininetEnv:
    """
    Simple dumbbell Mininet environment (single instance).

    Topology:  c1..cn -> s1 --[delay]--> s2 --[bw/queue]--> s3 -> x1..xn

    Parameters:
        n         - number of sender/receiver pairs
        bw        - bottleneck bandwidth in Mbps
        delay     - one-way propagation delay in ms (applied on s1-s2 link)
        qsize     - bottleneck queue size in bytes (overrides bdp_mult if set)
        bdp_mult  - queue size as a multiple of BDP (default 1.0)
        loss      - packet loss percentage on s1-s2 (float, optional)
        duration  - iperf3 flow duration in seconds
        cport     - iperf3 client source port (oc_listener matches on this)
    """

    def __init__(self, n=1, bw=10, delay=20, qsize=None, bdp_mult=1.0,
                 loss=None, duration=60, cport=11111):
        self.n        = n
        self.bw       = bw
        self.delay    = delay
        self.bdp_mult = bdp_mult
        bdp           = bw * (2 ** 20) * delay * 1e-3 / 8   # delay is RTT, not one-way
        self.qsize    = qsize if qsize is not None else max(int(bdp_mult * bdp), 1500)
        self.loss     = loss
        self.duration = duration
        self.cport    = cport
        self.net      = None

    def start(self):
        # Delete any leftover Linux bridge devices from a previous episode.
        for br in ['s1', 's2', 's3']:
            subprocess.run(f'ip link set {br} down 2>/dev/null; ip link del {br} 2>/dev/null; true',
                           shell=True, capture_output=True)

        setLogLevel('info')
        topo = DumbbellTopo(n=self.n)
        self.net = Mininet(topo=topo, link=TCLink, switch=LinuxBridgeSwitch,
                           controller=None, autoSetMacs=True, autoStaticArp=True)
        self.net.start()
        self._configure_links()

    def _s1s2_intfs(self):
        """Return (s1_intf_name, s2_intf_name) for the s1-s2 link."""
        s1, s2 = self.net.get('s1', 's2')
        for intf in s1.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s2:
                    return (s1, intf.name, s2, peer.name)
        raise RuntimeError('s1-s2 link not found')

    def _s2s3_intfs(self):
        """Return (s2_intf_name, s3_intf_name) for the s2-s3 link."""
        s2, s3 = self.net.get('s2', 's3')
        for intf in s2.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s3:
                    return (s2, intf.name, s3, peer.name)
        raise RuntimeError('s2-s3 link not found')

    def set_link(self, bw=None, delay=None, loss=None):
        """
        Live-update bottleneck bandwidth and/or propagation delay.

        Only the fields that are not None are changed.  Thread-safe: can be
        called from a background scheduler thread while iperf3 is running.

        State is updated first so qsize is always recomputed from the final
        (bw, delay) pair — handles the case where both change at once.
        """
        # 1. Update state with whatever is changing
        if bw    is not None: self.bw    = bw
        if delay is not None: self.delay = delay
        if loss  is not None: self.loss  = loss

        # 2. Recompute BDP-based queue size whenever bw or delay changed
        if bw is not None or delay is not None:
            bdp = self.bw * (2 ** 20) * self.delay * 1e-3 / 8   # delay is RTT
            self.qsize = max(int(self.bdp_mult * bdp), 1500)

        # 3. Apply tc changes — only touch the links that actually changed
        if bw is not None or delay is not None:
            # bw lives on s2-s3 as tbf (handle 1:0); always reapply when
            # either bw or delay changed since qsize may have shifted
            s2, s2_intf, s3, s3_intf = self._s2s3_intfs()
            _change_bw(s2, s2_intf, self.bw, self.qsize)
            _change_bw(s3, s3_intf, self.bw, self.qsize)

        if delay is not None:
            # delay lives on s1 side only (handle 3:0) — one side = RTT = delay
            s1, s1_intf, s2, s2_intf = self._s1s2_intfs()
            _change_delay(s1, s1_intf, self.delay, self.loss)

    def _configure_links(self):
        s1, s2, s3 = self.net.get('s1', 's2', 's3')

        # s1-s2: propagation delay (and optional loss) — applied on s1 side only.
        # Applying on both sides would double the RTT; one side gives RTT = delay.
        for intf in s1.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s2:
                    _configure_link(s1, intf.name, delay=self.delay, loss=self.loss)
                    break

        # s2-s3: bandwidth bottleneck + queue
        for intf in s2.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s3:
                    _configure_link(s2, intf.name, bw=self.bw, qsize=self.qsize)
                    _configure_link(s3, peer.name, bw=self.bw, qsize=self.qsize)
                    break

    def run_iperf(self, monitor_interval=0.1):
        port = 5201
        for i in range(1, self.n + 1):
            x = self.net.get(f'x{i}')
            x.cmd(f'iperf3 -p {port} -i {monitor_interval} --json -s &')

        time.sleep(0.5)

        for i in range(1, self.n + 1):
            c = self.net.get(f'c{i}')
            x = self.net.get(f'x{i}')
            c.cmd(
                f'iperf3 -p {port} --cport={self.cport + i - 1}'
                f' -i {monitor_interval} -C mutant --json'
                f' -t {self.duration} -c {x.IP()} --forceflush'
                f' > /tmp/iperf_{self.cport}_{i}.json 2>&1 &'
            )

    def stop(self):
        if not self.net:
            return
        self.net.stop()
        self.net = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flows',    type=int,   default=1)
    parser.add_argument('--bw',       type=float, default=10.0)
    parser.add_argument('--delay',    type=float, default=20.0)
    parser.add_argument('--qsize',    type=int,   default=None)
    parser.add_argument('--bdp-mult', type=float, default=1.0)
    parser.add_argument('--loss',     type=float, default=None)
    parser.add_argument('--duration', type=int,   default=60)
    parser.add_argument('--cport',    type=int,   default=11111)
    args = parser.parse_args()

    env = MininetEnv(
        n=args.flows, bw=args.bw, delay=args.delay,
        qsize=args.qsize, bdp_mult=args.bdp_mult,
        loss=args.loss, duration=args.duration, cport=args.cport,
    )

    try:
        env.start()
        env.run_iperf()
        time.sleep(args.duration + 2)
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
    sys.exit(0)
