import argparse
import re
import subprocess
import sys
import time

from mininet.net import Mininet
from mininet.node import Switch, Host
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.log import setLogLevel

try:
    from olympus.environments.base import NetworkEnv
except Exception:  # allow running this file standalone, outside the package
    NetworkEnv = object


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
    def build(self, n=1, prefix=''):
        s1 = self.addSwitch(f'{prefix}s1', cls=LinuxBridgeSwitch)
        s2 = self.addSwitch(f'{prefix}s2', cls=LinuxBridgeSwitch)
        s3 = self.addSwitch(f'{prefix}s3', cls=LinuxBridgeSwitch)

        self.addLink(s1, s2)
        self.addLink(s2, s3)

        for i in range(1, n + 1):
            c = self.addHost(f'{prefix}c{i}', cls=Host)
            self.addLink(c, s1)
            x = self.addHost(f'{prefix}x{i}', cls=Host)
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


class MininetEnv(NetworkEnv):
    """
    Simple dumbbell Mininet environment.

    Topology:  c1..cn -> s1 --[delay]--> s2 --[bw/queue]--> s3 -> x1..xn

    Parameters:
        n           - number of sender/receiver pairs
        bw          - bottleneck bandwidth in Mbps
        delay       - one-way propagation delay in ms (applied on s1-s2 link)
        qsize       - bottleneck queue size in bytes (overrides bdp_mult if set)
        bdp_mult    - queue size as a multiple of BDP (default 1.0)
        loss        - packet loss percentage on s1-s2 (float, optional)
        duration    - iperf3 flow duration in seconds
        cport       - iperf3 client source port (oc_bridge matches on this)
        cc_algo     - TCP congestion control algorithm name
        unique_cports - when true, use cport+i-1 per flow. The default keeps
                      the same cport across network namespaces so one listener
                      can attach to every flow.
        instance_id - unique integer per parallel instance (None = legacy mode,
                      no prefix). Each instance gets isolated bridge names
                      i{id}s1/s2/s3 and its own iperf3 server port 5201+id.
        per_flow_delays - optional list of one-way delays in ms, one per flow,
                      applied as netem on each sender's access link
                      (c_i -> s1). When set, the shared s1-s2 link carries NO
                      propagation delay, so flow i's RTT ~= per_flow_delays[i-1]
                      alone (inter-RTT fairness); `delay` then sizes the
                      bottleneck BDP buffer and serves as the reference for
                      live delay changes: set_link(delay=X) scales every
                      flow's own access-link delay by X/delay. None leaves the
                      topology identical to legacy behaviour (delay on s1-s2).
    """

    def __init__(self, n=1, bw=10, delay=20, qsize=None, bdp_mult=1.0,
                 loss=None, duration=60, cport=11111, cc_algo='mutant',
                 instance_id=None, unique_cports=False, per_flow_delays=None):
        self.n        = n
        self.bw       = bw
        self.delay    = delay
        self.bdp_mult = bdp_mult
        bdp           = bw * (2 ** 20) * delay * 1e-3 / 8
        self.qsize    = qsize if qsize is not None else max(int(bdp_mult * bdp), 1500)
        self.loss     = loss
        self.duration = duration
        self.cport    = cport
        self.cc_algo  = cc_algo
        self.unique_cports = unique_cports
        self.per_flow_delays = (
            [float(d) for d in per_flow_delays] if per_flow_delays else None)
        # Episode base delay, kept immutable so scheduled delay changes can
        # scale per-flow delays as a fraction of it (set_link mutates
        # self.delay for BDP sizing).
        self._base_delay = float(delay)
        self.net      = None

        # Parallel-instance isolation
        self.prefix     = f'i{instance_id}' if instance_id is not None else ''
        self.iperf_port = 5201 + (instance_id if instance_id is not None else 0)

    def _cleanup_leftover_interfaces(self):
        if not self.prefix:
            return
        pattern = re.compile(
            r'^{}(?:c|x|s)\d+-eth\d+$'.format(re.escape(self.prefix)))
        try:
            out = subprocess.run(
                ['ip', '-o', 'link', 'show'],
                capture_output=True, text=True, check=False)
        except OSError:
            return
        for line in out.stdout.splitlines():
            parts = line.split(':', 2)
            if len(parts) < 2:
                continue
            name = parts[1].strip().split('@', 1)[0]
            if not pattern.match(name):
                continue
            subprocess.run(
                ['ip', 'link', 'del', name],
                capture_output=True, check=False)

    def start(self):
        p = self.prefix
        # Clean up any leftover bridges from a previous episode on this slot.
        for br in [f'{p}s1', f'{p}s2', f'{p}s3']:
            subprocess.run(
                f'ip link set {br} down 2>/dev/null; ip link del {br} 2>/dev/null; true',
                shell=True, capture_output=True)
        self._cleanup_leftover_interfaces()

        setLogLevel('warning')
        topo = DumbbellTopo(n=self.n, prefix=p)
        self.net = Mininet(topo=topo, link=TCLink, switch=LinuxBridgeSwitch,
                           controller=None, autoSetMacs=True, autoStaticArp=True)
        self.net.start()
        self._configure_links()

    def _s1s2_intfs(self):
        p = self.prefix
        s1, s2 = self.net.get(f'{p}s1', f'{p}s2')
        for intf in s1.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s2:
                    return (s1, intf.name, s2, peer.name)
        raise RuntimeError(f'{p}s1-{p}s2 link not found')

    def _s2s3_intfs(self):
        p = self.prefix
        s2, s3 = self.net.get(f'{p}s2', f'{p}s3')
        for intf in s2.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s3:
                    return (s2, intf.name, s3, peer.name)
        raise RuntimeError(f'{p}s2-{p}s3 link not found')

    def set_link(self, bw=None, delay=None, loss=None):
        """
        Live-update bottleneck bandwidth and/or propagation delay.
        Thread-safe: can be called from a background scheduler thread.
        """
        if bw    is not None: self.bw    = bw
        if delay is not None: self.delay = delay
        if loss  is not None: self.loss  = loss

        if bw is not None or delay is not None:
            bdp = self.bw * (2 ** 20) * self.delay * 1e-3 / 8
            self.qsize = max(int(self.bdp_mult * bdp), 1500)

        if bw is not None or delay is not None:
            s2, s2_intf, s3, s3_intf = self._s2s3_intfs()
            _change_bw(s2, s2_intf, self.bw, self.qsize)
            _change_bw(s3, s3_intf, self.bw, self.qsize)

        if delay is not None:
            if self.per_flow_delays:
                # No shared s1-s2 netem exists in per-flow-RTT mode: each
                # flow's delay lives on its own access link. Scale every
                # flow's delay by the same fraction of the episode base so a
                # scheduled change moves all RTTs while preserving the
                # inter-flow heterogeneity this mode exists to create.
                frac = (float(delay) / self._base_delay
                        if self._base_delay > 0 else 1.0)
                for i in range(1, self.n + 1):
                    base_extra = (self.per_flow_delays[i - 1]
                                  if i - 1 < len(self.per_flow_delays) else 0.0)
                    if not base_extra or base_extra <= 0.0:
                        continue
                    c, intf_name = self._ci_s1_intf(i)
                    if c is None:
                        continue
                    _change_delay(c, intf_name, round(base_extra * frac, 3))
            else:
                s1, s1_intf, s2, s2_intf = self._s1s2_intfs()
                _change_delay(s1, s1_intf, self.delay, self.loss)

    def _ci_s1_intf(self, i):
        """Return (host, intf_name) for sender c{i}'s link toward s1."""
        p = self.prefix
        c  = self.net.get(f'{p}c{i}')
        s1 = self.net.get(f'{p}s1')
        for intf in c.intfList():
            if intf.link and _peer_intf(intf).node is s1:
                return c, intf.name
        return None, None

    def _configure_per_flow_delays(self):
        """Apply each flow's propagation delay on its own sender access link.

        Used for inter-RTT fairness: per_flow_delays[i-1] is the full one-way
        delay (ms) for flow i, applied as netem on the c{i}-s1 access link.
        When this is set the shared s1-s2 link carries NO netem (see
        _configure_links), so each flow's RTT is determined solely by its own
        access link."""
        if not self.per_flow_delays:
            return
        for i in range(1, self.n + 1):
            extra = (self.per_flow_delays[i - 1]
                     if i - 1 < len(self.per_flow_delays) else 0.0)
            if not extra or extra <= 0.0:
                continue
            c, intf_name = self._ci_s1_intf(i)
            if c is None:
                continue
            _configure_link(c, intf_name, delay=extra, loss=None)

    def _configure_links(self):
        p = self.prefix
        s1, s2, s3 = self.net.get(f'{p}s1', f'{p}s2', f'{p}s3')

        # With per-flow delays, every flow's full propagation delay lives on
        # its own sender access link (c_i -> s1), so the shared s1-s2 link
        # carries no netem. self.delay is then used only to size the
        # bottleneck BDP buffer (qsize, computed in __init__/set_link).
        if not self.per_flow_delays:
            for intf in s1.intfList():
                if intf.link:
                    peer = _peer_intf(intf)
                    if peer.node is s2:
                        _configure_link(s1, intf.name, delay=self.delay, loss=self.loss)
                        break

        for intf in s2.intfList():
            if intf.link:
                peer = _peer_intf(intf)
                if peer.node is s3:
                    _configure_link(s2, intf.name, bw=self.bw, qsize=self.qsize)
                    _configure_link(s3, peer.name, bw=self.bw, qsize=self.qsize)
                    break

        self._configure_per_flow_delays()

    def run_iperf(self, monitor_interval=0.1, start_delays=None,
                  flow_durations=None):
        """Launch iperf3 server on every receiver, then iperf3 client on every
        sender. `start_delays` is an optional list of N seconds (one per flow).
        `flow_durations`, when supplied, gives each flow an exact iperf runtime.
        Without `flow_durations`, delayed flows are shortened so they finish at
        the same wall-clock time as the rest of the episode."""
        p    = self.prefix
        port = self.iperf_port
        for i in range(1, self.n + 1):
            x = self.net.get(f'{p}x{i}')
            x.cmd(
                f'iperf3 -p {port} -i {monitor_interval} --one-off --json -s'
                f' > /tmp/iperf_server_{self.cport}_{i}.json 2>&1 &'
            )

        time.sleep(0.5)

        if start_delays is None:
            start_delays = [0.0] * self.n
        if len(start_delays) < self.n:
            start_delays = list(start_delays) + [0.0] * (self.n - len(start_delays))
        if flow_durations is None:
            flow_durations = []
        if len(flow_durations) < self.n:
            flow_durations = list(flow_durations) + [None] * (self.n - len(flow_durations))

        for i in range(1, self.n + 1):
            c = self.net.get(f'{p}c{i}')
            x = self.net.get(f'{p}x{i}')
            delay = max(0.0, float(start_delays[i - 1]))
            if flow_durations[i - 1] is None:
                t_run = max(1.0, float(self.duration) - delay)
            else:
                t_run = max(1.0, float(flow_durations[i - 1]))
            client_cport = self.cport + i - 1 if self.unique_cports else self.cport
            iperf_cmd = (
                f'iperf3 -p {port} --cport={client_cport}'
                f' -i {monitor_interval} -C {self.cc_algo} --json'
                f' -t {t_run:.3f} -c {x.IP()} --forceflush'
                f' > /tmp/iperf_{self.cport}_{i}.json 2>&1'
            )
            if delay > 0.0:
                c.cmd(f'(sleep {delay:.3f} && {iperf_cmd}) &')
            else:
                c.cmd(f'{iperf_cmd} &')

    def stop(self):
        if not self.net:
            return
        self.net.stop()
        self.net = None


# Backend entry point resolved by olympus.environments.make_env.
ENV_CLASS = MininetEnv


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
