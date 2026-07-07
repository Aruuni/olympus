"""Reward for CleanSlate's normalized RayNet observations."""


class RewardCalc:
    def __init__(self, loss_weight=5.0):
        self.loss_weight = float(loss_weight)
        self.last_components = {}

    def step(self, info: dict) -> float:
        throughput = float(
            info.get('throughput_norm', info.get('avg_thr', 0.0)) or 0.0)
        loss = float(info.get('loss_norm', info.get('loss_rate', 0.0)) or 0.0)
        delay_metric = float(info.get('delay_metric', 1.0) or 0.0)
        reward = max(0.0, (throughput - self.loss_weight * loss) * delay_metric)
        self.last_components = {
            'throughput': throughput,
            'loss': loss,
            'delay_metric': delay_metric,
            'unclipped': (throughput - self.loss_weight * loss) * delay_metric,
        }
        return reward

    @property
    def max_tput(self) -> float:
        return 1.0

    @property
    def min_rtt_us(self) -> float:
        return 0.0

    @property
    def kalman_min_rtt_us(self) -> float:
        return 0.0


def make_reward_calc() -> RewardCalc:
    return RewardCalc()
