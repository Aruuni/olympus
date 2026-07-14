#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = [
    ('benchmark_responsiveness', 'responsiveness.py'),
    ('benchmark_responsive_fairness', 'responsive_fairness.py'),
    ('benchmark_fairness', 'fairness.py'),
    ('benchmark_inter_rtt_fairness', 'inter_rtt_fairness.py'),
    ('benchmark_convergence_4flow', 'convergence_4flow.py'),
]

def main():
    for suite, filename in SUITES:
        runner = HERE / suite / filename
        if subprocess.run([sys.executable, str(runner)], cwd=HERE.parent).returncode:
            return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
