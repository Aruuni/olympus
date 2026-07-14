# Backend-neutral checkpoint benchmarks

Each suite contains an Olympus eval manifest (`config.yaml`), a backend-neutral
scenario (`scenario.yaml`), and a thin runner. Runners call `olympus/eval.py`
and then plot only the state/reward metrics exported by Olympus workers. Kernel,
external Astraea, and legacy Orca baselines are intentionally out of scope.

Olympus model approaches are registered once in `benchmarks_new/config.yaml`.
Each benchmark selects approach names through `matrix.checkpoints`; checkpoint
paths, plot labels, and training metadata come from that shared registry.

Run one suite with its project virtualenv (and `sudo -E` when Mininet is in the
manifest), for example:

    sudo -E ./venv_training/bin/python benchmarks_new/benchmark_fairness/fairness.py

Use `--plot-only` to regenerate the summary from existing standard-profile data.
