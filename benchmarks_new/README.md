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

The paper experiment reproductions can be run separately with:

    sudo -E ./venv_training/bin/python \
      benchmarks_new/benchmark_PAPER_inter_rtt_fairness/inter_rtt_fairness.py

    sudo -E ./venv_training/bin/python \
      benchmarks_new/benchmark_PAPER_intra_rtt_fairness/intra_rtt_fairness.py

    sudo -E ./venv_training/bin/python \
      benchmarks_new/benchmark_paper_efficiency/efficiency.py

Use `--debug` for the short two-condition smoke-test versions.

Render the three paper experiments as one shared figure from their existing
episode data:

    ./venv_training/bin/python benchmarks_new/paper_figure.py

This writes `benchmarks_new/paper_figure.pdf` and a matching PNG. Use
`--debug` to read each experiment's `data_debug` output instead.
