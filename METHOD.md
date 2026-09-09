# GradAudit reproduction

Each model-specific implementation computes assistant-token gradients, constructs the reference statistics and sensitive mask, scores held-out audit samples, and writes per-sample scores plus ROC metrics. The released scripts preserve the configuration used for the corresponding main-table result. Statistical reporting uses 10,000 stratified sample-level bootstrap replicates with the fixed seed recorded by `scripts/bootstrap_scores.py`.

Run `./scripts/run_one.sh MODEL GPU` for an individual experiment or `./run_all.sh GPU` for all six experiments.
