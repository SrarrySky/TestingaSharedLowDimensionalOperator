# 🔗 Paper-to-code and evidence map

This map accompanies *Testing a Shared Low-Dimensional Operator Representation for Time-Varying Financial Dynamics* by Jiyu Mo and Dongyang Kuang, prepared for an ICMLC 2027 submission. It identifies the implementation and configuration for each research component. Section names are used in place of unverified proceedings page numbers. The manuscript working directory is outside the public release.

| Paper component | Implementation | Configuration or check | Scope |
|---|---|---|---|
| A-share sector data and fixed chronological split | `src/core/data.py` | `config/data.yaml`, `data/README.md` | Requires separately obtained Tushare data |
| Admissible directional support | `src/causal/causal_support.py` | `config/causal.yaml`, `tests/test_causal_support.py` | Main branch is lagged Lasso stability selection; support is predictive, not an identified causal graph |
| Shared latent operator and penalized estimation | `src/model/dlcssm.py`, `src/model/kalman_em.py` | `config/dlcssm.yaml`, `tests/test_penalized_em.py` | Train fit; forward filter for Test |
| Gauge and low-rank geometry checks | `src/model/geometry.py` | `tests/test_theory.py`, `scripts/tvp_lowrank_diagnostic.py`, [derivations, Sections 6–8](../paper_supplement/MATHEMATICAL_DERIVATIONS.md#6-proposition-1-latent-coordinate-invariance) | Numerical checks under stated assumptions |
| Frozen and pathwise propagation | `src/model/propagation.py`, `src/oos/pathwise_oos.py` | `tests/test_propagation.py`, `tests/test_pathwise_oos.py`, [derivations, Sections 9–15](../paper_supplement/MATHEMATICAL_DERIVATIONS.md#9-companion-representation-and-coordinate-responses) | Conditional on fitted transition and uncertainty model |
| Static/Sparse VAR, TVP-VAR, smooth VAR, HMM, fixed-network baselines | `src/baselines/`, `src/experiments/baseline_experiment.py` | `config/baselines.json` | Comparators and risk summaries |
| Synthetic mechanism comparison | `src/experiments/synthetic_benchmark.py` | `config/synthetic_main.json` | Full run required for empirical results; smoke test is smaller |
| Five-day downside target and M0–M5 predictors | `src/core/evaluation.py`, `src/oos/oos_final.py`, `src/oos/oos_specs.py` | `config/oos_final.json`, `config/oos_predictive.json` | Common horizon-purged rows, Train scaling, Validation selection, Test score |
| Staged publication run | `scripts/run_main.py` | `config/main_runner.json` | Stage manifests and ledgers must be inspected |
| Post-acceptance S5 and block sensitivity | `scripts/verify_synthetic_run.py`, `scripts/summarize_oos_block_sensitivity.py` | [reproducibility runbook](REPRODUCIBILITY.md#post-acceptance-supplement-experiments), [robustness results](../paper_supplement/ROBUSTNESS_AND_UNCERTAINTY_RESULTS.md) | Later analysis, separate from published-paper claims |

The project-level index is in the [README](../README.md), implementation details are in [Methods](METHODS.md), complete mathematical arguments are in the [derivation supplement](../paper_supplement/MATHEMATICAL_DERIVATIONS.md), and post-acceptance numerical evidence is in the [robustness and uncertainty supplement](../paper_supplement/ROBUSTNESS_AND_UNCERTAINTY_RESULTS.md). The repository does not bundle the licensed raw data or full local run directories. Tests show that code paths satisfy their checked contracts; they do not establish the empirical findings without the corresponding run outputs.
