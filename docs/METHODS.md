# 📐 Methods, baselines and OOS design

This document maps the paper's methods to code, defines the competing baseline models, and fixes the out-of-sample (OOS) information protocol for the paper prepared for an ICMLC 2027 submission. This is a method and execution reference; numerical results require the corresponding generated reports. Module commands are repeated where useful; the project entry point is in the [README quick start](../README.md).

For the complete step-by-step proofs and supporting calculations, see [Mathematical derivations](../paper_supplement/MATHEMATICAL_DERIVATIONS.md). For the post-acceptance numerical sensitivity record, see [Robustness and uncertainty results](../paper_supplement/ROBUSTNESS_AND_UNCERTAINTY_RESULTS.md).

## Paper-to-code mapping

| Research object | Code |
|---|---|
| Tushare SW sectors + observed covariates | `src/core/data.py` |
| HMM filtered baseline | `src/baselines/hmm_baseline.py` |
| Admissible causal support | `src/causal/causal_support.py` |
| DLC-SSM parameterization | `src/model/dlcssm.py` |
| Kalman filter / RTS smoother / state M-step | `src/model/kalman_em.py` |
| Pair/path propagation and Gramian | `src/model/propagation.py` |
| Stability, non-normality, sensitivity, reversal geometry | `src/model/geometry.py` |
| Posterior/predictive uncertainty | `src/model/uncertainty.py` |
| Competing synthetic DGPs | `src/experiments/synthetic.py` |
| OOS downside metrics/tests | `src/core/evaluation.py` |

### Theory mapping

- Proposition 1 / gauge invariance: `DLCSSMParameters.apply_gauge`, `tests/test_theory.py::test_gauge_invariance`.
- Proposition 2 / low-rank operator variation: `geometry.low_rank_diagnostics`.
- Proposition 3 / exact frozen-pathwise decomposition: `propagation.frozen_pathwise_decomposition`.
- Corollary 1 / affine state-path bound: `propagation.state_migration_gap`, `tests/test_propagation.py`.
- Propagation-feature derivatives (derivation supplement, Section 10): `geometry.matrix_power_derivative`.
- Affine reversal boundary and sign uncertainty (derivation supplement, Section 15): `geometry.reversal_hyperplane`, `uncertainty.edge_sign_probability`.

The proposition and corollary labels above follow the current manuscript; the full proofs are indexed in the [derivation supplement](../paper_supplement/MATHEMATICAL_DERIVATIONS.md#6-proposition-1-latent-coordinate-invariance).

## Baseline models

### Files

| Path | Role |
|---|---|
| `src/baselines/var_baselines.py` | Static/Sparse VAR, coefficient-state TVP and smooth VAR |
| `src/baselines/network_baseline.py` | Fixed Train-correlation network-operator Kalman baseline |
| `src/baselines/connectedness.py` | VAR responses, generalized FEVD and DY summaries |
| `src/experiments/baseline_experiment.py` | Validation selection and unified comparison export |
| `config/baselines.json` | fixed baseline grid and bootstrap settings |
| `tests/test_var_baselines.py` | numerical and information-isolation tests |

The HMM branch uses the optional `hmmlearn` dependency. Install the project with `[hmm,dev]` when that branch is needed.

### Run

From the project root:

```bash
python -m pip install -e ".[hmm,dev]"
python -m experiments.baseline_experiment --dataset data/processed/dataset.npz --config config/baselines.json --output outputs/baseline_comparison
```

This runs Static VAR, Sparse VAR, TVP-VAR, one-sided smooth TV-VAR, fixed-network SSM and HMM plus soft-regime VAR. HMM is required, not silently omitted if its optional dependency is missing. It also exports DY and TVP risk features. This standalone command does not retrain DLC-SSM automatically; supply a previously generated checkpoint when comparing models on the same sample.

To include a DLC checkpoint in the same-sample comparison:

```bash
python -m experiments.baseline_experiment --dataset data/processed/dataset.npz --config config/baselines.json --dlc-model outputs/checkpoints/dlcssm_train.npz --model-fit-end 2019-12-31 --output outputs/baseline_comparison
```

The endpoint must be the actual checkpoint fitting endpoint and no later than the Train boundary. The date does not prove training history; retain matching dataset/model declarations and reject checkpoints with incompatible Train scaling.

Connect the generated baseline input to the OOS exporter:

```bash
python -m oos.oos_final --dataset data/processed/dataset.npz --model outputs/checkpoints/dlcssm_train.npz --model-fit-end 2019-12-31 --baselines outputs/baseline_comparison/baselines.csv --config config/oos_final.json --output outputs/checkpoints/oos_final_with_baselines
```

Focused and full tests:

```bash
python -m pytest tests/test_var_baselines.py -q
python -m pytest -q
```

Keep `horizon` and `n_regimes` consistent between `config/baselines.json` and `config/oos_final.json`. The baseline producer does not implement BK; never represent DY as BK.

### Outputs

The command writes to `outputs/baseline_comparison/`. Keep a separate directory for every cited run.

| File | Contents |
|---|---|
| `summary.csv` | Validation/Test MSE, raw-return-unit RMSE and mean joint log predictive density for each model |
| `candidates.csv` | All candidates with their Validation MSE, for auditing parameter selection |
| `comparisons.csv` | Moving-block bootstrap 95% intervals for daily MSE differences on the same Test sample |
| `losses.csv` | MSE and joint NLL series aligned by target date |
| `baselines.csv` | Complete HMM/DY/TVP features and date metadata required by the original `oos_final` |
| `baseline_paths.npz` | Full TVP coefficient paths, covariances, DY matrices with to/from/net, forecasts and reproducible parameters |
| `report.json` | Selected configuration, version, HMM restarts, stability and method boundaries |

With DLC, `comparisons.csv` is DLC minus baseline; negative means lower DLC MSE. Without DLC, TVP is the reference. DY is a risk summary of VAR dynamics, not an independent return forecast. Raw-scale RMSE is lower-is-better and log predictive density is higher-is-better. These densities are plug-in quantities, not full Bayesian densities.

Directional output example:

```python
import numpy as np
z = np.load("outputs/baseline_comparison/baseline_paths.npz", allow_pickle=False)
print(z["dates"][-1], z["sector_names"])
print(z["dy_to"][-1], z["dy_from"][-1], z["dy_net"][-1])
print(z["dy_gfevd"][-1])  # [receiving target, shock source]
```

### Definitions

#### Static / Sparse VAR

Includes an unpenalized intercept. Static uses ridge; the underlying `StaticVAR(penalty=0)` class also supports OLS. Sparse uses equation-by-equation Lasso, which genuinely produces zero coefficients; a ridge model is not called a sparse VAR. Lag and penalty are selected solely by Validation MSE; model coefficients are fitted on Train only.

#### TVP-VAR

The observation equation is `x_t = B_t' h_t + epsilon_t`, with `h_t=[1,x_(t-1)',...,x_(t-p)']'`. All elements of the coefficient matrix `B_t` vary freely, with no shared low-dimensional factor constraint:

```
vec(B_t) = vec(B_(t-1)) + eta_t
Cov(epsilon_t) = Sigma
Cov(eta_t) = Sigma ⊗ (q I)
Cov(vec(B_t) | observations) = Sigma ⊗ P_t
```

`Sigma` is estimated from Train ridge VAR residuals with a fixed eigenvalue floor; `q` is selected on a Validation grid. The Kalman recursion is performed exactly using the separable matrix-normal covariance, avoiding construction of a large Kronecker covariance. This is a full coefficient-state model, but not a model with an arbitrary structural process covariance. Initialization uses zero coefficients and `p0 I`; Train full-sample OLS coefficients are not used as a prior and then re-absorbed into Train. Estimating Sigma on Train and selecting parameters on Validation mean that Train features are retrospectively fitted features; the Train path must not be advertised as a real-time out-of-sample result. Validation/Test use forward filtering only, with no smoothing and no refitting of Sigma/q.

We do not claim a Primiceri stochastic-volatility TVP-SVAR, nor a line-by-line reproduction of the Antonakakis et al. forgetting-factor TVP. This implementation uses a fixed Train innovation covariance to respect the existing OOS frozen-parameter interface.

#### Smooth TV-VAR

Local linear ridge regression with a one-sided Epanechnikov kernel. Target day t uses only responses already observed within the preceding window; the window and ridge penalty are fixed in advance or selected on Validation. During Test, the local fit rolls forward under the same rule, constituting an explicitly declared real-time forecasting protocol. It is a causal forecasting adaptation of the smooth calendar-time baseline, not the two-sided estimation of Gao et al., the full TV-GIRF of Tan, or their structural identification/confidence-band implementation.

This smooth model does not enter the frozen-parameter metadata of `baselines.csv`; that CSV contains only HMM/TVP filtered features with fixed Train parameters, plus Static/Sparse diagnostic constants.

#### Fixed-network operator SSM

For each lag, the fixed basis contains identity and the Train correlation adjacency (zero diagonal, spectral-norm normalized). Only the low-dimensional weights follow a Gaussian random walk. A pooled Train ridge fit supplies the intercept and observation covariance; its coefficients do not seed the state prior. Forward Kalman filtering starts from zero mean and `p0 I`. Lag and process variance are selected on Validation MSE, using the same declared grid as TVP. Predictive covariance includes state uncertainty. The exported network operator path uses close-t filtered states; forecast means use the prior state instead.

This implements the fixed-network model class in the paper §23.1 B6. It is a correlation-adjacency robustness choice, not a claim of static causal discovery or an exact reproduction of Papamichalis et al. Synthetic S4 comparisons do not receive its oracle DGP basis. No spectral clipping is applied on Test.

#### DY connectedness

First compute `Psi_0=I` recursively from VAR(p), then compute the generalized FEVD for h=0,...,H-1:

```
theta_ij = [sum_h (Psi_h Sigma)_ij^2 / Sigma_jj]
           / [sum_h (Psi_h Sigma Psi_h')_ii]
theta_tilde_ij = theta_ij / sum_j theta_ij
FROM_i = 100 * sum_(j != i) theta_tilde_ij
TO_j   = 100 * sum_(i != j) theta_tilde_ij
NET_i  = TO_i - FROM_i
TCI    = mean_i FROM_i
```

Matrix rows are receiving units and columns are shock sources. The default dynamic DY is computed from the TVP same-day filtered coefficients and the Train Sigma (DY measure plus TVP parameter path); the Static VAR FEVD and static TCI diagnostics are also stored. Absolute coefficients or correlation coefficients are never labeled DY, and Cholesky FEVD is not used. This is predictive connectedness, not an identified structural causal effect. H=1 includes spillovers from contemporaneously correlated innovations and is therefore not guaranteed to be zero.

#### TVP risk field

`baseline__tvp_risk_hH` is `trace(sum_(h=0)^(H-1) Psi_h Sigma Psi_h') / N`, with coefficients and covariances first converted back to raw return units. It is the average H-step innovation forecast-error variance of the frozen current coefficients, in squared-return units; it is not a downside semivariance and excludes uncertainty from future coefficient changes.

#### Timing, stability and common sample

- Row t of `summary` and `losses` predicts x_t using at most information available at t-1.
- Row t of `baselines.csv` is a post-close feature that may absorb x_t, used to forecast risk from t+1 onward.
- The HMM uses the existing forward filter; return forecasts use prior regime probabilities.
- DLC uses the previous day's state posterior and fixes future covariates at the previous day's values; the density includes the state predictive covariance.
- All models share the same Train standardization and the same Validation/Test dates; failed rows are not silently dropped.
- Full companion spectral radii are stored; Test coefficients are not clipped to manufacture stability. A finite-H FEVD does not imply infinite-horizon stability.
- The bootstrap conditions on fitted/selected models, does not re-estimate parameters, and applies no multiple-comparison correction.

### Scope and limitations

| Research object | Status |
|---|---|
| Static VAR / sparse VAR | implemented with validation selection |
| unrestricted coefficient-state / TVP | implemented with filtering and risk summaries |
| smooth calendar-time VAR | implemented as a one-sided predictive baseline, not a full literature reproduction |
| DY directional connectedness | implemented with generalized FEVD and directional summaries |
| fixed-network state-space | implemented with Train correlation basis and Validation process-variance selection |
| Regime-PCMCI / joint MS-VAR | not implemented; existing two-stage HMM is not either method |
| BK frequency connectedness / complete TV-GIRF | not implemented |
| M0--M5 downside regression | provided by `oos_experiment` after feature export |

Treat each listed model class according to this scope table; do not describe all paper baselines as implemented.

### Method references

- Author-maintained software documentation for the Diebold–Yilmaz measures: https://search.r-project.org/CRAN/refmans/ConnectednessApproach/html/TimeConnectedness.html
- Antonakakis, Chatziantoniou and Gabauer (2020), the authors' TVP connectedness example: https://gabauerdavid.github.io/ConnectednessApproach/2020AntonakakisChatziantoniouGabauer
- Yan, Gao and Peng, the smooth TV-VAR methodology paper: https://arxiv.org/abs/2111.00450

The references above are used to verify methodological positioning; implementation differences are declared item by item in the Definitions subsection.

## HMM baseline comparison

This section describes the optional HMM comparison module. It contains no claim that a real-data comparison has already been run.

### Files and installation

The implementation is already part of the repository:

| Path | Role |
|---|---|
| `src/baselines/hmm_baseline.py` | filtered HMM baseline |
| `src/baselines/hmm_comparison.py` | HMM/VAR/DLC comparison entry point |
| `tests/test_hmm_comparison.py` | focused regression tests |

Install the optional dependency when running the HMM branch:

```bash
python -m pip install -e ".[hmm,dev]"
```

### Run

```bash
python -m pytest tests/test_hmm_comparison.py -q
python -m baselines.hmm_comparison
```

The baseline implementation is covered by the repository-wide test suite command. A real-data execution requires an authorized prepared dataset.

```bash
python -m baselines.hmm_comparison --states 2 3 4 --lags 1 3 --ridges 0.01 1.0 --n-init 5 --n-iter 200 --bootstrap-draws 1000 --block-length 20 --output outputs/hmm_comparison
```

Use a new output directory for each declared experiment; do not overwrite a run that may be cited.

### Comparison protocol

- The Train/Validation/Test split is chronological and unchanged by the comparison; date ordering and boundaries are checked.
- All models use Train-fitted standardization and only Train-varying covariates.
- HMM state count, lag and ridge are selected by Validation prediction MSE; parameters are fit on Train and are not refit on Validation or Test.
- HMM predictions use prior filtered probabilities (`filtered[t-1] @ transmat`), not probabilities that have already absorbed the target day's observation.
- DLC-SSM predictions use the state filtered through t-1 and persistence for the unknown future covariate; the target-day innovation is not used.
- Models are evaluated on common Validation/Test dates. The primary metric is mean standardized squared error with raw-return RMSE also reported.
- Loss differences are DLC-SSM minus comparator, so negative means lower DLC-SSM MSE. Moving-block bootstrap intervals are conditional on the fitted models.

### Outputs

All outputs are written to `outputs/hmm_comparison/` by default:

| File | Contents |
|---|---|
| `summary.csv` | MSE, raw-unit RMSE and common sample size of the three models on Validation / Test |
| `candidates.csv` | Complete validation candidate records for HMM/static VAR |
| `comparisons.csv` | Test-set loss differences of DLC-SSM relative to both comparators, with block bootstrap CIs |
| `losses.csv` | Date, segment and daily loss of the three models; Train is the fitting sample and does not represent OOS |
| `features.csv` | Filtered K-1 HMM probabilities plus information date / fitting cutoff date |
| `report.json` | Selected parameters, all initialization diagnostics, protocol, limitations and dependency versions |
| `hmm_predictions.npz` | HMM parameters, VAR coefficients, standardization parameters, all K probabilities and raw-unit forecasts |
| `dlcssm_train.npz` | The DLC-SSM model fitted on Train only in this run, with metadata |

`features.csv` contains only HMM columns required by the OOS loader; it is not a complete baseline CSV. DY/TVP and optional BK columns must come from genuine producers and must not be zero-filled. State labels are fit-specific. Train probabilities use the Train fit; Validation/Test probabilities use frozen parameters and forward filtering.

### Scope and limitations

This is a two-stage HMM plus soft-regime VAR comparison. It is not a jointly estimated Markov-switching VAR and is not Regime-PCMCI. State labels are fit specific and must not be interpreted as universal stress labels. The comparison does not establish that DLC-SSM wins; report results only from a completed, audited run directory.

Method reference: [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html) for initialization, training scores, state labels and convergence diagnostics.

### Verification status

- The focused tests validate filtering, prediction-time information isolation, and selection invariance. The full test suite can be run with `pytest -q`.
- Real-data sample sizes, metrics, confidence intervals and dependency versions must be read from the generated `report.json`; none are asserted here.

## Strict OOS specification

The executable contract is `oos_final`. Run it with `--dataset`, `--model`, `--model-fit-end`, `--baselines`, `--config config/oos_final.json`, and `--output`. The baseline CSV must contain `date`, `information_date`, `parameter_fit_end`, `baseline__hmm_probability_0..K-2`, `baseline__dy_connectedness_hH`, and `baseline__tvp_risk_hH` (plus `baseline__bk_connectedness_hH` when enabled). Values are date keyed, filtered probabilities are K−1 columns, and parameter fitting must end on or before the Train boundary.

M0–M5 are nested and immutable: controls; HMM probabilities; DY/TVP summaries; frozen companion geometry; state migration gap; state susceptibility. The full companion gain is `max_h ||C_t^h||₂`; the observable gain remains a diagnostic. Future labels are horizon-purged when their endpoint crosses a split boundary. The model is filtered once chronologically, and feature screening/scaling is fit on Train only and applied unchanged to Validation/Test. Frozen geometry remains evaluated at the filtered mean. M4 uses the conditional posterior predictive mean migration gap as specified below.

The baseline metadata is a producer declaration and is checked for temporal consistency, but cannot establish causal provenance by itself. Preserve the generated `specification.json` with every result.

### Pathwise predictive migration (v2)

At each original dataset row t, sample z_t from N(filtered_mean_t, filtered_cov_t). For each draw, simulate H transitions with fitted F, G and Q; use the current standardized u_t at every future step (persistence scenario). No future observed covariate, filtered state or smoothed state is used. The frozen comparator on a draw is C(z_t_draw)^h, paired with that same draw's future path. Products are C_{t+h} ... C_{t+1}; maximize over h=1,...,H before averaging draws. This incorporates both initial-state and transition uncertainty.

`geom__state_migration_gap_hH` (M4/M5) is E[max_h ||Phi_h-C(z_t)^h||_2 | F_t]. `geom__predicted_path_gain_hH` is E[max_h ||Phi_h||_2 | F_t]. The latter is a candidate diagnostic, not an extra M4/M5 regressor. The ten columns `predictive__{migration_gap,path_gain}_{mean,std,lower,upper,exceedance_probability}_hH` provide the full summaries; only the migration mean alias enters M4. Existing mean-state-path metrics remain as `diag__mean_path_migration_gap_hH` and `diag__mean_path_gain_hH`. Do not pool v1 and v2 feature exports.

Configuration declares `predictive_draws` (256), `predictive_seed` (0), `predictive_credible` (0.95), `path_gain_threshold` (1.0), and `migration_gap_threshold` (1.0). Intervals are equal-tailed quantiles; exceedance is strictly greater than the threshold. The thresholds are explicit fixed reference values, not estimated risk cutoffs or evidence of economic significance. Predeclare any changes before examining Test. The RNG stream is SeedSequence([seed, original_dataset_row_index]); append-only data updates keep previous draws unchanged. Larger draws improve Monte Carlo precision at added computational cost. There is no silent zero-draw fallback to the old mean path.

These summaries condition on fitted parameters and the persistence scenario; they do not include parameter-estimation uncertainty or a stochastic covariate forecast. The existing point frozen features and susceptibility are unchanged. `specification.json` records the new version, configuration and metric semantics. This command remains a feature/design exporter: it does not fit an OOS risk regression or claim empirical incremental performance.

Run the focused checks from the project root:

```bash
python -m unittest discover -s tests -p test_pathwise_oos.py -v
```

The temporal restrictions also follow the general leakage principle described in [scikit-learn's official guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). The specific pathwise definitions follow the published paper §14.7.

## OOS downside predictive pipeline

`oos_final` remains the strict feature/design exporter; `oos_experiment` adds Train/Validation/Test regression scoring after that export. The published-paper and later post-acceptance evidence should be identified by their own run configurations and reports, not by this command guide alone.

### Installation and execution

Install the project and the optional HMM dependency from the project root:

```bash
python -m pip install -e ".[hmm]"
```

The model-fit endpoint must be the actual endpoint recorded for the checkpoint and must not be later than the Train boundary. A date argument is not proof of checkpoint provenance; the run must record the model and dataset paths.

```bash
python -m oos.oos_experiment --dataset data/processed/dataset.npz --model outputs/checkpoints/dlcssm_train.npz --model-fit-end 2019-12-31 --output outputs/oos_downside_run1
```

The flow is: generate HMM/DY/TVP baselines, export posterior-predictive M0--M5 purged designs, fit Ridge on Train, select alpha on Validation, freeze the selected models, and score Test.

The defaults are `config/baselines.json`, `config/oos_final.json`, and `config/oos_predictive.json`. Automatic baseline production requires matching `horizon` and `n_regimes` in the relevant configurations and `include_bk=false`. The current producer does not supply BK; enabling it requires an external baseline CSV with the declared BK column.

Real-data execution can be slow because of baseline selection and posterior predictive draws. Predeclare any changes to draws, horizon or alpha grid and use a new output directory. A directory left by a failed stage is not a result.

### Reuse compatible inputs

With an existing baseline CSV, skip baseline generation:

```bash
python -m oos.oos_experiment --dataset data/processed/dataset.npz --model outputs/checkpoints/dlcssm_train.npz --model-fit-end 2019-12-31 --baselines outputs/baseline_comparison/baselines.csv --output outputs/oos_downside_run1
```

With an existing compatible `oos_final` export, score only the predictive stage:

```bash
python -m oos.oos_experiment --design-dir outputs/checkpoints/oos_final_pathwise_v2 --output outputs/oos_downside_scoring_run1
```

The directory must contain `designs.npz` and `specification.json` with the declared current specification version. Do not substitute an older export or mix caches from different data, models and configurations.

### Target and leakage controls

The target is:

`Y_t = mean(min(raw_market_return[t+1:t+H+1], 0)^2)`.

- M0--M5 use common valid rows; horizon purge removes targets crossing a split.
- Feature screening, means and scales are fit on Train only; states are forward filtered.
- Each Ridge candidate is fit on Train and alpha is selected on Validation only.
- Models are not silently refit on Train+Validation and Test labels are never used.
- Nonnegative prediction clipping is fixed before evaluation.
- Metrics and per-row predictions are exported for independent recomputation.
- Comparisons use the predeclared loss, HAC and moving-block bootstrap contracts.
- Missing Test dates and zero-variance comparisons are reported rather than hidden.

### Outputs

| Output | Contents |
|---|---|
| `outputs/oos_downside_run1/metrics.csv` | M0–M5 Validation/Test metrics; Validation scores select tuning settings |
| `outputs/oos_downside_run1/comparisons.csv` | Eight Test comparisons, bootstrap intervals, raw DM and Holm-adjusted p-values |
| `outputs/oos_downside_run1/predictions.csv` | Per-origin actual values, predictions and squared losses |
| `outputs/oos_downside_run1/candidates.csv` | Validation MSE for each alpha candidate |
| `outputs/oos_downside_run1/report.json` | Configuration, sample counts, selected alpha, coefficients, effective features and limitations |
| `outputs/oos_downside_run1/features/` | Feature export, unless an existing design directory was supplied |
| `outputs/oos_downside_run1/baselines/` | Automatic baseline outputs, unless a baseline CSV was supplied |

Coefficients in `report.json` correspond to Train-standardized features in the declared order. The default prediction applies `max(linear_prediction, 0)`.

### Verification and scope

```bash
python -m unittest discover -s tests -p test_oos_experiment.py -v
python -m unittest discover -s tests -p test_pathwise_oos.py -v
```

The focused tests cover the synthetic integration path, Test-label isolation, independent metric recomputation, seed reproducibility, financial loss scaling, identical predictions, split crossing and missing rows. Synthetic fixtures are not financial evidence.

The optional `hmmlearn` dependency is required for the automatic HMM branch. Real-data incremental results require a completed run report; the tests alone do not establish them.

This implements the declared downside-semivariance primary predictive target. ES, maximum drawdown, quantile loss, tail-event Brier/AUC, log predictive score and multi-horizon robustness remain separate experiments.

For nested Ridge models selected on Validation, the DM statistic is an asymptotic descriptive comparison, not a calibrated nested-model significance test. Bootstrap intervals condition on the fitted and selected models; they do not repeat estimation or selection and are per-comparison, not simultaneous, intervals. If upstream baseline specifications were chosen on Validation, those Validation scores are not unbiased generalization estimates. Report held-out Test evidence separately.

Geometry columns may be constant and may be removed by the declared Train screening; nested predictions can therefore coincide. Do not translate a successful run into a claim of structural-risk significance.

### Method references

- scikit-learn documents chronological ordering and gap semantics in [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- statsmodels documents Newey–West/Bartlett HAC settings and their time-series assumptions in [cov_hac](https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).

The implementation uses the repository's split and H-step purge. Exact M0--M5 definitions and the generated `specification.json` are authoritative.

## Consolidated technical reference

The following sections consolidate the algorithm, API, extension, and limitation contracts for release `0.1.0`.

## Algorithms and numerical contracts

This page summarizes the principal algorithms implemented by release `0.1.0`. It complements the equations in [`METHODS.md`](METHODS.md) and the bidirectional traceability table in [`PAPER_TO_CODE.md`](PAPER_TO_CODE.md).

### Directional support screening

`src/causal/causal_support.py` estimates an admissible lagged support using only the training split. The primary branch applies multivariate lagged Lasso with moving-block stability selection. Optional LPCMCI and PCMCI+ paths are sensitivity analyses. None of these masks is interpreted as an independently identified structural causal graph.

Inputs are finite multivariate time series, a lag range, split boundaries, resampling settings, and method-specific thresholds. Outputs include the boolean lagged support and diagnostics. Invalid dimensions, unsupported methods, and insufficient samples fail explicitly. Bootstrap randomness is controlled by configured seeds; changing worker count must not be assumed to preserve bitwise ordering across numerical backends.

### Dynamic latent conditional state-space model

`src/model/dlcssm.py` represents lag matrices as

```text
A_l(z_t) = A0_l + sum_r z_(t,r) B_(r,l).
```

The latent state follows a linear Gaussian transition and enters the time-varying autoregressive observation operator. Training uses a penalized generalized-EM procedure: forward filtering, RTS smoothing on the training window, Gaussian sufficient-statistic updates, and a sparse-group proximal update constrained to the admissible support.

Termination is controlled by configured iteration limits and likelihood/change tolerances. A returned fit is a floating-point estimate, not a certificate of a global optimum. Stabilization, covariance symmetrization, ridge terms, and positive-semidefinite solves are implementation safeguards. They can affect the result near singular or weakly identified settings.

### Kalman filter and RTS smoother

`src/model/kalman_em.py` implements a time-varying-observation Kalman filter and Rauch--Tung--Striebel smoother. The filter returns one-step predictions, filtered moments, innovations, and the observed-data Gaussian log-likelihood. The smoother is restricted to training estimation. Test-facing state features use forward-filtered moments and never backward-smoothed future information.

Covariance matrices are explicitly symmetrized. Linear systems use Cholesky factorization with increasing diagonal jitter rather than explicit inversion. Failure after all jitter attempts raises a linear-algebra error.

### Propagation and geometry

`src/model/propagation.py` builds block companion matrices and finite-horizon products. `src/model/geometry.py` evaluates spectral radius, non-normality, transient gain, subspace angles, frozen/pathwise differences, and analytic state sensitivities. Matrix powers use the full companion state unless a function explicitly applies the contemporaneous selection matrix.

Finite-horizon amplification is descriptive. Spectral radius below one is an asymptotic stability condition for a fixed operator; it does not preclude transient growth for a non-normal operator or establish stability along a time-varying path.

### Out-of-sample protocol

`src/oos/` constructs M0--M5 features and the future downside-semivariance target. Target windows crossing split boundaries are removed. Feature screening and standardization are fitted on Train only; tuning uses Validation only; Test is used for final scoring. Posterior-predictive path features condition on fitted parameters and a persisted-covariate scenario. They include filtered state and transition uncertainty but not parameter-estimation uncertainty.

### Baselines

`src/baselines/` contains static/sparse VAR, TVP-style, smooth VAR, HMM, fixed-network, and generalized forecast-error variance decomposition utilities. HMM test features are one-sided filtered probabilities. Generalized connectedness is predictive and ordering-invariant under its stated covariance model; it is not a structural shock decomposition.

### Failure and status semantics

The staged runner records successful, failed, skipped, and unsupported cells. It does not silently replace an unavailable estimator with another method. Publication mode requires every configured required stage to succeed. Smoke or `--allow-incomplete` execution is engineering evidence only.

## Public API and command reference

This page identifies the supported entry points in release `0.1.0`. Internal symbols beginning with an underscore are implementation details and may change without compatibility guarantees.

### Command-line interface

After an editable installation, run `dlcssm --help`. The principal commands are:

| Command | Purpose | Primary side effect |
|---|---|---|
| `dlcssm download --config config/data.yaml` | Acquire authorized source data | Writes under `data/raw/` and `data/interim/` |
| `dlcssm prepare --config config/data.yaml` | Build split-aware model arrays | Writes `data/processed/` artifacts |
| `dlcssm causal ...` | Estimate Train-only admissible support | Writes a support checkpoint and diagnostics |
| `dlcssm fit ...` | Fit the DLC-SSM | Writes a model checkpoint and fit diagnostics |
| `dlcssm synthetic ...` | Generate a credential-free synthetic example | Writes the requested synthetic artifact |
| `dlcssm theory-checks` | Run numerical mathematical checks | Prints check results; no empirical claim |

Use each command's `--help` output for required arguments. Configuration files under `config/` are the source of truth for scientific parameters.

### Publication runner

The staged orchestration interface is:

```bash
python -B scripts/run_main.py --config CONFIG --output OUTPUT
```

`--resume` reuses only verified successful cells. `--retry-failed` retries failed or unsupported cells in a finalized incomplete run. `--publication-mode` requires all declared required stages to succeed. `--allow-incomplete` is for smoke/debug execution and must not be used as evidence of a complete paper run.

### Python modules

| Module | Public responsibility |
|---|---|
| `core.data` | Data acquisition, split-aware preparation, serialization |
| `core.evaluation` | Targets, losses, OOS estimators, bootstrap comparisons |
| `core.experiment_contract` | Stage records, validation, and run ledgers |
| `causal.causal_support` | Train-only directional support screening |
| `model.kalman_em` | Filtering, smoothing, and Gaussian transition updates |
| `model.dlcssm` | Parameter objects, fitting, filtering, and persistence |
| `model.propagation` | Companion construction and finite-horizon propagation |
| `model.geometry` | Stability, non-normality, gain, and subspace diagnostics |
| `model.uncertainty` | Conditional posterior-predictive summaries |
| `baselines.*` | Comparator estimators and connectedness summaries |
| `oos.*` | Leakage-controlled feature export and predictive evaluation |
| `experiments.*` | Synthetic, geometry, baseline, validation, and ablation runs |

Array shape, validation, and return contracts are documented in source docstrings. The repository is a research prototype rather than a stable third-party library API; consumers should pin a release or commit.

### Serialization compatibility

Checkpoints and result directories are coupled to their configuration, dataset, and specification version. Do not mix a model, feature export, baseline CSV, or cached run from different configurations merely because array shapes match. Preserve generated metadata and manifests with every local run.

## Limitations and known constraints

This page records limitations of release `0.1.0`. A successful command or test does not remove these restrictions.

### Scientific interpretation

- Directional support masks are predictive admissibility screens, not identified structural causal graphs.
- The fitted latent representation is gauge-dependent. Observable operators and gauge-invariant subspaces are the appropriate comparison objects.
- Generalized-EM fitting is nonconvex and can depend on initialization, regularization, scaling, and numerical tolerances. Convergence to a global optimum is not claimed.
- Stability of one frozen companion matrix does not establish stability of a time-varying operator path, and it does not rule out transient amplification.
- Posterior-predictive geometry conditions on fitted parameters. Parameter uncertainty and a stochastic future-covariate model are not included.
- Synthetic fixtures verify recovery under controlled generators; they are not evidence of financial-market performance.

### Out-of-sample evidence

- The M0--M5 protocol uses one chronological Train/Validation/Test partition. Generalization to other periods, markets, frequencies, or asset definitions has not been established by the repository alone.
- Validation scores are used for model selection and are not unbiased final performance estimates.
- Diebold--Mariano statistics for nested selected models are descriptive asymptotic comparisons. Bootstrap intervals condition on fitted and selected models and are not simultaneous intervals.
- Geometry features can be constant on Train and may be removed by the declared screening rule; nested models can therefore produce identical predictions.

### Data and access

- Tushare credentials and permissions are required for real-data acquisition.
- Provider revisions can change a newly downloaded dataset. The public code cannot certify equality to an unavailable historical provider snapshot.
- Third-party financial data are governed by provider terms, not by the repository's Apache-2.0 code license.

### Numerical and platform constraints

- The implementation uses floating-point linear algebra. Near-singular covariance matrices may require jitter or regularization and can remain sensitive to scale.
- Bitwise equality across operating systems, BLAS/LAPACK libraries, process counts, and dependency versions is not guaranteed.
- Full-run wall time and peak memory are configuration- and hardware-dependent; no universal resource bound is asserted.
- GPU acceleration and distributed execution are not supported interfaces.

### Release maturity

The project is classified as an alpha research prototype in `pyproject.toml`. Public interfaces and serialized artifacts may evolve. Pin the exact release or commit used for scientific work and record deviations from the supplied configuration.

### Reporting a result-affecting issue

An issue should state the affected revision and configuration, observable behavior, expected behavior, minimal reproduction, environment, and potential scientific impact. Do not include credentials or restricted data.

## 🧩 Extending the experiment code

This guide applies to the public ICMLC 2027 experiment repository. Keep a new scientific design distinguishable from the published configuration and preserve the Train/Validation/Test information boundary.

### Data and covariates

`src/core/data.py` builds date-aligned sector returns, observed covariates, split labels and Train-fitted scalers. A new source or variable should declare its source, units, availability date, missingness rule and transformation. It must not use information after the prediction origin. Preserve sector/date alignment and add a boundary test under `tests/`. The licensed Tushare data are obtained by users separately; see [Data and provenance](../data/README.md).

### Model and baseline

New operator behavior belongs in `src/model/`, and new comparators in `src/baselines/` with an experiment adapter in `src/experiments/`. State whether the model uses Train-only fit, Validation selection, online filtering, or future covariate assumptions. Do not replace a failed model cell with another seed or comparator. Add focused numerical and information-isolation tests.

### OOS factor or target

The declared nested feature sets live in `src/oos/oos_specs.py`, feature construction in `src/oos/oos_final.py`, and the five-day target in `src/core/evaluation.py`. A new feature must be available at the origin, preserve the common valid-row comparison, and receive a new specification version. Update `config/oos_final.json` or a new configuration, record normalization and uncertainty semantics, and test split-boundary purging. Validation selects regression settings; Test labels must not influence them.

### Verification and evidence

Run `python -m pytest -q` and relevant focused tests after a code change. Use an independent output root for a changed target, feature roster, split, model, seed design or inference rule. Check manifests, attempted and successful cells, source identity and per-row predictions before citing an aggregate. Update the [paper-to-code map](PAPER_TO_CODE.md), [reproducibility page](REPRODUCIBILITY.md) and README when a scientific contract changes. The post-acceptance commands are recorded in the supplement section of [the reproducibility guide](REPRODUCIBILITY.md#post-acceptance-supplement-experiments).
