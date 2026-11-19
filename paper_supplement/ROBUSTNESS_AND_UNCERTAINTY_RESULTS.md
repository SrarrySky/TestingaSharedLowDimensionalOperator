# Robustness and Uncertainty Results

This document reports the post-acceptance robustness and uncertainty analyses referenced in the paper. It is a self-contained numerical results record rather than an additional paper section. All synthetic intervals are descriptive finite-seed summaries. The financial moving-block intervals condition on fixed forecasts, selected penalties, model specification, and screened support.

## 1. Ten-seed S5 audit

The structurally matched S5 positive-control mechanism was evaluated using seeds 7, 19, 31, and 42--48. All seven methods were rerun under a common implementation, producing 70 successful method-by-seed fits. Values below are the mean plus or minus the sample standard deviation across the ten independently generated datasets. Lower values are better except for transmitter Spearman rho.

| Method | Test MSE | Coefficient RMSE | Kernel RMSE | Peak-gain RMSE | Transmitter rho |
|---|---:|---:|---:|---:|---:|
| Fixed-network SSM | 0.963247 +/- 0.032406 | 0.027628 +/- 0.003979 | 0.014131 +/- 0.002774 | 0.034061 +/- 0.019435 | -0.054762 +/- 0.386052 |
| TVP-VAR | 0.959779 +/- 0.034722 | 0.030971 +/- 0.003721 | 0.015253 +/- 0.002376 | 0.099491 +/- 0.036286 | 0.404762 +/- 0.307380 |
| Sparse VAR | 0.961417 +/- 0.032888 | 0.030610 +/- 0.002599 | 0.015482 +/- 0.002495 | 0.088217 +/- 0.026526 | 0.359524 +/- 0.373851 |
| Static VAR | 0.963351 +/- 0.033077 | 0.034913 +/- 0.003551 | 0.017159 +/- 0.002088 | 0.118624 +/- 0.030420 | 0.314286 +/- 0.349891 |
| DLC-SSM | 1.000361 +/- 0.030213 | 0.039192 +/- 0.005119 | 0.019305 +/- 0.003075 | 0.141180 +/- 0.039616 | 0.264286 +/- 0.470930 |
| HMM + soft regime VAR | 0.970773 +/- 0.032563 | 0.038744 +/- 0.004665 | 0.018571 +/- 0.002329 | 0.070148 +/- 0.024079 | 0.214286 +/- 0.364561 |
| Smooth TV-VAR | 1.088082 +/- 0.036207 | 0.090329 +/- 0.003703 | 0.045978 +/- 0.002011 | 0.417084 +/- 0.033282 | 0.154762 +/- 0.376169 |

Coefficient RMSE measures dynamic coefficient-path error. Kernel RMSE and peak-gain RMSE evaluate frozen propagation recovery at horizon 10.

### 1.1 Paired seed-level contrasts

Entries are mean DLC-SSM-minus-comparator differences with descriptive 95% Student-t intervals across the ten matched seeds. Positive differences are unfavorable for MSE and RMSE metrics; positive differences are favorable for transmitter rho.

| Comparator | Metric | Mean difference | Descriptive 95% interval |
|---|---|---:|---:|
| TVP-VAR | Test MSE | 0.040582 | [0.031139, 0.050026] |
| TVP-VAR | Coefficient RMSE | 0.008221 | [0.005713, 0.010729] |
| TVP-VAR | Kernel RMSE | 0.004051 | [0.002677, 0.005425] |
| TVP-VAR | Peak-gain RMSE | 0.041689 | [0.025632, 0.057746] |
| TVP-VAR | Transmitter rho | -0.140476 | [-0.370177, 0.089225] |
| Fixed-network SSM | Test MSE | 0.037114 | [0.030708, 0.043520] |
| Fixed-network SSM | Coefficient RMSE | 0.011564 | [0.007799, 0.015329] |
| Fixed-network SSM | Kernel RMSE | 0.005174 | [0.003724, 0.006623] |
| Fixed-network SSM | Peak-gain RMSE | 0.107119 | [0.074243, 0.139995] |
| Fixed-network SSM | Transmitter rho | 0.319048 | [-0.008497, 0.646592] |

The loss-based contrasts are positive throughout. By contrast, both transmitter-rank intervals include zero and therefore do not establish a stable ordering between DLC-SSM and the two comparators.

### 1.2 Test-MSE wins

| Method | Seeds with lowest Test MSE |
|---|---:|
| TVP-VAR | 4 |
| Sparse VAR | 4 |
| Fixed-network SSM | 2 |
| DLC-SSM | 0 |

The selected DLC-SSM fit satisfies the convergence criterion on eight seeds and reaches the iteration cap on two seeds, 31 and 46.

## 2. Oracle-support recovery sensitivity

Each condition contains three seeds. Values are the mean plus or minus the sample standard deviation. The generating lag and state dimension are fixed, and oracle support removes support-selection mismatch.

| T | Noise scale | Maximum principal angle (degrees) | Projector distance | Coefficient RMSE | Converged | Reached iteration cap |
|---:|---:|---:|---:|---:|---:|---:|
| 1200 | 0.25 | 82.020 +/- 4.274 | 0.9885 +/- 0.0098 | 0.0393 +/- 0.0021 | 0/3 | 3/3 |
| 1200 | 0.10 | 81.543 +/- 1.293 | 0.9890 +/- 0.0032 | 0.0474 +/- 0.0049 | 1/3 | 2/3 |
| 1200 | 0.05 | 83.794 +/- 4.514 | 0.9921 +/- 0.0098 | 0.0354 +/- 0.0049 | 1/3 | 2/3 |
| 2400 | 0.10 | 86.391 +/- 1.549 | 0.9978 +/- 0.0018 | 0.0288 +/- 0.0023 | 0/3 | 3/3 |

At noise scale 0.10, doubling T from 1200 to 2400 lowers coefficient-path RMSE from 0.0474 to 0.0288 but does not improve the principal-angle or projector-distance diagnostics. Across all four conditions, only 2 of 12 fits satisfy the convergence criterion and the remaining 10 reach the iteration cap. The analysis therefore does not distinguish weak identification from optimization failure under the evaluated computational budget.

`objective_history_length` includes the initial objective evaluation and should not be interpreted as the number of completed EM updates.

## 3. Moving-block interval sensitivity

Positive relative improvement means lower candidate MSE. Point estimates, predictions, design matrices, selected ridge penalties, model specification, and screened support are identical across block lengths. Only the moving-block interval calculation changes.

The nested predictor sequence is:

- M0: conventional market controls;
- M1: M0 plus filtered HMM probabilities;
- M2: M1 plus connectedness and flexible-TVP risk controls;
- M3: M2 plus frozen propagation geometry;
- M4: M3 plus predictive pathwise geometry;
- M5: M4 plus state-susceptibility features.

| Comparison | Relative-MSE point estimate | Block 10 interval | Block 20 interval | Block 40 interval |
|---|---:|---:|---:|---:|
| M0 to M1 | -0.17% | [-0.88%, 0.44%] | [-1.09%, 0.45%] | [-1.40%, 0.41%] |
| M1 to M2 | -0.81% | [-1.19%, -0.35%] | [-1.17%, -0.24%] | [-1.27%, -0.17%] |
| M2 to M3 | -7.04% | [-12.17%, -3.71%] | [-13.99%, -3.83%] | [-14.08%, -3.87%] |
| M3 to M4 | -2.70% | [-7.37%, 0.42%] | [-8.18%, 0.50%] | [-7.79%, 0.33%] |
| M4 to M5 | 0.29% | [-0.18%, 0.78%] | [-0.20%, 0.87%] | [-0.11%, 0.98%] |
| M2 to M4 | -9.93% | [-18.62%, -5.43%] | [-21.42%, -5.24%] | [-21.36%, -4.83%] |
| M2 to M5 | -9.60% | [-18.04%, -5.13%] | [-20.99%, -5.03%] | [-20.95%, -4.59%] |
| M0 to M5 | -10.68% | [-19.26%, -6.15%] | [-22.35%, -5.89%] | [-22.87%, -5.30%] |

The M2-to-M3 interval remains negative for all three block lengths, so the adverse frozen-geometry increment is stable across the evaluated dependence-horizon choices. The M3-to-M4 and M4-to-M5 intervals include zero for all three block lengths, leaving the later feature increments interval-uncertain.
