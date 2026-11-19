# 🗃️ Data and provenance

This page is the data reference for the experiment code for *Testing a Shared Low-Dimensional Operator Representation for Time-Varying Financial Dynamics*, prepared for an ICMLC 2027 submission. The repository does not bundle or certify a Tushare data snapshot. The authoritative pipeline settings are in [`config/data.yaml`](../config/data.yaml).

## Source and coverage

The default dataset comprises daily Chinese A-share SW 2021 Level-1 sector indices. The configured source window is 2012-01-01 through 2026-08-31; the modeling splits begin in 2014 after preparation and warm-up. Tushare Pro `index_classify` identifies sectors, `sw_daily` supplies sector market data, and `index_daily` supplies the CSI 300 (`000300.SH`) broad-market series. Optional USD/CNH input uses `fx_daily`; it is disabled in the default config.

## Variables and target

The prepared dataset contains sector returns `x_t` and observed covariates `u_t`. The default covariates are CSI 300 simple return, trailing 20-day annualized realized volatility, cross-sector turnover proxy, and cross-sectional sector-return dispersion. The turnover proxy is the daily median of `amount / float_mv`; the builder uses the median of `log1p(amount)` when floating market value is unavailable.

The M0 OOS regression additionally calculates the mean pairwise sector-return correlation over a trailing 20-observation window. Its formula and the future five-day downside-semivariance target are documented in the [README factor definitions](../README.md).

## Chronological split and information controls

- Train: 2014-03-01 to 2019-12-31.
- Validation: 2020-01-01 to 2022-12-31.
- Test: 2023-01-01 to 2026-08-31.

Return and covariate standardizers are fitted on Train only, and directional support is screened on Train only. Smoothing is confined to training estimation; Test states use forward filtering. Future pathwise state features use draws from the fitted transition model, not observed future states. OOS target windows crossing a split boundary are excluded.

## Local directories

- `raw/`: immutable source pulls from Tushare. Never commit.
- `interim/`: normalized long tables and audit artifacts. Never commit.
- `processed/`: model-ready `dataset.npz`, `dataset_metadata.json`, split metadata, and scalers. Never commit.

Set `TUSHARE_TOKEN` in the local environment, then run the following from the repository root:

```bash
dlcssm download --config config/data.yaml
dlcssm prepare --config config/data.yaml
```

Each user needs their own account permissions. Raw and processed financial data are excluded from the public release, as are local experiment outputs and the manuscript working directory. The code's Apache-2.0 license does not grant rights to third-party data; see [third-party notices](../THIRD_PARTY_NOTICE.md).
