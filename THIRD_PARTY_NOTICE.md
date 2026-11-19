# ⚖️ Third-party notices and data terms

The [Apache-2.0 license](LICENSE) governs the original repository code under its terms. It does not grant rights to external datasets, package code, trademarks, or the published paper. This file is attribution and provenance guidance, not a replacement for the upstream notices distributed with packages.

## Financial data

The data pipeline uses Tushare Pro `index_classify`, `sw_daily` and `index_daily`; optional FX data use `fx_daily`. Access, use and redistribution are governed by the provider's applicable terms and account permissions. Raw and processed financial data are not included in the public repository. Users obtain their own authorization and retain source metadata for their run. See [Data and provenance](data/README.md).

## Software dependencies

Runtime packages declared in [pyproject.toml](pyproject.toml) include NumPy, SciPy, pandas, scikit-learn, PyYAML, Typer, tqdm, joblib, psutil, Matplotlib and Tushare. Optional branches use hmmlearn and Tigramite. Development and documentation extras are declared in the same file. Each dependency has its own license and attribution requirements; installation does not relicense it under this repository's Apache-2.0 license. Consult installed distribution metadata and upstream license files for the exact versions used in a run.

## Scholarly methods

The implementation uses established methods such as Kalman filtering and smoothing, generalized EM, VAR and TVP-VAR, HMM filtering, generalized FEVD, Diebold–Yilmaz connectedness, and Ridge regression. Academic attribution for these methods is separate from software and data licensing. The [paper-to-code map](docs/PAPER_TO_CODE.md) identifies their roles in this repository.
