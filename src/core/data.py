"""Data acquisition, cleaning, scaling, and train-only preparation utilities.

The preparation boundary is deliberate: statistics used for imputation,
normalisation, feature screening, and split construction must be estimated
from the training portion only to avoid look-ahead leakage in the experiments.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_yaml, set_global_seed


@dataclass(frozen=True)
class PreparedDataset:
    """Immutable array bundle passed from data preparation into experiments."""
    x: np.ndarray
    u: np.ndarray
    dates: np.ndarray
    sector_codes: np.ndarray
    sector_names: np.ndarray
    covariate_names: np.ndarray
    split: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    u_mean: np.ndarray
    u_scale: np.ndarray


class TushareProvider:
    """Small retrying Tushare Pro adapter.

    No token is ever persisted by this class.

    Retries are limited to transient API failures; credentials remain in the
    process environment and are never copied into tracked configuration.
    """

    def __init__(
        self,
        token_env: str = "TUSHARE_TOKEN",
        api_url_env: str = "TUSHARE_API_URL",
        retries: int = 5,
        base_backoff_seconds: float = 1.0,
        request_pause_seconds: float = 0.15,
    ) -> None:
        """
          init  .
        :param token_env: Input argument used by this operation.
        :param api_url_env: Input argument used by this operation.
        :param retries: Input argument used by this operation.
        :param base_backoff_seconds: Input argument used by this operation.
        :param request_pause_seconds: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        token = os.getenv(token_env)
        api_url = os.getenv(api_url_env)
        if not token:
            raise RuntimeError(
                f"Missing {token_env}. Export your Tushare token in the environment; "
                "do not store it in a tracked config file."
            )
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the package with `pip install tushare`.") from exc
        self.pro = ts.pro_api(token)

        if api_url:
            self.pro._DataApi__token = token
            self.pro._DataApi__http_url = api_url

        self.retries = retries
        self.base_backoff_seconds = base_backoff_seconds
        self.request_pause_seconds = request_pause_seconds

    def call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        """Call one provider endpoint with bounded exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                fn = getattr(self.pro, api_name)
                frame = fn(**kwargs)
                if not isinstance(frame, pd.DataFrame):
                    raise RuntimeError(f"Tushare {api_name} did not return a DataFrame")
                time.sleep(self.request_pause_seconds)
                return frame
            except Exception as exc:  # provider errors are dynamic
                last_error = exc
                if attempt + 1 == self.retries:
                    break
                time.sleep(self.base_backoff_seconds * (2**attempt))
        raise RuntimeError(f"Tushare call failed: {api_name} {kwargs}") from last_error

    def fetch_sw_level1(self, taxonomy: str = "SW2021") -> pd.DataFrame:
        """Fetch SW level-1 classification with compatibility fallbacks."""
        attempts = [
            {"level": "L1", "src": taxonomy},
            {"level": "L1"},
            {"src": taxonomy},
            {},
        ]
        last: Exception | None = None
        for kwargs in attempts:
            try:
                df = self.call("index_classify", **kwargs)
                if not df.empty:
                    return _normalize_sw_classification(df, taxonomy)
            except Exception as exc:
                last = exc
        raise RuntimeError("Unable to resolve SW industry classification") from last

    def fetch_sw_daily(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch sw daily.
        :param ts_code: Input argument used by this operation.
        :param start: Input argument used by this operation.
        :param end: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        return self.call(
            "sw_daily",
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields=(
                "ts_code,trade_date,name,open,high,low,close,change,pct_change,"
                "vol,amount,pe,pb,float_mv,total_mv"
            ),
        )

    def fetch_index_daily(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch index daily.
        :param ts_code: Input argument used by this operation.
        :param start: Input argument used by this operation.
        :param end: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        return self.call(
            "index_daily",
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
        )

    def fetch_fx_daily(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        # fx_daily has a smaller row limit; split by ~2 years.
        """
        Fetch fx daily.
        :param ts_code: Input argument used by this operation.
        :param start: Input argument used by this operation.
        :param end: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        cursor = start_ts
        while cursor <= end_ts:
            right = min(cursor + pd.Timedelta(days=700), end_ts)
            frames.append(
                self.call(
                    "fx_daily",
                    ts_code=ts_code,
                    start_date=cursor.strftime("%Y%m%d"),
                    end_date=right.strftime("%Y%m%d"),
                )
            )
            cursor = right + pd.Timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates()


def _normalize_sw_classification(df: pd.DataFrame, taxonomy: str) -> pd.DataFrame:
    """
     normalize sw classification.
    :param df: Input argument used by this operation.
    :param taxonomy: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}

    code_col = next(
        (lower[k] for k in ("ts_code", "index_code", "industry_code") if k in lower),
        None,
    )
    if code_col is None:
        raise ValueError(f"Cannot identify SW index code column from {list(out.columns)}")

    name_col = next(
        (lower[k] for k in ("industry_name", "name", "l1_name") if k in lower),
        None,
    )
    if name_col is None:
        raise ValueError(f"Cannot identify SW industry name column from {list(out.columns)}")

    if "level" in lower:
        level = out[lower["level"]].astype(str).str.upper()
        out = out[level.isin({"L1", "1", "一级行业"})]

    if "src" in lower:
        src = out[lower["src"]].astype(str).str.upper()
        wanted = taxonomy.upper()
        exact = out[src == wanted]
        if not exact.empty:
            out = exact

    if "is_pub" in lower:
        pub = out[lower["is_pub"]].astype(str)
        published = out[pub.isin({"1", "Y", "TRUE", "True"})]
        if not published.empty:
            out = published

    codes = out[code_col].astype(str).str.strip()
    codes = codes.where(codes.str.contains(r"\."), codes + ".SI")
    result = pd.DataFrame(
        {
            "ts_code": codes,
            "name": out[name_col].astype(str).str.strip(),
        }
    ).drop_duplicates("ts_code")
    result = result[result["ts_code"].str.match(r"^8\d{5}\.SI$")]
    return result.sort_values("ts_code").reset_index(drop=True)


def download_from_tushare(config_path: str | Path) -> dict[str, Path]:
    """
    Download from tushare.
    :param config_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    cfg = load_yaml(config_path)
    set_global_seed(int(cfg["seed"]))
    storage = cfg["storage"]
    raw_dir = Path(storage["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    provider = TushareProvider(
        token_env=cfg["token_env"],
        retries=int(cfg["api"]["retries"]),
        base_backoff_seconds=float(cfg["api"]["base_backoff_seconds"]),
        request_pause_seconds=float(cfg["api"]["request_pause_seconds"]),
    )
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]

    classification = provider.fetch_sw_level1(cfg["industry"]["taxonomy"])
    n = len(classification)
    lo = int(cfg["industry"]["expected_count_min"])
    hi = int(cfg["industry"]["expected_count_max"])
    if not (lo <= n <= hi):
        raise RuntimeError(
            f"Expected approximately {lo}-{hi} SW L1 sectors, got {n}. "
            "Inspect the provider classification before continuing."
        )

    class_path = raw_dir / "sw_level1_classification.csv.gz"
    classification.to_csv(class_path, index=False, compression="gzip")

    sector_frames = []
    for row in classification.itertuples(index=False):
        frame = provider.fetch_sw_daily(row.ts_code, start, end)
        if frame.empty:
            raise RuntimeError(f"No sw_daily data returned for {row.ts_code} {row.name}")
        sector_frames.append(frame)
    sectors = pd.concat(sector_frames, ignore_index=True)
    sectors = sectors.drop_duplicates(["ts_code", "trade_date"])
    sectors_path = raw_dir / "sw_level1_daily.csv.gz"
    sectors.to_csv(sectors_path, index=False, compression="gzip")

    market = provider.fetch_index_daily(cfg["market"]["broad_index"], start, end)
    market_path = raw_dir / "broad_market_daily.csv.gz"
    market.to_csv(market_path, index=False, compression="gzip")

    out = {
        "classification": class_path,
        "sectors": sectors_path,
        "market": market_path,
    }

    if bool(cfg["market"].get("include_fx", False)):
        fx = provider.fetch_fx_daily(cfg["market"]["fx_code"], start, end)
        fx_path = raw_dir / "fx_daily.csv.gz"
        fx.to_csv(fx_path, index=False, compression="gzip")
        out["fx"] = fx_path

    return out


def _parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
     parse date col.
    :param df: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="raise")
    return out


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
     fit standardizer.
    :param values: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return mean, scale


def _split_labels(dates: pd.DatetimeIndex, splits: dict[str, Any]) -> np.ndarray:
    """
     split labels.
    :param dates: Input argument used by this operation.
    :param splits: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    labels = np.full(len(dates), "outside", dtype="U10")
    for label in ("train", "validation", "test"):
        start = pd.Timestamp(splits[label]["start"])
        end = pd.Timestamp(splits[label]["end"])
        mask = (dates >= start) & (dates <= end)
        if np.any(labels[mask] != "outside"):
            raise ValueError(f"Overlapping split definitions at {label}")
        labels[mask] = label
    return labels


def prepare_dataset(config_path: str | Path) -> PreparedDataset:
    """
    Prepare dataset.
    :param config_path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    cfg = load_yaml(config_path)
    set_global_seed(int(cfg["seed"]))
    raw = Path(cfg["storage"]["raw_dir"])
    processed = Path(cfg["storage"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    classification = pd.read_csv(raw / "sw_level1_classification.csv.gz")

    sectors = _parse_date_col(pd.read_csv(raw / "sw_level1_daily.csv.gz"))

    market = _parse_date_col(pd.read_csv(raw / "broad_market_daily.csv.gz"))

    sector_codes = classification["ts_code"].astype(str).tolist()

    sector_names_map = dict(
        zip(
            classification["ts_code"],
            classification["name"],
            strict=False,
        )
    )

    sectors = sectors[sectors["ts_code"].isin(sector_codes)].copy()

    # -------------------------------------------------
    # Sector returns: derive from close, not pct_change
    # -------------------------------------------------

    sectors["close"] = pd.to_numeric(
        sectors["close"],
        errors="coerce",
    )

    sectors = sectors.sort_values(["ts_code", "date"])

    sectors["return"] = sectors.groupby("ts_code", sort=False)["close"].pct_change(fill_method=None)

    ret = sectors.pivot(
        index="date",
        columns="ts_code",
        values="return",
    ).sort_index()

    # Fit sector coverage on Train only, before any imputation or date dropping.
    # Validation/Test missingness must never change the selected sector universe.
    analysis_start = pd.Timestamp(cfg["splits"]["train"]["start"])
    analysis_end = pd.Timestamp(cfg["splits"]["test"]["end"])
    ret = ret.loc[(ret.index >= analysis_start) & (ret.index <= analysis_end)]

    coverage_split = _split_labels(pd.DatetimeIndex(ret.index), cfg["splits"])
    train_returns = ret.loc[coverage_split == "train"]
    if train_returns.empty:
        raise RuntimeError("No Train observations available for sector coverage screening.")

    threshold = float(cfg["preprocess"]["min_sector_history_fraction"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_sector_history_fraction must be finite and in [0, 1].")
    history_fraction = train_returns.notna().mean(axis=0)
    # A sector without any Train return cannot be fitted, even at threshold zero.
    eligible = (history_fraction >= threshold) & train_returns.notna().any(axis=0)
    keep = history_fraction[eligible].index.tolist()
    if not keep:
        raise RuntimeError("No sectors pass Train-only sector coverage screening.")
    ret = ret[keep]

    if cfg["preprocess"]["missing_policy"] == "drop_dates":
        ret = ret.dropna(
            axis=0,
            how="any",
        )

    elif cfg["preprocess"]["missing_policy"] == "cross_sectional_median":
        ret = ret.apply(
            lambda row: row.fillna(row.median()),
            axis=1,
        )

    else:
        raise ValueError("Unsupported missing_policy")

    # -------------------------------------------------
    # Broad-market return
    # -------------------------------------------------

    market["close"] = pd.to_numeric(
        market["close"],
        errors="coerce",
    )

    market = market.sort_values("date")

    market["market_return"] = market["close"].pct_change(fill_method=None)

    market_series = market.set_index("date")["market_return"].sort_index()

    window = int(cfg["preprocess"]["realized_vol_window"])
    realized_vol = market_series.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)

    amount = sectors.pivot(index="date", columns="ts_code", values="amount").reindex(columns=keep)
    if "float_mv" in sectors:
        float_mv = sectors.pivot(index="date", columns="ts_code", values="float_mv").reindex(
            columns=keep
        )
        turnover = (amount / float_mv.replace(0, np.nan)).median(axis=1)
    else:
        turnover = np.log1p(amount).median(axis=1)
    turnover.name = "turnover_proxy"

    dispersion = ret.std(axis=1, ddof=1)
    dispersion.name = "cross_sectional_dispersion"

    cov = pd.concat(
        [
            market_series.rename("market_return"),
            realized_vol.rename("realized_volatility"),
            turnover,
            dispersion,
        ],
        axis=1,
    )

    if bool(cfg["market"].get("include_fx", False)):
        fx_path = raw / "fx_daily.csv.gz"
        fx = _parse_date_col(pd.read_csv(fx_path))
        mid = (
            pd.to_numeric(fx["bid_close"], errors="coerce")
            + pd.to_numeric(fx["ask_close"], errors="coerce")
        ) / 2.0
        fx_series = pd.Series(mid.to_numpy(), index=fx["date"]).sort_index()
        fx_return = np.log(fx_series).diff()
        shift = int(cfg["market"].get("fx_shift_calendar_days", 1))
        fx_return.index = fx_return.index + pd.Timedelta(days=shift)
        cov["rmb_fx_return"] = fx_return

    cov_names = list(cfg["preprocess"]["covariates"])
    if "rmb_fx_return" in cov.columns and "rmb_fx_return" not in cov_names:
        cov_names.append("rmb_fx_return")

    joined = ret.join(cov[cov_names], how="inner").dropna()
    ret = joined[keep]
    cov = joined[cov_names]
    dates = pd.DatetimeIndex(joined.index)
    split = _split_labels(dates, cfg["splits"])
    in_declared = split != "outside"
    ret = ret.loc[in_declared]
    cov = cov.loc[in_declared]
    dates = dates[in_declared]
    split = split[in_declared]

    if (
        not np.any(split == "train")
        or not np.any(split == "validation")
        or not np.any(split == "test")
    ):
        raise RuntimeError(
            "All Train/Validation/Test splits must contain observations after alignment."
        )

    x_raw = ret.to_numpy(dtype=float)
    u_raw = cov.to_numpy(dtype=float)
    train_mask = split == "train"

    x_mean, x_scale = _fit_standardizer(x_raw[train_mask])
    u_mean, u_scale = _fit_standardizer(u_raw[train_mask])

    x = x_raw.copy()
    u = u_raw.copy()
    if bool(cfg["preprocess"]["standardize_returns"]):
        x = (x - x_mean) / x_scale
    if bool(cfg["preprocess"]["standardize_covariates"]):
        u = (u - u_mean) / u_scale

    sector_names = np.asarray([sector_names_map.get(code, code) for code in keep], dtype="U64")
    dataset = PreparedDataset(
        x=x,
        u=u,
        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),
        sector_codes=np.asarray(keep, dtype="U16"),
        sector_names=sector_names,
        covariate_names=np.asarray(cov_names, dtype="U64"),
        split=split,
        x_mean=x_mean,
        x_scale=x_scale,
        u_mean=u_mean,
        u_scale=u_scale,
    )
    save_prepared_dataset(dataset, cfg["storage"]["dataset_file"])

    metadata = {
        "rows": int(len(dates)),
        "sectors": int(len(keep)),
        "covariates": cov_names,
        "sector_codes": keep,
        "sector_screening": {
            "fit_split": "train",
            "train_start": str(analysis_start.date()),
            "train_end": str(pd.Timestamp(cfg["splits"]["train"]["end"]).date()),
            "observations_before_missing_policy": int(len(train_returns)),
            "min_sector_history_fraction": threshold,
            "history_fraction": {str(k): float(v) for k, v in history_fraction.items()},
            "kept_sector_codes": keep,
            "dropped_sector_codes": history_fraction[~eligible].index.tolist(),
        },
        "sector_names": sector_names.tolist(),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "split_counts": {k: int(np.sum(split == k)) for k in ("train", "validation", "test")},
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "u_mean": u_mean.tolist(),
        "u_scale": u_scale.tolist(),
    }
    Path(cfg["storage"]["metadata_file"]).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dataset


def save_prepared_dataset(dataset: PreparedDataset, path: str | Path) -> None:
    """
    Save prepared dataset.
    :param dataset: Input argument used by this operation.
    :param path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=dataset.x,
        u=dataset.u,
        dates=dataset.dates,
        sector_codes=dataset.sector_codes,
        sector_names=dataset.sector_names,
        covariate_names=dataset.covariate_names,
        split=dataset.split,
        x_mean=dataset.x_mean,
        x_scale=dataset.x_scale,
        u_mean=dataset.u_mean,
        u_scale=dataset.u_scale,
    )


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    """
    Load prepared dataset.
    :param path: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    with np.load(path, allow_pickle=False) as z:
        return PreparedDataset(
            x=z["x"],
            u=z["u"],
            dates=z["dates"],
            sector_codes=z["sector_codes"],
            sector_names=z["sector_names"],
            covariate_names=z["covariate_names"],
            split=z["split"],
            x_mean=z["x_mean"],
            x_scale=z["x_scale"],
            u_mean=z["u_mean"],
            u_scale=z["u_scale"],
        )
