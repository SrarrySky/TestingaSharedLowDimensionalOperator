"""Construction of out-of-sample features under train-only preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.data import PreparedDataset
from model.dlcssm import DLCSSM
from model.geometry import nonnormality, shift_adjusted_nonnormality, spectral_radius
from model.propagation import selection_matrix, transmitter_scores
from model.uncertainty import gaussian_state_draws, posterior_metric_summary


@dataclass(frozen=True)
class OOSFeatureMatrix:
    """Chronologically aligned OOS candidate feature matrix.
    X contains candidate regression features.
    auxiliary contains aligned diagnostic quantities that should not automatically enter regressions, such as the peak propagation horizon.
    state_mean/state_cov preserve the forward-filtered latent posterior so posterior-aware geometry can be added later without refiltering.
    """

    X: np.ndarray
    feature_names: tuple[str, ...]
    dates: np.ndarray
    split: np.ndarray
    row_indices: np.ndarray
    state_mean: np.ndarray
    state_cov: np.ndarray
    auxiliary: np.ndarray
    auxiliary_names: tuple[str, ...]
    valid_rows: np.ndarray


def _align_extra_feature(
    values: np.ndarray,
    row_indices: np.ndarray,
    full_length: int,
    name: str,
) -> np.ndarray:
    """Align either full-dataset or already-aligned feature vectors."""
    x = np.asarray(
        values,
        dtype=float,
    )

    if x.ndim != 1:
        raise ValueError(f"Extra feature {name!r} must be 1D")

    aligned_length = len(row_indices)

    if len(x) == aligned_length:
        return x.copy()
    if len(x) == full_length:
        return x[row_indices].copy()

    raise ValueError(
        f"Extra feature {name!r} has length {len(x)}, "
        f"expected either {aligned_length} aligned observations "
        f"or {full_length} full-dataset observations."
    )


def build_point_oos_feature_matrix(
    ds: PreparedDataset,
    model: DLCSSM,
    *,
    horizons: tuple[int, ...] = (3, 5, 10),
    discount: float = 0.95,
    include_controls: bool = True,
    extra_features: Mapping[str, np.ndarray] | None = None,
    posterior_draws: int = 128,
    posterior_seed: int = 0,
) -> OOSFeatureMatrix:
    """Build point-estimate OOS geometry features using filtered states.
    IMPORTANT
    ---------
    The model is filtered ONCE over the complete chronological dataset.
    Because DLCSSM.filter() is forward-only, the state attached to date t depends only on information available through t.
    We deliberately do NOT:
        - filter Validation/Test independently;
        - use RTS-smoothed Validation/Test states;
        - use future filtered states;
        - use raw latent coordinates as scientific predictors.
    Features include posterior uncertainty summaries when "posterior_draws" is positive.
    The draws are generated independently for each filtered state using its Gaussian
    posterior, so all uncertainty features remain strictly forward-looking.
    """

    if model.params_ is None:
        raise RuntimeError("DLCSSM must be fit or loaded before building OOS features")

    params = model.params_

    horizons = tuple(sorted(set(int(h) for h in horizons)))

    if not horizons:
        raise ValueError("At least one horizon is required")
    if horizons[0] < 1:
        raise ValueError("All horizons must be >= 1")
    if not (0.0 < discount <= 1.0):
        raise ValueError("discount must lie in (0, 1]")
    if posterior_draws < 0:
        raise ValueError("posterior_draws must be non-negative")

    T = ds.x.shape[0]
    L = params.lag
    n = params.n

    if T <= L:
        raise ValueError("Dataset is shorter than model lag")

    # ------------------------------------------------------------
    # CRITICAL OOS RULE:
    # Filter the entire chronological sample ONCE.
    # Do NOT do:
    #     model.filter(x_test, u_test)
    # because that resets the state at the Test boundary.
    # ------------------------------------------------------------
    filt = model.filter(
        ds.x,
        ds.u,
    )
    row_indices = np.arange(
        L,
        T,
        dtype=int,
    )
    M = len(row_indices)

    if filt.mean.shape[0] != M:
        raise RuntimeError(
            "Filtered-state alignment mismatch: " f"expected {M}, got {filt.mean.shape[0]}"
        )

    dates = ds.dates[row_indices].copy()
    split = ds.split[row_indices].copy()

    # ------------------------------------------------------------
    # Candidate feature storage
    # ------------------------------------------------------------
    features: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------
    # Existing observable controls u_t.
    # ds.u has already been standardized with Train parameters by the data pipeline, but the later OOS feature transform may standardize the complete feature set again consistently.
    # ------------------------------------------------------------
    if include_controls:
        for j, raw_name in enumerate(ds.covariate_names):
            name = str(raw_name)
            features[f"control__{name}"] = ds.u[
                row_indices,
                j,
            ].astype(float)

    # ------------------------------------------------------------
    # Geometry arrays
    # ------------------------------------------------------------
    rho = np.empty(
        M,
        dtype=float,
    )
    companion_nn = np.empty(
        M,
        dtype=float,
    )
    dynamic_nn = np.empty(
        M,
        dtype=float,
    )
    posterior_rho_std = np.zeros(M, dtype=float)
    posterior_rho_lower = np.zeros(M, dtype=float)
    posterior_rho_upper = np.zeros(M, dtype=float)
    rng = np.random.default_rng(posterior_seed)
    max_horizon = max(horizons)

    # Exact observable response-operator norm:
    # ||J C^h J' ||_2
    # This is kept as auxiliary diagnostic information.
    observable_exact = np.empty(
        (
            M,
            max_horizon,
        ),
        dtype=float,
    )
    observable_peak_horizon = np.empty(
        M,
        dtype=int,
    )

    # Main horizon-specific feature containers.
    observable_peak_gain = {
        H: np.empty(
            M,
            dtype=float,
        )
        for H in horizons
    }

    total_path_energy = {
        H: np.empty(
            M,
            dtype=float,
        )
        for H in horizons
    }

    max_transmitter = {
        H: np.empty(
            M,
            dtype=float,
        )
        for H in horizons
    }

    transmitter_hhi = {
        H: np.empty(
            M,
            dtype=float,
        )
        for H in horizons
    }

    J = selection_matrix(
        n,
        L,
    )
    horizon_set = set(horizons)

    # ------------------------------------------------------------
    # One pass over filtered states.
    # We construct C, C^2, ..., C^H sequentially so we do not repeatedly call matrix_power for every horizon.
    # ------------------------------------------------------------
    for k, z in enumerate(filt.mean):
        C = params.companion(z)
        rho[k] = spectral_radius(C)
        if posterior_draws > 0:
            draws = gaussian_state_draws(filt.mean[k], filt.cov[k], posterior_draws, rng)
            summary = posterior_metric_summary(
                draws, lambda zd: spectral_radius(params.companion(zd))
            )
            posterior_rho_std[k] = summary["std"]
            posterior_rho_lower[k] = summary["lower"]
            posterior_rho_upper[k] = summary["upper"]
        companion_nn[k] = nonnormality(C)
        dynamic_nn[k] = shift_adjusted_nonnormality(
            C,
            n=n,
            lag=L,
        )
        power = np.eye(
            C.shape[0],
            dtype=float,
        )

        # Public propagation convention: P[source, target]
        P_accum = np.zeros(
            (
                n,
                n,
            ),
            dtype=float,
        )
        running_peak_gain = 0.0

        for h in range(
            1,
            max_horizon + 1,
        ):
            power = C @ power

            # Matrix orientation: R[target, source]
            R = J @ power @ J.T
            obs_norm = float(
                np.linalg.norm(
                    R,
                    2,
                )
            )
            observable_exact[
                k,
                h - 1,
            ] = obs_norm
            running_peak_gain = max(
                running_peak_gain,
                obs_norm,
            )

            # IMPORTANT:
            # R is [target, source].
            # P is [source, target].
            P_accum += (discount ** (h - 1)) * (R * R).T

            if h in horizon_set:
                observable_peak_gain[h][k] = running_peak_gain
                q = P_accum.copy()
                # System directional propagation should exclude
                # self-responses.
                np.fill_diagonal(
                    q,
                    0.0,
                )
                total_energy = float(q.sum())
                total_path_energy[h][k] = total_energy
                trans, _, _ = transmitter_scores(P_accum)
                max_transmitter[h][k] = float(np.max(trans))
                trans_total = float(trans.sum())

                if trans_total > 1e-18:
                    shares = trans / trans_total
                    transmitter_hhi[h][k] = float(np.sum(shares * shares))
                else:
                    transmitter_hhi[h][k] = 0.0

        observable_peak_horizon[k] = int(np.argmax(observable_exact[k])) + 1

    # ------------------------------------------------------------
    # Base geometry
    # ------------------------------------------------------------
    features["geom__spectral_radius"] = rho
    features["geom__companion_nonnormality"] = companion_nn
    features["geom__dynamic_nonnormality"] = dynamic_nn
    if posterior_draws > 0:
        features["uncertainty__spectral_radius_std"] = posterior_rho_std
        features["uncertainty__spectral_radius_lower"] = posterior_rho_lower
        features["uncertainty__spectral_radius_upper"] = posterior_rho_upper

    # ------------------------------------------------------------
    # Horizon-dependent geometry
    # ------------------------------------------------------------
    for H in horizons:
        features[f"geom__observable_peak_gain_h{H}"] = observable_peak_gain[H]
        features[f"geom__total_path_energy_h{H}"] = total_path_energy[H]
        features[f"geom__max_transmitter_h{H}"] = max_transmitter[H]
        features[f"geom__transmitter_hhi_h{H}"] = transmitter_hhi[H]

    # ------------------------------------------------------------
    # Future modules can append HMM / DY / pathwise / posterior features.
    # ------------------------------------------------------------
    if extra_features is not None:
        for name, values in extra_features.items():
            if name in features:
                raise ValueError(f"Duplicate feature name: {name}")

            features[name] = _align_extra_feature(
                values,
                row_indices,
                full_length=T,
                name=name,
            )

    feature_names = tuple(features.keys())
    X = np.column_stack([features[name] for name in feature_names])

    # ------------------------------------------------------------
    # Auxiliary / diagnostic values.
    # These are intentionally NOT automatically placed into X.
    # In particular, peak_horizon is categorical/descriptive rather than something we should blindly z-score as a continuous regressor.
    # ------------------------------------------------------------
    auxiliary: dict[
        str,
        np.ndarray,
    ] = {}

    for h in range(
        1,
        max_horizon + 1,
    ):
        auxiliary[f"diag__observable_norm_exact_h{h}"] = observable_exact[
            :,
            h - 1,
        ]

    auxiliary[f"diag__observable_peak_horizon_h{max_horizon}"] = observable_peak_horizon.astype(
        float
    )
    auxiliary_names = tuple(auxiliary.keys())
    auxiliary_matrix = np.column_stack([auxiliary[name] for name in auxiliary_names])

    # Later rolling baselines may introduce warm-up NaNs.
    # We keep the aligned rows but explicitly mark availability.
    valid_rows = np.all(
        np.isfinite(X),
        axis=1,
    )

    return OOSFeatureMatrix(
        X=X,
        feature_names=feature_names,
        dates=dates,
        split=split,
        row_indices=row_indices,
        state_mean=filt.mean.copy(),
        state_cov=filt.cov.copy(),
        auxiliary=auxiliary_matrix,
        auxiliary_names=auxiliary_names,
        valid_rows=valid_rows,
    )


def append_oos_features(
    bundle: OOSFeatureMatrix,
    features: Mapping[
        str,
        np.ndarray,
    ],
) -> OOSFeatureMatrix:
    """Append additional already-computed OOS features.
    Useful later for:
        - HMM filtered stress probability
        - Diebold-Yilmaz connectedness
        - TVP risk summaries
        - posterior mean geometry
        - state-migration gap
        - susceptibility
    Input vectors may either already have aligned length M or have full-dataset length compatible with bundle.row_indices.
    """
    current_names = list(bundle.feature_names)
    new_columns = [bundle.X]
    M = len(bundle.row_indices)
    max_original_index = int(bundle.row_indices.max())

    for name, values in features.items():
        if name in current_names:
            raise ValueError(f"Duplicate feature name: {name}")

        x = np.asarray(
            values,
            dtype=float,
        )

        if x.ndim != 1:
            raise ValueError(f"Feature {name!r} must be 1D")

        if len(x) == M:
            aligned = x
        elif len(x) > max_original_index:
            aligned = x[bundle.row_indices]
        else:
            raise ValueError(
                f"Cannot align feature {name!r}; " f"length={len(x)}, aligned length={M}"
            )

        new_columns.append(aligned[:, None])
        current_names.append(name)

    X = np.column_stack(new_columns)
    valid_rows = np.all(
        np.isfinite(X),
        axis=1,
    )

    return OOSFeatureMatrix(
        X=X,
        feature_names=tuple(current_names),
        dates=bundle.dates.copy(),
        split=bundle.split.copy(),
        row_indices=bundle.row_indices.copy(),
        state_mean=bundle.state_mean.copy(),
        state_cov=bundle.state_cov.copy(),
        auxiliary=bundle.auxiliary.copy(),
        auxiliary_names=bundle.auxiliary_names,
        valid_rows=valid_rows,
    )


def save_oos_feature_matrix(
    bundle: OOSFeatureMatrix,
    path: str | Path,
) -> None:
    """Save aligned OOS feature bundle without pickle."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        X=bundle.X,
        feature_names=np.asarray(
            bundle.feature_names,
            dtype="U128",
        ),
        dates=bundle.dates,
        split=bundle.split,
        row_indices=bundle.row_indices,
        state_mean=bundle.state_mean,
        state_cov=bundle.state_cov,
        auxiliary=bundle.auxiliary,
        auxiliary_names=np.asarray(
            bundle.auxiliary_names,
            dtype="U128",
        ),
        valid_rows=bundle.valid_rows.astype(np.uint8),
    )


def load_oos_feature_matrix(
    path: str | Path,
) -> OOSFeatureMatrix:
    """Load an OOS feature bundle."""
    with np.load(
        path,
        allow_pickle=False,
    ) as z:
        return OOSFeatureMatrix(
            X=z["X"],
            feature_names=tuple(str(x) for x in z["feature_names"]),
            dates=z["dates"],
            split=z["split"],
            row_indices=z["row_indices"],
            state_mean=z["state_mean"],
            state_cov=z["state_cov"],
            auxiliary=z["auxiliary"],
            auxiliary_names=tuple(str(x) for x in z["auxiliary_names"]),
            valid_rows=z["valid_rows"].astype(bool),
        )
