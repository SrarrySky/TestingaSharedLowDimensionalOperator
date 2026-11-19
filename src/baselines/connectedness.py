"""Generalized FEVD and Diebold-Yilmaz connectedness (percent units).

Rows are receiving targets, columns are shock sources. These are predictive
connectedness measures, not identified causal effects. No Cholesky ordering.
"""

from __future__ import annotations

import numpy as np

from baselines.var_baselines import positive_int


def impulse_responses(a, horizon):
    """Psi_0,...,Psi_{H-1}, with Psi_h=sum_l A_l Psi_{h-l}."""
    H = positive_int(horizon, "horizon")
    a = np.asarray(a, dtype=float)
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("A must be finite [lag, target, source]")
    n = a.shape[1]
    psi = np.zeros((H, n, n))
    psi[0] = np.eye(n)
    for h in range(1, H):
        for ell in range(1, min(len(a), h) + 1):
            psi[h] += a[ell - 1] @ psi[h - ell]
    if not np.isfinite(psi).all():
        raise FloatingPointError("Nonfinite finite-horizon impulse response")
    return psi


def generalized_fevd(a, sigma, horizon=5):
    """
    Generalized fevd.
    :param a: Input argument used by this operation.
    :param sigma: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    psi = impulse_responses(a, horizon)
    sigma = np.asarray(sigma, dtype=float)
    n = psi.shape[1]
    if sigma.shape != (n, n) or not np.isfinite(sigma).all() or not np.allclose(sigma, sigma.T):
        raise ValueError("Sigma must be finite and symmetric with matching dimensions")
    np.linalg.cholesky(sigma)
    ps = psi @ sigma
    denominator = np.einsum("hij,hij->i", ps, psi)
    numerator = np.sum(ps**2, axis=0) / np.diag(sigma)[None, :]
    raw = numerator / denominator[:, None]
    normalized = raw / raw.sum(axis=1, keepdims=True)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Nonfinite generalized FEVD")
    return raw, normalized


def dy_connectedness(a, sigma, horizon=5):
    """
    Dy connectedness.
    :param a: Input argument used by this operation.
    :param sigma: Input argument used by this operation.
    :param horizon: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    raw, theta = generalized_fevd(a, sigma, horizon)
    off = theta.copy()
    np.fill_diagonal(off, 0.0)
    from_ = 100 * off.sum(axis=1)
    to = 100 * off.sum(axis=0)
    return dict(
        gfevd_raw=raw, gfevd=theta, from_=from_, to=to, net=to - from_, total=float(from_.mean())
    )


def innovation_risk(a, sigma, horizon=5):
    """Mean-sector H-step forecast-error variance, frozen coefficients at t.

    Sum_{h=0}^{H-1} Psi_h Sigma Psi_h'; pass raw-unit A/Sigma for return^2.
    Conditional innovation risk excludes uncertainty in future coefficients.
    """
    psi = impulse_responses(a, horizon)
    return float(np.trace(np.sum((psi @ sigma) @ psi.transpose(0, 2, 1), axis=0)) / psi.shape[1])


def companion_radius(a):
    """
    Companion radius.
    :param a: Input argument used by this operation.
    :return: Result produced by this operation.
    """
    a = np.asarray(a)
    p, n, _ = a.shape
    c = np.zeros((p * n, p * n))
    c[:n] = np.concatenate(a, axis=1)
    if p > 1:
        c[n:, :-n] = np.eye((p - 1) * n)
    return float(np.max(np.abs(np.linalg.eigvals(c))))


def frequency_connectedness(
    a,
    sigma,
    *,
    bands=((0.0, np.pi / 4.0), (np.pi / 4.0, np.pi)),
    horizon=256,
    frequency_grid=None,
):
    """Baruník--Křehlík-style frequency FEVD for a reduced-form VAR.

    The implementation evaluates the VAR transfer function on a deterministic
    frequency grid and integrates generalized forecast-error shares over the
    declared bands.  Rows are receiving targets and columns are shock sources,
    matching :func:`generalized_fevd`.  The result is a predictive frequency
    decomposition; it is not a structural identification or causal effect.

    "horizon" controls the frequency-grid resolution when "frequency_grid"
    is omitted.  Bands use radians and are half-open except for the final grid
    endpoint.  The returned "frequency_gfevd" has shape "(band, target,
    source)" and each target row sums to one within numerical tolerance.
    """
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("A must be finite with shape (lag, n, n)")
    n = a.shape[1]
    if sigma.shape != (n, n) or not np.isfinite(sigma).all() or not np.allclose(sigma, sigma.T):
        raise ValueError("Sigma must be finite symmetric with matching dimensions")
    np.linalg.cholesky(sigma)
    if frequency_grid is None:
        if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer) or horizon < 8:
            raise ValueError("horizon must be an integer >= 8")
        omega = np.linspace(0.0, np.pi, int(horizon), endpoint=True)
    else:
        omega = np.asarray(frequency_grid, dtype=float)
        if omega.ndim != 1 or len(omega) < 2 or not np.isfinite(omega).all():
            raise ValueError("frequency_grid must be a finite one-dimensional grid")
        if omega[0] < 0 or omega[-1] > np.pi or np.any(np.diff(omega) <= 0):
            raise ValueError("frequency_grid must increase within [0, pi]")
    clean_bands = []
    for band in bands:
        if len(band) != 2:
            raise ValueError("Each frequency band must be (low, high)")
        low, high = map(float, band)
        if not 0 <= low < high <= np.pi:
            raise ValueError("Frequency bands must satisfy 0 <= low < high <= pi")
        clean_bands.append((low, high))
    if not clean_bands:
        raise ValueError("At least one frequency band is required")

    # Include every requested boundary in the integration grid.  This makes
    # band results invariant to whether a boundary happened to coincide with
    # the default grid, while retaining a deterministic trapezoidal rule.
    omega = np.unique(np.concatenate((omega, np.asarray(clean_bands).ravel())))

    def transfer(w):
        """
        Transfer.
        :param w: Input argument used by this operation.
        :return: Result produced by this operation.
        """
        polynomial = np.eye(n, dtype=complex)
        for ell, matrix in enumerate(a, start=1):
            polynomial -= matrix * np.exp(-1j * w * ell)
        return np.linalg.inv(polynomial)

    density = np.empty((len(omega), n, n), dtype=float)
    diagonal = np.diag(sigma)
    for index, w in enumerate(omega):
        psi = transfer(w)
        # Generalized spectral FEVD density: source j's innovation variance is
        # divided out.  Normalization is performed against the *full* frequency
        # range below, not separately inside each band.
        response = psi @ sigma
        density[index] = np.abs(response) ** 2 / diagonal[None, :]

    shares = np.empty((len(clean_bands), n, n), dtype=float)
    widths = np.empty(len(clean_bands), dtype=float)
    full_integrated = np.trapezoid(density, omega, axis=0)
    denominator = np.maximum(full_integrated.sum(axis=1, keepdims=True), 1e-15)
    for index, (low, high) in enumerate(clean_bands):
        mask = (omega >= low) & (omega <= high)
        if mask.sum() < 2:
            raise ValueError(f"Frequency band {(low, high)} has fewer than two grid points")
        integrated = np.trapezoid(density[mask], omega[mask], axis=0)
        shares[index] = integrated / denominator
        widths[index] = high - low
    off = shares.copy()
    for matrix in off:
        np.fill_diagonal(matrix, 0.0)
    from_ = 100.0 * off.sum(axis=2)
    to = 100.0 * off.sum(axis=1)
    net = to - from_
    total = 100.0 * off.sum(axis=(1, 2)) / n
    return {
        "frequency_gfevd": shares,
        "frequency_grid": omega,
        "bands": np.asarray(clean_bands),
        "band_widths": widths,
        "from_": from_,
        "to": to,
        "net": net,
        "total": total,
        "definition": "generalized frequency FEVD; predictive, reduced-form, no structural causal identification",
    }
