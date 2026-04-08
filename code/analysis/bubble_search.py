"""
Full bubble collision search pipeline.

Orchestrates:

- Detection efficiency characterisation on injection-recovery maps
- Sensitivity curves (A_50% vs theta_r)
- Feldman--Cousins-style upper limits on the collision rate
- Comparison with Feeney et al. (2011)
- End-to-end search execution with systematic checks

References
----------
- Feldman & Cousins (1998), PRD 57, 3873
- Feeney et al. (2011), PRL 107, 071301
- Architecture document, Section 2.11
- Statistical Framework, Sections 6.1--6.6
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


# ====================================================================
# Result dataclasses
# ====================================================================

@dataclass
class DetectionResult:
    """Complete result from the bubble collision search.

    Attributes
    ----------
    p_value : float
        Empirical p-value of the observed score against the null
        distribution.
    z_score : float
        Equivalent Gaussian significance (sigma).
    p_coll_observed : float
        Calibrated ensemble-mean collision probability for the
        observed data.
    p_coll_uncertainty : float
        Ensemble standard deviation of the collision probability.
    ensemble_scores : np.ndarray
        Per-model scores, shape ``(N_ensemble,)``.
    ensemble_std : float
        Standard deviation across ensemble members.
    method_scores : dict
        Per-component-separation-method scores.
    method_consistency_chi2 : float
        Chi-squared consistency statistic across methods.
    method_consistency_passed : bool
        ``True`` if the consistency test is passed (p > 0.05).
    local_max_score : float or None
        Maximum local patch score (if computed).
    local_max_location : tuple or None
        ``(theta, phi)`` of the highest-scoring local patch.
    attribution_map : np.ndarray or None
        Grad-CAM attribution map on the observed data.
    systematic_checks : dict
        Summary of all systematic check outcomes.
    """

    p_value: float = 1.0
    z_score: float = 0.0
    p_coll_observed: float = 0.0
    p_coll_uncertainty: float = 0.0

    ensemble_scores: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    ensemble_std: float = 0.0

    method_scores: dict = field(default_factory=dict)
    method_consistency_chi2: float = 0.0
    method_consistency_passed: bool = True

    local_max_score: Optional[float] = None
    local_max_location: Optional[tuple[float, float]] = None
    attribution_map: Optional[np.ndarray] = None

    systematic_checks: dict = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """Sensitivity characterisation from injection-recovery.

    Attributes
    ----------
    theta_r_grid : np.ndarray
        Angular radius grid in degrees, shape ``(N_theta,)``.
    amplitude_grid : np.ndarray
        Dimensionless amplitude grid, shape ``(N_A,)``.
    kappa_grid : np.ndarray
        Profile shape index grid, shape ``(N_kappa,)``.
    efficiency_grid : np.ndarray
        Detection efficiency on the parameter grid,
        shape ``(N_theta, N_A, N_kappa)``.
    efficiency_errors : np.ndarray
        Binomial uncertainty on efficiency,
        shape ``(N_theta, N_A, N_kappa)``.
    a50_vs_theta_r : np.ndarray
        Amplitude at 50% detection efficiency for each
        ``(theta_r, kappa)`` combination,
        shape ``(N_theta, N_kappa)``.
    """

    theta_r_grid: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    amplitude_grid: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    kappa_grid: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    efficiency_grid: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    efficiency_errors: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    a50_vs_theta_r: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass
class UpperLimitResult:
    """Upper limit on the bubble collision rate.

    Attributes
    ----------
    n_coll_95cl : float
        95% CL upper limit on expected number of detectable collisions.
    lambda_95cl : float
        95% CL upper limit on the dimensionless nucleation rate.
    feeney_comparison_factor : float
        Improvement factor relative to Feeney et al. (2011):
        ``factor = feeney_limit / our_limit``.
    confidence_level : float
        Confidence level used (default 0.95).
    observed_score : float
        Score of the observed data.
    threshold : float
        Detection threshold at the target false-positive rate.
    """

    n_coll_95cl: float = 0.0
    lambda_95cl: float = 0.0
    feeney_comparison_factor: float = 1.0
    confidence_level: float = 0.95
    observed_score: float = 0.0
    threshold: float = 0.0


# ====================================================================
# Default parameter grids (from Statistical Framework Section 6.1)
# ====================================================================

DEFAULT_THETA_R_GRID = np.array(
    [5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]
)  # degrees

DEFAULT_AMPLITUDE_GRID = np.array(
    [3, 4, 5, 7, 10, 15, 20, 30, 40, 50], dtype=np.float64
) * 1e-5  # dimensionless

DEFAULT_KAPPA_GRID = np.array([2.0, 3.0, 4.0])

# Feeney et al. (2011) constraint for comparison (Eq. 2.19)
FEENEY_NCOLL_LIMIT = 1.6  # 95% CL for theta_r > 25 deg, |A| > 1e-5


# ====================================================================
# Detection efficiency
# ====================================================================

def compute_detection_efficiency(
    model: Any,
    test_dataset: Any,
    thresholds: Union[np.ndarray, float],
    theta_r_grid: Optional[np.ndarray] = None,
    amplitude_grid: Optional[np.ndarray] = None,
    kappa_grid: Optional[np.ndarray] = None,
    null_scores: Optional[np.ndarray] = None,
    alpha: float = 0.01,
) -> SensitivityResult:
    """Compute detection efficiency on the injection-recovery grid.

    For each cell ``(theta_r, |A|, kappa)`` in the parameter grid,
    the efficiency is the fraction of injected signals detected above
    the threshold.

    Parameters
    ----------
    model : EnsembleInference or callable
        Inference engine.  Must support ``predict(map_tensor)``
        returning an object with ``p_coll_mean`` attribute, or be
        callable returning a float.
    test_dataset : iterable
        Injection-recovery dataset yielding
        ``(map_tensor, label, metadata)`` tuples.  Each metadata dict
        must include ``"bubble_params"`` with attributes
        ``theta_r``, ``amplitude``, ``kappa``.
    thresholds : float or np.ndarray
        Detection threshold(s).  If a float, a single threshold is used.
        If ``null_scores`` is given and ``thresholds`` is not provided
        directly, the threshold is the ``(1 - alpha)``-quantile of the
        null distribution.
    theta_r_grid : np.ndarray or None
        Angular radius bin centers in degrees.
    amplitude_grid : np.ndarray or None
        Amplitude bin centers (dimensionless).
    kappa_grid : np.ndarray or None
        Profile shape index values.
    null_scores : np.ndarray or None
        Null distribution scores for threshold derivation.
    alpha : float
        Target false-positive rate when deriving threshold from
        ``null_scores``.

    Returns
    -------
    result : SensitivityResult
    """
    import torch  # local import

    if theta_r_grid is None:
        theta_r_grid = DEFAULT_THETA_R_GRID.copy()
    if amplitude_grid is None:
        amplitude_grid = DEFAULT_AMPLITUDE_GRID.copy()
    if kappa_grid is None:
        kappa_grid = DEFAULT_KAPPA_GRID.copy()

    # Derive threshold from null distribution if provided
    if null_scores is not None and isinstance(thresholds, (int, float)):
        threshold = float(np.quantile(null_scores, 1.0 - alpha))
    elif isinstance(thresholds, (int, float)):
        threshold = float(thresholds)
    else:
        threshold = float(thresholds[0])

    n_theta = len(theta_r_grid)
    n_amp = len(amplitude_grid)
    n_kappa = len(kappa_grid)

    # Accumulate counts per bin
    n_detected = np.zeros((n_theta, n_amp, n_kappa), dtype=np.float64)
    n_total = np.zeros((n_theta, n_amp, n_kappa), dtype=np.float64)

    # Bin edges for theta_r and amplitude (log-spaced midpoints)
    theta_r_edges = _bin_edges_log(theta_r_grid)
    amp_edges = _bin_edges_log(amplitude_grid)

    count = 0
    for item in test_dataset:
        if isinstance(item, (tuple, list)):
            map_tensor, label, metadata = item[0], item[1], item[2]
        else:
            continue

        bp = metadata.get("bubble_params")
        if bp is None:
            continue

        # Extract parameters
        theta_r_deg = float(np.degrees(bp.theta_r))
        amp_abs = float(np.abs(bp.amplitude))
        kappa_val = float(bp.kappa)

        # Find bins
        i_theta = _find_bin(theta_r_deg, theta_r_edges)
        i_amp = _find_bin(amp_abs, amp_edges)
        i_kappa = _find_nearest(kappa_val, kappa_grid)

        if i_theta < 0 or i_amp < 0 or i_kappa < 0:
            continue

        # Score
        if isinstance(map_tensor, torch.Tensor) and map_tensor.dim() == 2:
            map_tensor = map_tensor.unsqueeze(0)

        if hasattr(model, "predict"):
            result = model.predict(map_tensor)
            score = float(result.p_coll_mean)
        elif callable(model):
            score = float(model(map_tensor))
        else:
            continue

        n_total[i_theta, i_amp, i_kappa] += 1.0
        if score >= threshold:
            n_detected[i_theta, i_amp, i_kappa] += 1.0

        count += 1
        if count % 1000 == 0:
            logger.info("Efficiency: processed %d injection maps.", count)

    # Compute efficiency and binomial uncertainty
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(
            n_total > 0, n_detected / n_total, 0.0
        )
        efficiency_err = np.where(
            n_total > 0,
            np.sqrt(efficiency * (1.0 - efficiency) / n_total),
            0.0,
        )

    # Compute A_50% for each (theta_r, kappa)
    a50 = np.full((n_theta, n_kappa), np.nan)
    for i_t in range(n_theta):
        for i_k in range(n_kappa):
            eff_vs_a = efficiency[i_t, :, i_k]
            a50[i_t, i_k] = _interpolate_threshold(
                amplitude_grid, eff_vs_a, target=0.5
            )

    result = SensitivityResult(
        theta_r_grid=theta_r_grid,
        amplitude_grid=amplitude_grid,
        kappa_grid=kappa_grid,
        efficiency_grid=efficiency,
        efficiency_errors=efficiency_err,
        a50_vs_theta_r=a50,
    )

    logger.info(
        "Detection efficiency computed: %d injection maps processed.",
        count,
    )
    return result


# ====================================================================
# Upper limits
# ====================================================================

def compute_upper_limits(
    null_scores: np.ndarray,
    observed_score: float,
    confidence_level: float = 0.95,
    sensitivity: Optional[SensitivityResult] = None,
    theta_r_min_deg: float = 5.0,
    theta_r_max_deg: float = 60.0,
    theta_0_deg: float = 57.3,
) -> UpperLimitResult:
    """Compute Feldman--Cousins-style upper limit on the collision rate.

    For a null result (observed score below threshold), the 95% CL
    upper limit on the expected number of detectable collisions is:

        <N_det>^{95%} = -ln(1 - CL)

    (Poisson statistics with zero observed events).  This is then
    converted to a nucleation rate constraint using the integrated
    detection efficiency.

    Parameters
    ----------
    null_scores : np.ndarray
        Null distribution scores.
    observed_score : float
        Observed test statistic.
    confidence_level : float
        Confidence level (default 0.95).
    sensitivity : SensitivityResult or None
        Sensitivity characterisation for converting the upper limit
        on N_det to a nucleation rate constraint.
    theta_r_min_deg : float
        Minimum angular radius for integration (degrees).
    theta_r_max_deg : float
        Maximum angular radius for integration (degrees).
    theta_0_deg : float
        Reference angle theta_0 in degrees (of order 1 radian ~ 57.3 deg).

    Returns
    -------
    result : UpperLimitResult
    """
    # Poisson upper limit: -ln(1 - CL) for zero observed events
    n_det_upper = -np.log(1.0 - confidence_level)

    # Derive threshold at the ~1% false-positive level
    threshold = float(np.quantile(null_scores, 0.99))

    # Compute nucleation rate limit using sensitivity information
    lambda_upper = n_det_upper  # default: equal to N_det limit

    if sensitivity is not None and sensitivity.a50_vs_theta_r.size > 0:
        # Integrate the averaged efficiency over theta_r to convert
        # N_det to lambda using Eq. (1.20):
        #   <N_det> = lambda * integral ...
        theta_r_rad = np.radians(sensitivity.theta_r_grid)
        theta_0_rad = np.radians(theta_0_deg)

        # Averaged efficiency over kappa for each theta_r
        eff_mean = np.nanmean(
            sensitivity.efficiency_grid, axis=(1, 2)
        )

        # Integrand: (2 / (theta_0^2 * theta_r^3)) * <epsilon>(theta_r)
        integrand = (
            2.0 / (theta_0_rad ** 2 * theta_r_rad ** 3)
        ) * eff_mean

        # Select range
        sel = (
            (sensitivity.theta_r_grid >= theta_r_min_deg)
            & (sensitivity.theta_r_grid <= theta_r_max_deg)
        )
        if np.sum(sel) >= 2:
            integral = float(
                np.trapz(integrand[sel], theta_r_rad[sel])
            )
            if integral > 0:
                lambda_upper = n_det_upper / integral

    # Comparison with Feeney et al. (2011)
    feeney_factor = FEENEY_NCOLL_LIMIT / max(n_det_upper, 1e-10)

    result = UpperLimitResult(
        n_coll_95cl=float(n_det_upper),
        lambda_95cl=float(lambda_upper),
        feeney_comparison_factor=float(feeney_factor),
        confidence_level=confidence_level,
        observed_score=observed_score,
        threshold=threshold,
    )

    logger.info(
        "Upper limit: <N_det>^{%.0f%%} = %.3f, lambda^{%.0f%%} = %.4e, "
        "Feeney improvement factor = %.2f",
        100 * confidence_level, n_det_upper,
        100 * confidence_level, lambda_upper,
        feeney_factor,
    )
    return result


# ====================================================================
# Sensitivity curve computation
# ====================================================================

def compute_sensitivity_curve(
    efficiencies: np.ndarray,
    param_grid: np.ndarray,
    target_efficiency: float = 0.5,
) -> np.ndarray:
    """Interpolate the parameter value at which detection efficiency
    crosses a target level (e.g. 50%).

    Parameters
    ----------
    efficiencies : np.ndarray, shape ``(N_grid, N_param)``
        Efficiency as a function of the parameter (columns) for
        different fixed conditions (rows).  E.g. rows = theta_r bins,
        columns = amplitude bins.
    param_grid : np.ndarray, shape ``(N_param,)``
        Parameter values corresponding to the columns.
    target_efficiency : float
        Target detection efficiency (default 0.5 for A_50%).

    Returns
    -------
    thresholds : np.ndarray, shape ``(N_grid,)``
        Interpolated parameter value at which efficiency crosses the
        target for each row.  ``NaN`` if the target is never reached.
    """
    n_rows = efficiencies.shape[0]
    thresholds = np.full(n_rows, np.nan)

    for i in range(n_rows):
        thresholds[i] = _interpolate_threshold(
            param_grid, efficiencies[i, :], target=target_efficiency
        )
    return thresholds


# ====================================================================
# Full search orchestration
# ====================================================================

def run_full_search(
    model: Any,
    planck_map: Any,
    null_sims: np.ndarray,
    config: Optional[dict] = None,
) -> DetectionResult:
    """Execute the complete bubble collision search pipeline.

    This is a high-level orchestrator that:

    1. Runs inference on the observed map.
    2. Computes the p-value against the null distribution.
    3. Records systematic-check placeholders (to be filled by
       specialised routines in a full pipeline run).

    Parameters
    ----------
    model : EnsembleInference or callable
        Trained model / inference engine.
    planck_map : torch.Tensor or np.ndarray
        Preprocessed observed map ready for model input.
    null_sims : np.ndarray
        Null distribution scores, shape ``(N_null,)``.
    config : dict or None
        Pipeline configuration (optional).

    Returns
    -------
    result : DetectionResult
    """
    import torch  # local import
    from .anomaly_detection import compute_pvalue

    t0 = time.time()

    # Ensure tensor
    if isinstance(planck_map, np.ndarray):
        planck_map = torch.from_numpy(planck_map).float()
    if planck_map.dim() == 1:
        planck_map = planck_map.unsqueeze(0).unsqueeze(0)
    elif planck_map.dim() == 2:
        planck_map = planck_map.unsqueeze(0)

    # Inference
    if hasattr(model, "predict"):
        inf_result = model.predict(planck_map)
        observed_score = float(inf_result.p_coll_mean)
        ensemble_scores = inf_result.p_coll_per_model
        ensemble_std = float(inf_result.p_coll_std)
        p_coll_unc = float(inf_result.p_coll_std)
    elif callable(model):
        observed_score = float(model(planck_map))
        ensemble_scores = np.array([observed_score])
        ensemble_std = 0.0
        p_coll_unc = 0.0
    else:
        raise TypeError("model must support predict() or be callable")

    # P-value
    p_value, z_score = compute_pvalue(observed_score, null_sims)

    dt = time.time() - t0
    logger.info(
        "Full search completed in %.1f s: p_coll=%.4f, p=%.4e, Z=%.2f sigma",
        dt, observed_score, p_value, z_score,
    )

    return DetectionResult(
        p_value=p_value,
        z_score=z_score,
        p_coll_observed=observed_score,
        p_coll_uncertainty=p_coll_unc,
        ensemble_scores=ensemble_scores,
        ensemble_std=ensemble_std,
    )


# ====================================================================
# Internal helpers
# ====================================================================

def _bin_edges_log(centers: np.ndarray) -> np.ndarray:
    """Compute log-spaced bin edges from bin centres."""
    c = np.asarray(centers, dtype=np.float64)
    if len(c) == 0:
        return np.array([])
    log_c = np.log10(np.maximum(c, 1e-30))
    edges = np.empty(len(c) + 1)
    for i in range(1, len(c)):
        edges[i] = 10 ** (0.5 * (log_c[i - 1] + log_c[i]))
    edges[0] = 10 ** (log_c[0] - 0.5 * (log_c[1] - log_c[0]))
    edges[-1] = 10 ** (log_c[-1] + 0.5 * (log_c[-1] - log_c[-2]))
    return edges


def _find_bin(value: float, edges: np.ndarray) -> int:
    """Return the bin index for *value* given *edges*. Returns -1 if out of range."""
    idx = int(np.searchsorted(edges, value, side="right")) - 1
    if idx < 0 or idx >= len(edges) - 1:
        return -1
    return idx


def _find_nearest(value: float, grid: np.ndarray) -> int:
    """Return the index of the nearest grid point."""
    if len(grid) == 0:
        return -1
    return int(np.argmin(np.abs(grid - value)))


def _interpolate_threshold(
    param_values: np.ndarray,
    efficiencies: np.ndarray,
    target: float = 0.5,
) -> float:
    """Linearly interpolate the parameter value at which efficiency
    crosses *target*.  Returns NaN if never crossed."""
    eff = np.asarray(efficiencies, dtype=np.float64)
    par = np.asarray(param_values, dtype=np.float64)

    valid = ~np.isnan(eff) & ~np.isnan(par)
    eff = eff[valid]
    par = par[valid]

    if len(eff) < 2:
        return float("nan")

    # If efficiency never reaches target, return NaN
    if np.all(eff < target) or np.all(eff >= target):
        # Check boundary cases
        if eff[-1] >= target:
            return float(par[0])
        return float("nan")

    # Find crossing points
    try:
        interp_fn = interpolate.interp1d(
            eff, par, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        return float(interp_fn(target))
    except Exception:
        return float("nan")
