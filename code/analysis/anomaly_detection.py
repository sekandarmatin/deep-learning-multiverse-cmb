"""
Detection statistics for the bubble collision search.

Provides:

- Empirical p-value computation with Davison--Hinkley correction
- Gaussian significance (Z-score) conversion
- Look-elsewhere (global) p-value correction
- Benjamini--Hochberg false-discovery-rate control
- Null-distribution generation via ensemble inference
- Kolmogorov--Smirnov domain-shift diagnostic
- Cross-method consistency tests

References
----------
- Davison & Hinkley (1997), *Bootstrap Methods and Their Application*
- Benjamini & Hochberg (1995), JRSS-B 57, 289
- Architecture document, Section 2.10
- Statistical Framework, Sections 3--4, 7, 8
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ====================================================================
# P-value and significance
# ====================================================================

def compute_pvalue(
    observed_score: float,
    null_scores: np.ndarray,
) -> tuple[float, float]:
    """Compute the empirical p-value and equivalent Gaussian significance.

    Uses the Davison & Hinkley (1997) convention:

        p = (1 + k) / (1 + N)

    where *k* = number of null scores >= *observed_score* and
    *N* = ``len(null_scores)``.  The "+1" ensures the p-value is never
    exactly zero.

    The Gaussian significance (Z-score) is the probit of ``1 - p``:

        Z = Phi^{-1}(1 - p)

    Parameters
    ----------
    observed_score : float
        Test statistic evaluated on the observed data.
    null_scores : np.ndarray, shape ``(N_null,)``
        Test statistic values from null (LCDM) simulations.

    Returns
    -------
    p_value : float
        Empirical one-sided upper-tail p-value.
    z_score : float
        Equivalent Gaussian significance (number of sigma).
    """
    null = np.asarray(null_scores, dtype=np.float64).ravel()
    n_null = len(null)
    k = int(np.sum(null >= observed_score))

    p_value = (1.0 + k) / (1.0 + n_null)

    # Convert to Gaussian significance
    # Clip to avoid infinity at p = 0 or p = 1
    p_clipped = np.clip(p_value, 1e-15, 1.0 - 1e-15)
    z_score = float(stats.norm.ppf(1.0 - p_clipped))

    logger.info(
        "Observed score=%.6f, k=%d/%d null exceed it -> "
        "p=%.4e, Z=%.2f sigma",
        observed_score, k, n_null, p_value, z_score,
    )
    return float(p_value), z_score


# ====================================================================
# Look-elsewhere (global) p-value
# ====================================================================

def compute_global_pvalue(
    local_pvalue: float,
    n_effective_trials: int,
) -> tuple[float, float]:
    """Compute the global p-value using the Bonferroni correction.

    p_global = min(1, N_trials * p_local)

    Parameters
    ----------
    local_pvalue : float
        Local p-value (before look-elsewhere correction).
    n_effective_trials : int
        Effective number of independent trials.

    Returns
    -------
    p_global : float
        Global (look-elsewhere-corrected) p-value.
    z_global : float
        Equivalent global Gaussian significance.
    """
    p_global = min(1.0, n_effective_trials * local_pvalue)
    p_clipped = np.clip(p_global, 1e-15, 1.0 - 1e-15)
    z_global = float(stats.norm.ppf(1.0 - p_clipped))
    return float(p_global), z_global


# ====================================================================
# Multiple-testing correction (FDR)
# ====================================================================

def benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply the Benjamini--Hochberg FDR procedure.

    Given *M* p-values, sorts them in ascending order, finds the
    largest *k* such that ``p_(k) <= k * alpha / M``, and rejects
    hypotheses corresponding to ``p_(1), ..., p_(k)``.

    Parameters
    ----------
    pvalues : np.ndarray, shape ``(M,)``
        Unsorted array of p-values.
    alpha : float
        Target false-discovery rate.

    Returns
    -------
    rejected : np.ndarray, shape ``(M,)``, dtype bool
        ``True`` where the corresponding hypothesis is rejected
        (i.e., significant after FDR control).
    """
    pvals = np.asarray(pvalues, dtype=np.float64).ravel()
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool)

    # Sort p-values and track original indices
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]

    # BH threshold for each rank k (1-indexed)
    bh_threshold = (np.arange(1, m + 1) / m) * alpha

    # Find the largest k for which p_(k) <= k * alpha / M
    below = sorted_p <= bh_threshold
    if not np.any(below):
        return np.zeros(m, dtype=bool)

    k_max = int(np.max(np.where(below)[0]))  # 0-indexed

    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx[: k_max + 1]] = True

    n_rejected = int(rejected.sum())
    logger.info(
        "Benjamini-Hochberg (alpha=%.3f): %d / %d hypotheses rejected.",
        alpha, n_rejected, m,
    )
    return rejected


# ====================================================================
# Null distribution generation
# ====================================================================

def compute_null_distribution(
    model: Any,
    null_dataset: Any,
    n_sims: int = 10_000,
) -> np.ndarray:
    """Generate the null score distribution by running inference on null maps.

    Parameters
    ----------
    model : EnsembleInference or callable
        Trained model or inference engine.  Must support a
        ``predict(map_tensor) -> InferenceResult`` method or be a
        callable returning a float score for a single map tensor.
    null_dataset : iterable
        Dataset yielding ``(map_tensor, label, metadata)`` tuples.
        Only the first ``n_sims`` items are used.
    n_sims : int
        Number of null simulations to evaluate.

    Returns
    -------
    null_scores : np.ndarray, shape ``(n_sims,)``
        Ensemble-mean calibrated probability for each null map.
    """
    import torch  # local import to avoid hard dependency at module level

    scores: list[float] = []
    for i, item in enumerate(null_dataset):
        if i >= n_sims:
            break

        if isinstance(item, (tuple, list)):
            map_tensor = item[0]
        else:
            map_tensor = item

        # Ensure batch dimension
        if isinstance(map_tensor, torch.Tensor):
            if map_tensor.dim() == 2:
                map_tensor = map_tensor.unsqueeze(0)
        elif isinstance(map_tensor, np.ndarray):
            map_tensor = torch.from_numpy(map_tensor).float()
            if map_tensor.dim() == 2:
                map_tensor = map_tensor.unsqueeze(0)

        # Get score
        if hasattr(model, "predict"):
            result = model.predict(map_tensor)
            score = float(result.p_coll_mean)
        elif callable(model):
            score = float(model(map_tensor))
        else:
            raise TypeError(
                f"model must have a 'predict' method or be callable; "
                f"got {type(model)}"
            )

        scores.append(score)

        if (i + 1) % 1000 == 0:
            logger.info(
                "Null distribution: %d / %d scores computed.", i + 1, n_sims,
            )

    null_scores = np.array(scores, dtype=np.float64)
    logger.info(
        "Null distribution: %d scores (median=%.4f, std=%.4f).",
        len(null_scores), float(np.median(null_scores)),
        float(np.std(null_scores)),
    )
    return null_scores


# ====================================================================
# Domain-shift diagnostic (KS test)
# ====================================================================

def ks_domain_shift_test(
    sim_scores: np.ndarray,
    real_scores: np.ndarray,
) -> tuple[float, float]:
    """Kolmogorov--Smirnov two-sample test for domain shift.

    Compares the distributions of network scores (or pixel values)
    between simulations and real data.  A statistically significant
    difference (p < 0.05) indicates a domain shift that must be
    investigated.

    Parameters
    ----------
    sim_scores : np.ndarray
        Score (or pixel value) distribution from simulations.
    real_scores : np.ndarray
        Score (or pixel value) distribution from real data.

    Returns
    -------
    d_ks : float
        KS statistic (maximum absolute CDF difference).
    p_value : float
        Two-sided p-value.
    """
    sim = np.asarray(sim_scores, dtype=np.float64).ravel()
    real = np.asarray(real_scores, dtype=np.float64).ravel()

    d_ks, p_value = stats.ks_2samp(sim, real)
    logger.info(
        "KS domain-shift test: D=%.4f, p=%.4e (n_sim=%d, n_real=%d).",
        d_ks, p_value, len(sim), len(real),
    )
    return float(d_ks), float(p_value)


# ====================================================================
# Cross-method consistency test
# ====================================================================

def run_consistency_checks(
    results_dict: dict[str, float],
    expected_scatter: Optional[float] = None,
) -> tuple[float, bool, float]:
    """Test consistency of test statistics across component-separation methods.

    Implements the chi^2 consistency test from the Statistical Framework
    Section 7.1:

        chi^2 = sum_k (Lambda_k - Lambda_bar)^2 / sigma^2

    Under the null hypothesis of consistency (same CMB + different noise),
    chi^2 follows a chi-squared distribution with ``K - 1`` degrees of
    freedom.

    Parameters
    ----------
    results_dict : dict
        Mapping from method name to test-statistic value, e.g.
        ``{"SMICA": 0.72, "Commander": 0.68, "NILC": 0.70, "SEVEM": 0.71}``.
    expected_scatter : float or None
        Expected per-method scatter ``sigma_Lambda`` from simulations.
        If ``None``, it is estimated as the sample standard deviation
        of the input scores (less conservative but useful when no
        simulation estimate is available).

    Returns
    -------
    chi2 : float
        Consistency chi-squared.
    passed : bool
        ``True`` if the p-value exceeds 0.05 (i.e. chi^2 is consistent
        with the null hypothesis of no inter-method discrepancy).
    p_value : float
        p-value of the chi^2 statistic.
    """
    scores = np.array(list(results_dict.values()), dtype=np.float64)
    k = len(scores)
    if k < 2:
        logger.warning(
            "Cannot run consistency test with fewer than 2 methods."
        )
        return 0.0, True, 1.0

    mean_score = float(np.mean(scores))

    if expected_scatter is not None and expected_scatter > 0:
        sigma = expected_scatter
    else:
        sigma = float(np.std(scores, ddof=1))
        if sigma <= 0:
            logger.info("All method scores identical; trivially consistent.")
            return 0.0, True, 1.0

    chi2 = float(np.sum((scores - mean_score) ** 2) / sigma ** 2)
    ndof = k - 1
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=ndof))
    passed = p_value > 0.05

    logger.info(
        "Consistency test: chi2=%.2f (ndof=%d), p=%.4f -> %s",
        chi2, ndof, p_value, "PASS" if passed else "FAIL",
    )
    for name, val in results_dict.items():
        logger.info("  %s: %.4f", name, val)

    return chi2, passed, p_value
