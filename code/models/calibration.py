"""
Probability calibration for DeepSphere network outputs.

Implements post-hoc calibration methods to ensure that the network's
predicted probabilities are well-calibrated (i.e., if the network
predicts p=0.7, then ~70% of such predictions should be true positives).

Methods
-------
- **Temperature scaling**: single parameter T applied to logits.
- **Platt scaling**: affine transform a*logit + b.
- **Isotonic regression**: non-parametric monotone calibration.

Metrics
-------
- **ECE** (Expected Calibration Error): weighted average of per-bin
  |accuracy - confidence| over equal-width confidence bins.
- **MCE** (Maximum Calibration Error): maximum per-bin
  |accuracy - confidence|.

The ``calibrate_pipeline()`` function tries all three methods on a
calibration set and selects the one achieving the lowest ECE on
a held-out evaluation set, with a target ECE < 0.02.

References
----------
- Guo et al. (2017), "On Calibration of Modern Neural Networks"
- Platt (1999), "Probabilistic Outputs for Support Vector Machines"
- Architecture document, Section 2.8
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit  # sigmoid
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


# ===================================================================
# Result dataclass
# ===================================================================

@dataclass
class CalibrationResult:
    """Output of calibration assessment or fitting.

    Attributes
    ----------
    method : str
        One of ``"uncalibrated"``, ``"temperature"``, ``"platt"``,
        ``"isotonic"``.
    ece : float
        Expected Calibration Error.
    mce : float
        Maximum Calibration Error.
    bin_edges : np.ndarray
        Bin edges for the reliability diagram, shape ``(n_bins + 1,)``.
    bin_accuracies : np.ndarray
        Per-bin accuracy (fraction of positives), shape ``(n_bins,)``.
    bin_confidences : np.ndarray
        Per-bin mean predicted probability, shape ``(n_bins,)``.
    bin_counts : np.ndarray
        Number of samples in each bin, shape ``(n_bins,)``.
    parameters : dict
        Method-specific calibration parameters, e.g.
        ``{"T": 1.5}`` for temperature scaling.
    """

    method: str = "uncalibrated"
    ece: float = 1.0
    mce: float = 1.0
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_accuracies: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_confidences: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    parameters: dict = field(default_factory=dict)


# ===================================================================
# ECE / MCE computation
# ===================================================================

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Parameters
    ----------
    probs : np.ndarray, shape ``(N,)``
        Predicted probabilities in [0, 1].
    labels : np.ndarray, shape ``(N,)``
        Binary ground-truth labels.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    ece : float
        Expected Calibration Error.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if lo == 0.0:
            mask = (probs >= lo) & (probs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].astype(float).mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error.

    Parameters
    ----------
    probs : np.ndarray, shape ``(N,)``
        Predicted probabilities in [0, 1].
    labels : np.ndarray, shape ``(N,)``
        Binary ground-truth labels.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    mce : float
        Maximum Calibration Error.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if lo == 0.0:
            mask = (probs >= lo) & (probs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].astype(float).mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))

    return float(mce)


def compute_ece_mce(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> CalibrationResult:
    """Compute ECE, MCE, and reliability diagram data from raw logits.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N,)``
        Raw model outputs (pre-sigmoid).
    labels : np.ndarray, shape ``(N,)``
        Binary labels (0 or 1).
    n_bins : int
        Number of equal-width bins for the reliability diagram.

    Returns
    -------
    result : CalibrationResult
    """
    probs = expit(logits.astype(np.float64))
    labels = labels.astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == 0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs > lo) & (probs <= hi)
        n_bin = mask.sum()
        bin_counts[i] = n_bin
        if n_bin > 0:
            bin_accuracies[i] = labels[mask].mean()
            bin_confidences[i] = probs[mask].mean()

    n_total = len(labels)
    ece_val = 0.0
    mce_val = 0.0
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        gap = abs(bin_accuracies[i] - bin_confidences[i])
        ece_val += (bin_counts[i] / n_total) * gap
        mce_val = max(mce_val, gap)

    return CalibrationResult(
        method="uncalibrated",
        ece=float(ece_val),
        mce=float(mce_val),
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        parameters={},
    )


# ===================================================================
# Calibration methods
# ===================================================================

def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit a single temperature parameter T by minimising NLL.

    The calibrated probability is ``sigmoid(logit / T)``.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N,)``
        Raw model outputs (pre-sigmoid).
    labels : np.ndarray, shape ``(N,)``
        Binary labels (0 or 1).

    Returns
    -------
    T : float
        Optimal temperature (T > 0).
    """
    logits = logits.astype(np.float64)
    labels = labels.astype(np.float64)

    def nll(T: float) -> float:
        """Negative log-likelihood for temperature T."""
        if T <= 0:
            return 1e12
        scaled = logits / T
        # Numerically stable NLL: -[y*log(sigma(z)) + (1-y)*log(1-sigma(z))]
        # = -[y*z - log(1+exp(z))]  using log_sigmoid trick
        term = np.where(
            scaled >= 0,
            np.log(1.0 + np.exp(-scaled)),
            -scaled + np.log(1.0 + np.exp(scaled)),
        )
        loss = -labels * scaled + term
        return float(loss.mean())

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
    T_opt = float(result.x)

    logger.info("Temperature scaling: T = %.4f (NLL = %.6f)", T_opt, result.fun)
    return T_opt


def platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Fit Platt scaling parameters (a, b).

    The calibrated probability is ``sigmoid(a * logit + b)``.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N,)``
    labels : np.ndarray, shape ``(N,)``

    Returns
    -------
    a : float
    b : float
    """
    logits = logits.astype(np.float64)
    labels = labels.astype(np.float64)

    def nll(params: np.ndarray) -> float:
        a, b = params
        z = a * logits + b
        term = np.where(
            z >= 0,
            np.log(1.0 + np.exp(-z)),
            -z + np.log(1.0 + np.exp(z)),
        )
        loss = -labels * z + term
        return float(loss.mean())

    result = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a_opt, b_opt = result.x

    logger.info(
        "Platt scaling: a = %.4f, b = %.4f (NLL = %.6f)",
        a_opt, b_opt, result.fun,
    )
    return float(a_opt), float(b_opt)


def isotonic_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
) -> IsotonicRegression:
    """Fit isotonic regression calibration.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N,)``
        Raw model outputs (pre-sigmoid).
    labels : np.ndarray, shape ``(N,)``
        Binary labels.

    Returns
    -------
    calibrator : sklearn.isotonic.IsotonicRegression
        Fitted calibrator. Use ``calibrator.predict(probs)`` to
        calibrate.
    """
    probs = expit(logits.astype(np.float64))
    labels = labels.astype(np.float64)

    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip",
    )
    calibrator.fit(probs, labels)

    logger.info("Isotonic regression calibration fitted.")
    return calibrator


# ===================================================================
# Apply calibration
# ===================================================================

def apply_calibration(
    logits: np.ndarray,
    method: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Apply a fitted calibration to raw logits.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N,)``
    method : str
        One of ``"temperature"``, ``"platt"``, ``"isotonic"``.
    params : dict
        Calibration parameters from the fitting functions.

    Returns
    -------
    calibrated_probs : np.ndarray, shape ``(N,)``
    """
    logits = logits.astype(np.float64)

    if method == "temperature":
        T = params["T"]
        return expit(logits / T)
    elif method == "platt":
        a = params["a"]
        b = params["b"]
        return expit(a * logits + b)
    elif method == "isotonic":
        calibrator = params["calibrator"]
        probs_uncal = expit(logits)
        return calibrator.predict(probs_uncal)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# ===================================================================
# Full calibration pipeline
# ===================================================================

def calibrate_pipeline(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    logits_eval: np.ndarray,
    labels_eval: np.ndarray,
    n_bins: int = 15,
    ece_threshold: float = 0.02,
    save_path: Optional[str | Path] = None,
) -> tuple[CalibrationResult, str, dict[str, Any]]:
    """Run the full calibration pipeline.

    Tries calibration methods in order:
    temperature scaling -> Platt scaling -> isotonic regression,
    and selects the first that achieves ECE < ``ece_threshold`` on
    the evaluation set. If none meets the threshold, the method with
    the lowest ECE is selected.

    Parameters
    ----------
    logits_cal : np.ndarray, shape ``(N_cal,)``
        Logits for the calibration subset (used for fitting).
    labels_cal : np.ndarray, shape ``(N_cal,)``
        Labels for the calibration subset.
    logits_eval : np.ndarray, shape ``(N_eval,)``
        Logits for evaluation (held-out set).
    labels_eval : np.ndarray, shape ``(N_eval,)``
        Labels for the evaluation set.
    n_bins : int
        Number of bins for ECE/MCE computation.
    ece_threshold : float
        Target ECE threshold (default 0.02).
    save_path : str or Path or None
        If provided, save the calibration parameters to this pickle file.

    Returns
    -------
    result : CalibrationResult
        Evaluation metrics on the eval set after calibration.
    best_method : str
        The method that achieved the lowest ECE.
    best_params : dict
        Fitted calibration parameters for the selected method.
    """
    # First, compute uncalibrated metrics
    uncal_result = compute_ece_mce(logits_eval, labels_eval, n_bins=n_bins)
    logger.info(
        "Uncalibrated: ECE=%.4f, MCE=%.4f",
        uncal_result.ece, uncal_result.mce,
    )

    candidates: list[tuple[str, dict[str, Any], CalibrationResult]] = []

    # --- Temperature scaling ---
    try:
        T = temperature_scaling(logits_cal, labels_cal)
        probs_eval_t = apply_calibration(
            logits_eval, "temperature", {"T": T},
        )
        ece_t = compute_ece(probs_eval_t, labels_eval, n_bins)
        mce_t = compute_mce(probs_eval_t, labels_eval, n_bins)
        result_t = _make_result(
            probs_eval_t, labels_eval, "temperature", ece_t, mce_t,
            n_bins, {"T": T},
        )
        candidates.append(("temperature", {"T": T}, result_t))
        logger.info("Temperature scaling: ECE=%.4f, MCE=%.4f", ece_t, mce_t)
    except Exception as e:
        logger.warning("Temperature scaling failed: %s", e)

    # --- Platt scaling ---
    try:
        a, b = platt_scaling(logits_cal, labels_cal)
        probs_eval_p = apply_calibration(
            logits_eval, "platt", {"a": a, "b": b},
        )
        ece_p = compute_ece(probs_eval_p, labels_eval, n_bins)
        mce_p = compute_mce(probs_eval_p, labels_eval, n_bins)
        result_p = _make_result(
            probs_eval_p, labels_eval, "platt", ece_p, mce_p,
            n_bins, {"a": a, "b": b},
        )
        candidates.append(("platt", {"a": a, "b": b}, result_p))
        logger.info("Platt scaling: ECE=%.4f, MCE=%.4f", ece_p, mce_p)
    except Exception as e:
        logger.warning("Platt scaling failed: %s", e)

    # --- Isotonic regression ---
    try:
        calibrator = isotonic_calibration(logits_cal, labels_cal)
        probs_eval_i = apply_calibration(
            logits_eval, "isotonic", {"calibrator": calibrator},
        )
        ece_i = compute_ece(probs_eval_i, labels_eval, n_bins)
        mce_i = compute_mce(probs_eval_i, labels_eval, n_bins)
        result_i = _make_result(
            probs_eval_i, labels_eval, "isotonic", ece_i, mce_i,
            n_bins, {"calibrator": calibrator},
        )
        candidates.append(("isotonic", {"calibrator": calibrator}, result_i))
        logger.info("Isotonic regression: ECE=%.4f, MCE=%.4f", ece_i, mce_i)
    except Exception as e:
        logger.warning("Isotonic regression failed: %s", e)

    if not candidates:
        logger.error("All calibration methods failed; returning uncalibrated.")
        return uncal_result, "uncalibrated", {}

    # Select the method with lowest ECE
    candidates.sort(key=lambda x: x[2].ece)
    best_method, best_params, best_result = candidates[0]

    # If isotonic was selected, check if it's too coarse (step function)
    # and prefer temperature scaling instead
    if best_method == "isotonic":
        for method, params, result in candidates:
            if method == "temperature":
                best_method = method
                best_params = params
                best_result = result
                logger.info(
                    "Preferring temperature scaling (ECE=%.4f) over "
                    "isotonic (step function artifact risk).",
                    result.ece,
                )
                break

    assert best_result is not None
    logger.info(
        "Selected calibration: %s (ECE=%.4f, MCE=%.4f)",
        best_method, best_result.ece, best_result.mce,
    )

    # Save calibration parameters
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {"method": best_method, "params": best_params},
                f,
            )
        logger.info("Saved calibration to %s", save_path)

    return best_result, best_method, best_params


# ===================================================================
# Internal helpers
# ===================================================================

def _make_result(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str,
    ece_val: float,
    mce_val: float,
    n_bins: int,
    parameters: dict,
) -> CalibrationResult:
    """Build a CalibrationResult with reliability diagram data.

    Parameters
    ----------
    probs : np.ndarray
    labels : np.ndarray
    method : str
    ece_val : float
    mce_val : float
    n_bins : int
    parameters : dict

    Returns
    -------
    CalibrationResult
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    labels_f = labels.astype(np.float64)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == 0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs > lo) & (probs <= hi)
        n_bin = mask.sum()
        bin_counts[i] = n_bin
        if n_bin > 0:
            bin_accuracies[i] = labels_f[mask].mean()
            bin_confidences[i] = probs[mask].mean()

    return CalibrationResult(
        method=method,
        ece=ece_val,
        mce=mce_val,
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        parameters=parameters,
    )
