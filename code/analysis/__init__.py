"""
Analysis modules for bubble collision detection in the CMB.

Modules
-------
power_spectrum
    Angular power spectrum estimation and spectral comparison.
anomaly_detection
    Detection statistics: p-values, Z-scores, FDR control,
    domain-shift diagnostics, and consistency tests.
bubble_search
    Full search pipeline: detection efficiency, sensitivity curves,
    upper limits, and end-to-end orchestration.
bayesian_evidence
    Bayesian model comparison via nested sampling (dynesty):
    Bayes factor, posteriors, and Jeffreys-scale interpretation.
"""

from .power_spectrum import (
    cl_to_dl,
    compare_power_spectra,
    compute_cl_ratio,
    compute_mean_cl_from_sims,
    estimate_cl,
)
from .anomaly_detection import (
    benjamini_hochberg,
    compute_global_pvalue,
    compute_null_distribution,
    compute_pvalue,
    ks_domain_shift_test,
    run_consistency_checks,
)
from .bubble_search import (
    DetectionResult,
    SensitivityResult,
    UpperLimitResult,
    compute_detection_efficiency,
    compute_sensitivity_curve,
    compute_upper_limits,
    run_full_search,
)
from .bayesian_evidence import (
    EvidenceResult,
    approximate_bayes_factor,
    bubble_prior_transform,
    compute_bayes_factor,
    interpret_bayes_factor,
    pixel_log_likelihood,
)

__all__ = [
    # power_spectrum
    "estimate_cl",
    "compute_cl_ratio",
    "cl_to_dl",
    "compare_power_spectra",
    "compute_mean_cl_from_sims",
    # anomaly_detection
    "compute_pvalue",
    "compute_global_pvalue",
    "benjamini_hochberg",
    "compute_null_distribution",
    "ks_domain_shift_test",
    "run_consistency_checks",
    # bubble_search
    "DetectionResult",
    "SensitivityResult",
    "UpperLimitResult",
    "compute_detection_efficiency",
    "compute_upper_limits",
    "compute_sensitivity_curve",
    "run_full_search",
    # bayesian_evidence
    "EvidenceResult",
    "bubble_prior_transform",
    "pixel_log_likelihood",
    "compute_bayes_factor",
    "interpret_bayes_factor",
    "approximate_bayes_factor",
]
