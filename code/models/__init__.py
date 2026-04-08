"""
Deep learning models for bubble collision detection in CMB maps.

Modules
-------
spherical_cnn
    DeepSphere graph-based spherical CNN architecture.
training
    Training pipeline with ensemble support.
inference
    Inference pipeline for Planck data and simulations.
calibration
    Post-hoc probability calibration.
"""

from .spherical_cnn import (
    DeepSphere,
    HealpixPool,
    SphericalGraphConvBlock,
    build_all_graphs,
    build_healpix_graph,
)
from .training import (
    TrainingConfig,
    TrainingResult,
    compute_overfitting_gap,
    evaluate_model,
    train_ensemble,
    train_single_model,
)
from .inference import (
    EnsembleInference,
    InferenceResult,
    load_planck_map,
    preprocess_planck_map,
    run_full_inference,
)
from .calibration import (
    CalibrationResult,
    apply_calibration,
    calibrate_pipeline,
    compute_ece,
    compute_ece_mce,
    compute_mce,
    isotonic_calibration,
    platt_scaling,
    temperature_scaling,
)

__all__ = [
    # spherical_cnn
    "DeepSphere",
    "HealpixPool",
    "SphericalGraphConvBlock",
    "build_all_graphs",
    "build_healpix_graph",
    # training
    "TrainingConfig",
    "TrainingResult",
    "compute_overfitting_gap",
    "evaluate_model",
    "train_ensemble",
    "train_single_model",
    # inference
    "EnsembleInference",
    "InferenceResult",
    "load_planck_map",
    "preprocess_planck_map",
    "run_full_inference",
    # calibration
    "CalibrationResult",
    "apply_calibration",
    "calibrate_pipeline",
    "compute_ece",
    "compute_ece_mce",
    "compute_mce",
    "isotonic_calibration",
    "platt_scaling",
    "temperature_scaling",
]
