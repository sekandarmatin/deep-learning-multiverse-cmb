"""
Inference pipeline for trained DeepSphere ensembles.

Provides:

- Loading and preprocessing of real Planck FITS maps
- Ensemble-averaged predictions with epistemic uncertainty
- MC-Dropout uncertainty estimation
- Grad-CAM attribution maps for physical interpretability
- Full inference on all four Planck component-separated maps

References
----------
- Architecture document, Section 2.7
- Theoretical framework, Section 4.5 (Eqs. 4.9--4.10)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .spherical_cnn import DeepSphere, build_all_graphs

logger = logging.getLogger(__name__)


# ===================================================================
# Result dataclass
# ===================================================================

@dataclass
class InferenceResult:
    """Result from inference on a single map.

    Attributes
    ----------
    logit_mean : float
        Ensemble mean of raw logits.
    logit_std : float
        Ensemble standard deviation of logits.
    p_coll_mean : float
        Ensemble mean calibrated probability of collision.
    p_coll_std : float
        Ensemble standard deviation of calibrated probabilities.
    p_coll_per_model : np.ndarray
        Individual model probabilities, shape ``(N_ensemble,)``.
    mc_dropout_mean : float
        MC-Dropout mean probability (from first ensemble member).
    mc_dropout_std : float
        MC-Dropout standard deviation.
    regression_params : np.ndarray or None
        Estimated collision parameters ``(theta_c, phi_c, theta_r, A, kappa)``
        if regression head is enabled, otherwise ``None``.
    """

    logit_mean: float = 0.0
    logit_std: float = 0.0
    p_coll_mean: float = 0.0
    p_coll_std: float = 0.0
    p_coll_per_model: np.ndarray = field(default_factory=lambda: np.array([]))
    mc_dropout_mean: float = 0.0
    mc_dropout_std: float = 0.0
    regression_params: Optional[np.ndarray] = None


# ===================================================================
# Planck map I/O
# ===================================================================

def load_planck_map(
    map_path: str | Path,
    nside_target: int = 64,
    field: int = 0,
) -> np.ndarray:
    """Load a Planck component-separated map and degrade to target Nside.

    The map is read and kept in RING ordering, matching the training
    pipeline (simulations use healpy synfast which outputs RING).

    Parameters
    ----------
    map_path : str or Path
        Path to the Planck FITS file.
    nside_target : int
        Target HEALPix resolution for degrading.
    field : int
        FITS field to read (0 = temperature, 1 = Q, 2 = U).

    Returns
    -------
    planck_map : np.ndarray, shape ``(12 * nside_target**2,)``, dtype float32
        Map in micro-Kelvin at target resolution, nested ordering.
    """
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Planck map not found: {map_path}")

    logger.info("Loading Planck map: %s (field=%d)", map_path, field)

    # Read map (RING ordering by default)
    raw_map = hp.read_map(str(map_path), field=field, dtype=np.float64)
    nside_in = hp.npix2nside(len(raw_map))

    # Degrade resolution if needed
    if nside_in != nside_target:
        logger.info(
            "Degrading Nside %d -> %d", nside_in, nside_target,
        )
        raw_map = hp.ud_grade(raw_map, nside_out=nside_target, order_in="RING")

    # Keep RING ordering (matches training pipeline)

    # Convert to microKelvin (Planck maps are typically in Kelvin)
    # Check if values are O(1e-5) => Kelvin; O(1) => already microK
    if np.nanstd(raw_map[raw_map != hp.UNSEEN]) < 1e-2:
        raw_map *= 1e6  # K -> uK
        logger.info("Converted Planck map from K to uK")

    return raw_map.astype(np.float32)


def preprocess_planck_map(
    planck_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    remove_monopole_dipole: bool = True,
) -> torch.Tensor:
    """Preprocess a Planck map for model input.

    Applies mask, removes monopole/dipole, standardises to zero mean
    and unit variance over unmasked pixels, and reshapes to
    ``(1, 1, N_pix)`` tensor.

    Parameters
    ----------
    planck_map : np.ndarray
        Input map in micro-Kelvin, shape ``(N_pix,)``.
    mask : np.ndarray or None
        Binary mask (1 = valid, 0 = masked). If ``None``, no masking.
    remove_monopole_dipole : bool
        Whether to remove the monopole (l=0) and dipole (l=1) from
        unmasked pixels.

    Returns
    -------
    tensor : torch.Tensor, shape ``(1, 1, N_pix)``
    """
    result = planck_map.copy().astype(np.float64)

    if mask is not None:
        valid = mask > 0.5
    else:
        valid = np.ones(len(result), dtype=bool)

    # Replace UNSEEN pixels
    unseen = np.abs(result - hp.UNSEEN) < 1.0
    result[unseen] = 0.0
    valid = valid & (~unseen)

    # Remove monopole and dipole from unmasked pixels
    if remove_monopole_dipole:
        nside = hp.npix2nside(len(result))
        # Map is already in RING ordering (matching training pipeline)
        result = hp.remove_monopole(result, gal_cut=0, fitval=False)
        result = hp.remove_dipole(result, gal_cut=0, fitval=False)

    # Standardise (zero mean, unit variance) over valid pixels
    mu = np.mean(result[valid])
    sigma = np.std(result[valid])
    if sigma < 1e-12:
        sigma = 1.0
    result = (result - mu) / sigma

    # Zero out masked pixels
    if mask is not None:
        result[~valid] = 0.0

    tensor = torch.tensor(result, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


# ===================================================================
# Ensemble inference
# ===================================================================

class EnsembleInference:
    """Inference engine for a deep ensemble of DeepSphere models.

    Parameters
    ----------
    model_paths : list of str or Path
        Paths to the N_ensemble model checkpoints (``.pt`` files).
    calibration_path : str or Path or None
        Path to a pickle file containing calibration parameters.
        Expected keys: ``"method"``, ``"params"``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    nside : int
        HEALPix Nside the models were trained on.
    K : int
        Chebyshev filter order used during training.
    channels : list of int or None
        Channel dimensions used during training.
    enable_regression : bool
        Whether the models have a regression head.
    """

    def __init__(
        self,
        model_paths: list[str | Path],
        calibration_path: Optional[str | Path] = None,
        device: str = "cuda",
        nside: int = 64,
        K: int = 20,
        channels: Optional[list[int]] = None,
        enable_regression: bool = False,
    ) -> None:
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.nside = nside
        self.n_ensemble = len(model_paths)

        # Build shared graphs once
        graphs = build_all_graphs()

        # Load models
        self.models: list[DeepSphere] = []
        for path in model_paths:
            model = DeepSphere(
                nside=nside,
                K=K,
                channels=channels,
                graphs=graphs,
                enable_regression=enable_regression,
            )
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=False,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        logger.info(
            "Loaded ensemble of %d models on %s", self.n_ensemble, self.device,
        )

        # Load calibration parameters
        self.calibration_method: Optional[str] = None
        self.calibration_params: Optional[dict] = None
        if calibration_path is not None:
            cal_path = Path(calibration_path)
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    cal = pickle.load(f)
                self.calibration_method = cal.get("method")
                self.calibration_params = cal.get("params")
                logger.info(
                    "Loaded calibration: method=%s", self.calibration_method,
                )

    def _apply_calibration(self, logit: float) -> float:
        """Apply calibration to a single logit to produce a probability.

        Parameters
        ----------
        logit : float
            Raw model output (pre-sigmoid).

        Returns
        -------
        prob : float
            Calibrated probability.
        """
        if self.calibration_method is None or self.calibration_params is None:
            # No calibration: plain sigmoid
            return float(1.0 / (1.0 + np.exp(-logit)))

        method = self.calibration_method
        params = self.calibration_params

        if method == "temperature":
            T = params["T"]
            return float(1.0 / (1.0 + np.exp(-logit / T)))
        elif method == "platt":
            a, b = params["a"], params["b"]
            return float(1.0 / (1.0 + np.exp(-(a * logit + b))))
        elif method == "isotonic":
            # Isotonic regression calibrator stored as sklearn object
            calibrator = params["calibrator"]
            prob_uncal = float(1.0 / (1.0 + np.exp(-logit)))
            return float(calibrator.predict([prob_uncal])[0])
        else:
            return float(1.0 / (1.0 + np.exp(-logit)))

    @torch.no_grad()
    def predict(self, map_tensor: torch.Tensor) -> InferenceResult:
        """Run ensemble inference on a single preprocessed map.

        Parameters
        ----------
        map_tensor : torch.Tensor, shape ``(1, 1, N_pix)``
            Preprocessed CMB map.

        Returns
        -------
        result : InferenceResult
        """
        map_tensor = map_tensor.to(self.device)

        logits: list[float] = []
        probs: list[float] = []
        reg_params_list: list[np.ndarray] = []

        for model in self.models:
            model.eval()
            output = model(map_tensor)
            if isinstance(output, tuple):
                logit_tensor, param_tensor = output
                reg_params_list.append(param_tensor.cpu().numpy().squeeze())
            else:
                logit_tensor = output

            logit_val = float(logit_tensor.squeeze().cpu().numpy())
            logits.append(logit_val)
            probs.append(self._apply_calibration(logit_val))

        logit_arr = np.array(logits)
        prob_arr = np.array(probs)

        result = InferenceResult(
            logit_mean=float(logit_arr.mean()),
            logit_std=float(logit_arr.std()),
            p_coll_mean=float(prob_arr.mean()),
            p_coll_std=float(prob_arr.std()),
            p_coll_per_model=prob_arr,
        )

        if reg_params_list:
            result.regression_params = np.mean(reg_params_list, axis=0)

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        data_loader: DataLoader,
    ) -> list[InferenceResult]:
        """Run inference on a full DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            Each batch yields ``(maps, labels, ...)`` or ``(maps, labels)``.

        Returns
        -------
        results : list of InferenceResult
            One per map in the dataset.
        """
        results: list[InferenceResult] = []
        for batch in data_loader:
            if len(batch) >= 2:
                maps = batch[0]
            else:
                maps = batch

            maps = maps.to(self.device)
            for i in range(maps.size(0)):
                single = maps[i : i + 1]
                result = self.predict(single)
                results.append(result)

        logger.info("Batch inference complete: %d maps", len(results))
        return results

    def predict_with_mc_dropout(
        self,
        map_tensor: torch.Tensor,
        n_passes: int = 50,
        model_index: int = 0,
    ) -> tuple[float, float]:
        """MC-Dropout uncertainty estimation.

        Enables dropout at inference time and performs multiple stochastic
        forward passes through a single ensemble member.

        Parameters
        ----------
        map_tensor : torch.Tensor, shape ``(1, 1, N_pix)``
            Preprocessed CMB map.
        n_passes : int
            Number of stochastic forward passes.
        model_index : int
            Which ensemble member to use (default: first).

        Returns
        -------
        mean_p : float
            Mean predicted probability across passes.
        std_p : float
            Standard deviation of predicted probabilities.
        """
        map_tensor = map_tensor.to(self.device)
        model = self.models[model_index]

        # Enable dropout during inference
        _enable_mc_dropout(model)

        probs: list[float] = []
        with torch.no_grad():
            for _ in range(n_passes):
                output = model(map_tensor)
                if isinstance(output, tuple):
                    logit = output[0]
                else:
                    logit = output
                logit_val = float(logit.squeeze().cpu().numpy())
                probs.append(self._apply_calibration(logit_val))

        # Restore eval mode (dropout off)
        model.eval()

        prob_arr = np.array(probs)
        return float(prob_arr.mean()), float(prob_arr.std())

    def compute_gradcam(
        self,
        map_tensor: torch.Tensor,
        layer_idx: int = -1,
        model_index: int = 0,
    ) -> np.ndarray:
        """Compute Grad-CAM attribution map.

        Implements Eq. (4.10) from the theoretical framework:
            ``M_GradCAM(n) = ReLU( sum_k alpha_k * A^k(n) )``
        where ``alpha_k = (1/N_pix) * sum_n dS/dA^k(n)`` and ``S`` is
        the classification score.

        Parameters
        ----------
        map_tensor : torch.Tensor, shape ``(1, 1, N_pix)``
            Preprocessed CMB map.
        layer_idx : int
            Index of the convolutional layer to use (``-1`` = last).
        model_index : int
            Which ensemble member to use.

        Returns
        -------
        attribution : np.ndarray, shape ``(12 * nside**2,)``
            Attribution weights per pixel, upsampled to full Nside=64
            resolution if the selected layer operates at a coarser scale.
        """
        map_tensor = map_tensor.to(self.device).requires_grad_(False)
        model = self.models[model_index]
        model.eval()

        # Enable gradients for feature maps
        map_input = map_tensor.detach().clone().requires_grad_(True)

        # Get feature maps (with gradient tracking)
        features = model.get_feature_maps(map_input)
        target_features = features[layer_idx]
        target_features.retain_grad()

        # Forward pass through GAP and FC (need to rebuild from features)
        # Use the last feature map to get the final logit
        gap = target_features.mean(dim=-1)  # (1, C)

        # Pass through FC layers
        fc_out = model.fc(gap)
        logit = model.classifier(fc_out)

        # Backward pass
        model.zero_grad()
        logit.backward()

        # Grad-CAM weights: global average of gradients
        gradients = target_features.grad  # (1, C, N_pix_layer)
        if gradients is None:
            logger.warning("No gradients computed; returning zeros.")
            npix = 12 * self.nside * self.nside
            return np.zeros(npix, dtype=np.float32)

        alpha_k = gradients.mean(dim=-1, keepdim=True)  # (1, C, 1)

        # Weighted combination of feature maps
        cam = (alpha_k * target_features).sum(dim=1).squeeze(0)  # (N_pix_layer,)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        # Normalise
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Upsample to full resolution if needed
        npix_full = 12 * self.nside * self.nside
        if len(cam) < npix_full:
            nside_layer = hp.npix2nside(len(cam))
            # Use HEALPix ud_grade for upsampling (nested ordering)
            cam_full = hp.ud_grade(cam, nside_out=self.nside, order_in="RING")
            cam = cam_full

        return cam.astype(np.float32)


# ===================================================================
# Helpers
# ===================================================================

def _enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout layers to train mode for MC-Dropout.

    Parameters
    ----------
    model : nn.Module
        Model whose dropout layers will be activated.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def run_full_inference(
    config: dict[str, Any],
    planck_maps: dict[str, str],
    ensemble: EnsembleInference,
    mask_path: Optional[str] = None,
    nside_target: int = 64,
    mc_dropout_passes: int = 50,
) -> dict[str, InferenceResult]:
    """Run inference on all four Planck component-separated maps.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    planck_maps : dict
        Mapping from method name to FITS file path, e.g.
        ``{"SMICA": "data/planck/smica.fits", ...}``.
    ensemble : EnsembleInference
        Trained ensemble inference engine.
    mask_path : str or None
        Path to the Planck Galactic mask FITS file.
    nside_target : int
        Target HEALPix resolution.
    mc_dropout_passes : int
        Number of MC-Dropout forward passes.

    Returns
    -------
    results : dict
        Mapping from method name to ``InferenceResult``.
    """
    # Load mask if provided
    mask: Optional[np.ndarray] = None
    if mask_path is not None:
        mask_raw = hp.read_map(mask_path, dtype=np.float64)
        mask_nside = hp.npix2nside(len(mask_raw))
        if mask_nside != nside_target:
            mask_raw = hp.ud_grade(mask_raw, nside_out=nside_target, order_in="RING")
        # Keep RING ordering (matches training pipeline)
        mask = (mask_raw > 0.5).astype(np.float64)

    results: dict[str, InferenceResult] = {}

    for method_name, fits_path in planck_maps.items():
        logger.info("Running inference on %s ...", method_name)

        # Load and preprocess
        planck_map = load_planck_map(fits_path, nside_target=nside_target)
        map_tensor = preprocess_planck_map(
            planck_map, mask=mask, remove_monopole_dipole=True,
        )

        # Ensemble prediction
        result = ensemble.predict(map_tensor)

        # MC-Dropout uncertainty
        mc_mean, mc_std = ensemble.predict_with_mc_dropout(
            map_tensor, n_passes=mc_dropout_passes,
        )
        result.mc_dropout_mean = mc_mean
        result.mc_dropout_std = mc_std

        results[method_name] = result

        logger.info(
            "%s: p_coll = %.4f +/- %.4f (ensemble), "
            "%.4f +/- %.4f (MC-Dropout)",
            method_name,
            result.p_coll_mean, result.p_coll_std,
            mc_mean, mc_std,
        )

    return results
