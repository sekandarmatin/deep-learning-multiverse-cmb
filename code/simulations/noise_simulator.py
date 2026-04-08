"""
Planck-like instrumental noise simulation.

Supports two noise models:

1. **Isotropic white noise** -- independent Gaussian draws per pixel with
   a configurable RMS level (default 45 muK at Nside = 64).
2. **FFP10 anisotropic noise** -- pre-computed Planck Full Focal Plane
   (version 10) noise realisations, degraded to the target resolution.

References
----------
- Planck Collaboration (2020), A&A 641, A3 (FFP10 simulations).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

# Default noise parameters
DEFAULT_SIGMA_UK: float = 45.0   # muK per pixel at Nside=64
DEFAULT_NSIDE: int = 64
DEFAULT_FFP10_DIR: str = "data/planck/ffp10/"
N_FFP10_REALIZATIONS: int = 300


def generate_white_noise(
    nside: int = DEFAULT_NSIDE,
    sigma_uK: float = DEFAULT_SIGMA_UK,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an isotropic white-noise HEALPix map.

    Each pixel is drawn independently from N(0, sigma_uK^2).

    Parameters
    ----------
    nside : int
        HEALPix Nside resolution parameter.
    sigma_uK : float
        RMS noise level in microKelvin per pixel.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noise_map : np.ndarray, shape (12 * nside**2,), dtype float32
        White-noise map in microKelvin.
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    noise_map = rng.normal(loc=0.0, scale=sigma_uK, size=npix)
    return noise_map.astype(np.float32)


def load_ffp10_noise(
    realization_index: int,
    nside_target: int = DEFAULT_NSIDE,
    ffp10_dir: Union[str, Path] = DEFAULT_FFP10_DIR,
) -> np.ndarray:
    """Load and degrade an FFP10 noise realisation to the target Nside.

    Expects FITS files following the Planck Legacy Archive naming
    convention.  If the file is at a higher resolution than *nside_target*,
    it is degraded using ``healpy.ud_grade``.

    Parameters
    ----------
    realization_index : int
        Index of the FFP10 noise realisation (0 to 299).
    nside_target : int
        Target HEALPix resolution for degrading.
    ffp10_dir : str or Path
        Directory containing FFP10 noise FITS files.

    Returns
    -------
    noise_map : np.ndarray, shape (12 * nside_target**2,), dtype float32
        Degraded noise map in microKelvin.

    Raises
    ------
    FileNotFoundError
        If the requested FFP10 file does not exist.
    ValueError
        If the realisation index is out of the valid range.
    """
    if not 0 <= realization_index < N_FFP10_REALIZATIONS:
        raise ValueError(
            f"realization_index must be in [0, {N_FFP10_REALIZATIONS}); "
            f"got {realization_index}"
        )

    ffp10_path = Path(ffp10_dir)

    # Try several common naming conventions for FFP10 noise files
    candidate_names = [
        f"ffp10_noise_{realization_index:04d}.fits",
        f"ffp10_noise_{realization_index:03d}.fits",
        f"COM_SimMap_noise-ffp10-smica_{realization_index:05d}_R3.00_full.fits",
        f"ffp10_noise_143_full_map_mc_{realization_index:05d}.fits",
    ]

    fits_file = None
    for name in candidate_names:
        candidate = ffp10_path / name
        if candidate.exists():
            fits_file = candidate
            break

    if fits_file is None:
        raise FileNotFoundError(
            f"FFP10 noise realisation {realization_index} not found in "
            f"{ffp10_path}.  Tried: {candidate_names}"
        )

    logger.info(
        "Loading FFP10 noise realisation %d from %s",
        realization_index,
        fits_file,
    )
    noise_map = hp.read_map(str(fits_file), dtype=np.float64)

    # Degrade to target resolution if necessary
    nside_file = hp.get_nside(noise_map)
    if nside_file != nside_target:
        logger.info(
            "Degrading FFP10 noise from Nside=%d to Nside=%d",
            nside_file,
            nside_target,
        )
        noise_map = hp.ud_grade(noise_map, nside_target)

    return noise_map.astype(np.float32)


def generate_noise(
    noise_type: str = "white",
    nside: int = DEFAULT_NSIDE,
    seed: Optional[int] = None,
    sigma_uK: float = DEFAULT_SIGMA_UK,
    ffp10_dir: Union[str, Path] = DEFAULT_FFP10_DIR,
) -> np.ndarray:
    """Unified noise generation interface.

    Parameters
    ----------
    noise_type : str
        One of:
        - ``"white"`` : isotropic Gaussian white noise.
        - ``"ffp10"`` : load a single FFP10 noise realisation, selected
          by ``seed % 300``.
        - ``"ffp10_full"`` : same as ``"ffp10"`` but intended for use
          with the full-resolution pipeline (name kept for config compat).
    nside : int
        HEALPix Nside resolution parameter.
    seed : int, optional
        Random seed.  For FFP10, ``seed % 300`` selects the realisation.
    sigma_uK : float
        White-noise RMS in microKelvin per pixel (used only for
        ``noise_type="white"``).
    ffp10_dir : str or Path
        Directory containing FFP10 FITS files.

    Returns
    -------
    noise_map : np.ndarray, shape (12 * nside**2,), dtype float32
        Noise map in microKelvin.

    Raises
    ------
    ValueError
        If *noise_type* is not one of the supported values.
    """
    noise_type = noise_type.lower().strip()

    if noise_type == "white":
        return generate_white_noise(nside=nside, sigma_uK=sigma_uK, seed=seed)

    if noise_type in ("ffp10", "ffp10_full"):
        if seed is not None:
            idx = seed % N_FFP10_REALIZATIONS
        else:
            idx = 0
        try:
            return load_ffp10_noise(
                realization_index=idx,
                nside_target=nside,
                ffp10_dir=ffp10_dir,
            )
        except FileNotFoundError:
            logger.warning(
                "FFP10 noise file not found (index=%d); falling back to "
                "white noise with sigma=%.1f muK.",
                idx,
                sigma_uK,
            )
            return generate_white_noise(
                nside=nside, sigma_uK=sigma_uK, seed=seed
            )

    raise ValueError(
        f"Unknown noise_type '{noise_type}'. "
        f"Supported: 'white', 'ffp10', 'ffp10_full'."
    )


def estimate_noise_rms(
    half_ring_map1: np.ndarray,
    half_ring_map2: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Estimate the noise RMS from half-ring difference maps.

    The half-ring difference removes the sky signal, leaving twice the
    noise:  n_diff = (map1 - map2) / 2.  The RMS of n_diff gives an
    unbiased estimate of the per-pixel noise level.

    Parameters
    ----------
    half_ring_map1 : np.ndarray
        First half-ring map.
    half_ring_map2 : np.ndarray
        Second half-ring map (same Nside as *half_ring_map1*).
    mask : np.ndarray, optional
        Binary mask (1 = valid, 0 = masked).  If provided, the RMS is
        computed only over valid (unmasked) pixels.

    Returns
    -------
    sigma_noise : float
        Estimated noise RMS in the same units as the input maps.
    """
    diff = (half_ring_map1.astype(np.float64) - half_ring_map2.astype(np.float64)) / 2.0

    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            logger.warning("Mask has no valid pixels; returning 0.0.")
            return 0.0
        diff = diff[valid]

    sigma = float(np.std(diff))
    return sigma
