"""
Gaussian CMB map generation from Planck 2018 best-fit LCDM power spectrum.

This module provides functions to compute theoretical angular power spectra
using CAMB (or from a cached file), generate spherical harmonic coefficients,
and synthesize full-sky HEALPix temperature maps at configurable resolution.

References
----------
- Planck Collaboration (2020), A&A 641, A6 (Planck 2018 cosmological parameters).
- Lewis, Challinor & Lasenby (2000), ApJ 538, 473 (CAMB).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planck 2018 best-fit cosmological parameters
# (TT,TE,EE+lowE+lensing, Table 2 of Planck 2018 VI)
# ---------------------------------------------------------------------------
PLANCK2018_BESTFIT: dict[str, float] = {
    "H0": 67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau": 0.0544,
    "As": 2.1e-9,        # 10^10 A_s = 3.044  =>  A_s = 2.1e-9
    "ns": 0.9649,
}

# Default simulation parameters
DEFAULT_NSIDE: int = 64
DEFAULT_LMAX: int = 191       # 3 * NSIDE - 1
DEFAULT_FWHM_ARCMIN: float = 80.0


def compute_theoretical_cl(
    cosmo_params: Optional[dict[str, float]] = None,
    lmax: int = DEFAULT_LMAX,
    cache_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Compute theoretical C_l using CAMB with Planck 2018 best-fit parameters.

    If a cache file exists at *cache_path*, the spectrum is loaded from disk
    rather than recomputed.  When CAMB is not available, the function falls
    back to an analytic approximation of the Planck power spectrum.

    Parameters
    ----------
    cosmo_params : dict, optional
        Cosmological parameters.  Keys: H0, ombh2, omch2, tau, As, ns.
        If ``None``, uses Planck 2018 best-fit values.
    lmax : int
        Maximum multipole.
    cache_path : str or Path, optional
        Path to cache the computed C_l as a ``.npy`` file.  If the file
        already exists it is loaded directly.

    Returns
    -------
    cl : np.ndarray, shape (lmax + 1,)
        Temperature power spectrum C_l in units of (microKelvin)^2.
    """
    # ------------------------------------------------------------------
    # Try to load from cache
    # ------------------------------------------------------------------
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached C_l from %s", cache_path)
            cl = np.load(cache_path)
            if cl.shape[0] >= lmax + 1:
                return cl[: lmax + 1].astype(np.float64)
            logger.warning(
                "Cached C_l has lmax=%d but lmax=%d requested; recomputing.",
                cl.shape[0] - 1,
                lmax,
            )

    params = cosmo_params if cosmo_params is not None else PLANCK2018_BESTFIT

    # ------------------------------------------------------------------
    # Attempt CAMB computation
    # ------------------------------------------------------------------
    try:
        import camb  # type: ignore

        logger.info("Computing C_l with CAMB (lmax=%d)", lmax)

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=params["H0"],
            ombh2=params["ombh2"],
            omch2=params["omch2"],
            tau=params["tau"],
        )
        pars.InitPower.set_params(
            As=params["As"],
            ns=params["ns"],
        )
        pars.set_for_lmax(lmax, lens_potential_accuracy=1)
        pars.WantTensors = False

        results = camb.get_results(pars)
        # get_cmb_power_spectra returns l(l+1)C_l / (2pi) in muK^2
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        totcl = powers["total"][:, 0]  # TT column

        # Convert from D_l = l(l+1)C_l/(2pi) to C_l
        ells = np.arange(totcl.shape[0], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            cl = np.where(ells > 0, totcl * 2.0 * np.pi / (ells * (ells + 1)), 0.0)
        cl[0] = 0.0  # monopole undefined

        cl = cl[: lmax + 1]

    except ImportError:
        logger.warning(
            "CAMB not installed; falling back to analytic approximation."
        )
        cl = _analytic_cl_approximation(params, lmax)

    # ------------------------------------------------------------------
    # Cache result
    # ------------------------------------------------------------------
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, cl)
        logger.info("Cached C_l to %s", cache_path)

    return cl


def _analytic_cl_approximation(
    params: dict[str, float],
    lmax: int,
) -> np.ndarray:
    """Analytic approximation of the CMB TT power spectrum.

    This is a simple Sachs-Wolfe + acoustic peak approximation used only
    when CAMB is not available.  It is *not* accurate for production
    science but is sufficient for pipeline testing.

    Parameters
    ----------
    params : dict
        Cosmological parameters (As, ns used).
    lmax : int
        Maximum multipole.

    Returns
    -------
    cl : np.ndarray, shape (lmax + 1,)
        Approximate C_l in (muK)^2.
    """
    ells = np.arange(lmax + 1, dtype=np.float64)
    ns = params.get("ns", 0.9649)
    As = params.get("As", 2.1e-9)

    # Scale amplitude so that the resulting map has roughly the correct
    # variance (order of ~100 muK rms).
    # D_l = l(l+1)C_l/(2pi) ~ A_s * (l/l_pivot)^{n_s - 1}  for Sachs-Wolfe
    # plus a Gaussian acoustic peak at l~220
    l_pivot = 80.0
    amplitude_scale = As / 2.1e-9  # normalise relative to fiducial

    with np.errstate(divide="ignore", invalid="ignore"):
        # Sachs-Wolfe plateau
        dl_sw = np.where(
            ells > 1,
            amplitude_scale * 1100.0 * (ells / l_pivot) ** (ns - 1),
            0.0,
        )
        # Acoustic peaks (very crude)
        dl_peaks = (
            amplitude_scale
            * 5500.0
            * np.exp(-((ells - 220.0) ** 2) / (2.0 * 60.0**2))
        )
        dl_peaks += (
            amplitude_scale
            * 3000.0
            * np.exp(-((ells - 550.0) ** 2) / (2.0 * 40.0**2))
        )
        # Damping tail
        damping = np.exp(-ells * (ells + 1) / (2.0 * 1500.0**2))

        dl = (dl_sw + dl_peaks) * damping
        cl = np.where(ells > 1, dl * 2.0 * np.pi / (ells * (ells + 1)), 0.0)

    cl[0] = 0.0
    cl[1] = 0.0  # dipole suppressed
    return cl


def generate_cmb_alm(
    cl: np.ndarray,
    lmax: int = DEFAULT_LMAX,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate spherical harmonic coefficients for a CMB realization.

    Draws a_lm from a Gaussian distribution with variance C_l for each
    (l, m) mode.  Useful when alm-level operations (e.g., rotation) are
    needed before converting to a pixel-domain map.

    Parameters
    ----------
    cl : np.ndarray, shape (lmax + 1,) or larger
        Input power spectrum in (muK)^2.
    lmax : int
        Maximum multipole.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    alm : np.ndarray, complex128
        Spherical harmonic coefficients in healpy ordering.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure cl has the right length
    cl_use = np.zeros(lmax + 1, dtype=np.float64)
    n = min(len(cl), lmax + 1)
    cl_use[:n] = cl[:n]

    # healpy.synalm generates complex a_lm from the power spectrum
    alm = hp.synalm(cl_use, lmax=lmax, new=True)
    return alm


def generate_cmb_map(
    cl: np.ndarray,
    nside: int = DEFAULT_NSIDE,
    fwhm_arcmin: float = DEFAULT_FWHM_ARCMIN,
    seed: Optional[int] = None,
    use_pixel_window: bool = True,
) -> np.ndarray:
    """Generate a single Gaussian CMB realization as a HEALPix map.

    Pipeline: C_l -> a_lm (Gaussian draw) -> beam smoothing -> pixel
    window -> map via healpy.synfast.

    Parameters
    ----------
    cl : np.ndarray, shape (lmax + 1,) or larger
        Input power spectrum in (muK)^2.
    nside : int
        HEALPix Nside parameter.
    fwhm_arcmin : float
        Beam full-width at half-maximum in arcminutes, passed to
        ``healpy.synfast``.
    seed : int, optional
        Random seed for reproducibility.
    use_pixel_window : bool
        Whether to apply the HEALPix pixel window function.

    Returns
    -------
    cmb_map : np.ndarray, shape (12 * nside**2,), dtype float32
        Simulated CMB temperature map in microKelvin.
    """
    lmax = 3 * nside - 1
    cl_use = np.zeros(lmax + 1, dtype=np.float64)
    n = min(len(cl), lmax + 1)
    cl_use[:n] = cl[:n]

    fwhm_rad = np.radians(fwhm_arcmin / 60.0)

    if seed is not None:
        np.random.seed(seed)

    cmb_map = hp.synfast(
        cl_use,
        nside=nside,
        lmax=lmax,
        fwhm=fwhm_rad,
        pixwin=use_pixel_window,
        new=True,
    )

    return cmb_map.astype(np.float32)
