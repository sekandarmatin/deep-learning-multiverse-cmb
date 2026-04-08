"""
Bayesian evidence computation for bubble collision model comparison.

Computes the Bayes factor B_10 = Z_1 / Z_0 comparing a single-collision
model (M_1) against the null LCDM model (M_0) using nested sampling
(dynesty).

Key components:

- Analytic null-model evidence (Gaussian CMB + Gaussian noise)
- Pixel-domain likelihood for the collision model
- Prior transform mapping the unit hypercube to physical parameters
- Nested-sampling driver returning evidence, posteriors, and diagnostics
- Jeffreys-scale interpretation of the Bayes factor

References
----------
- Skilling (2006), Bayesian Analysis 1, 833
- Speagle (2020), MNRAS 493, 3132 (dynesty)
- Jeffreys (1961), *Theory of Probability*
- Architecture document, Section 2.12
- Statistical Framework, Section 5
- Theoretical Framework, Eq. (2.2)--(2.3)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
T_CMB_UK = 2.7255e6  # CMB temperature in micro-Kelvin

# Default prior ranges (matching training-data distributions)
PRIOR_THETA_R_DEG = (5.0, 60.0)       # degrees
PRIOR_AMPLITUDE = (3e-5, 5e-4)        # dimensionless |A|
PRIOR_KAPPA = (2.0, 4.0)


# ====================================================================
# Result dataclass
# ====================================================================

@dataclass
class EvidenceResult:
    """Result from Bayesian evidence computation.

    Attributes
    ----------
    ln_Z0 : float
        Log-evidence for the null (LCDM) model.
    ln_Z1 : float
        Log-evidence for the collision model.
    ln_Z1_err : float
        Statistical uncertainty on ``ln_Z1`` from nested sampling.
    ln_bayes_factor : float
        ``ln(B_10) = ln_Z1 - ln_Z0``.
    bayes_factor : float
        ``B_10 = exp(ln_bayes_factor)``.
    interpretation : str
        Jeffreys-scale interpretation.
    posterior_samples : np.ndarray
        Weighted posterior samples, shape ``(N_samples, 5)``.
        Columns: ``(theta_c, phi_c, theta_r, A, kappa)``.
    posterior_weights : np.ndarray
        Importance weights, shape ``(N_samples,)``.
    n_likelihood_calls : int
        Total number of likelihood evaluations.
    wall_time_seconds : float
        Wall-clock computation time.
    """

    ln_Z0: float = 0.0
    ln_Z1: float = 0.0
    ln_Z1_err: float = 0.0
    ln_bayes_factor: float = 0.0
    bayes_factor: float = 1.0
    interpretation: str = "Not computed"
    posterior_samples: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5))
    )
    posterior_weights: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    n_likelihood_calls: int = 0
    wall_time_seconds: float = 0.0


# ====================================================================
# Prior transform
# ====================================================================

def bubble_prior_transform(
    unit_cube: np.ndarray,
    theta_r_range_deg: tuple[float, float] = PRIOR_THETA_R_DEG,
    amplitude_range: tuple[float, float] = PRIOR_AMPLITUDE,
    kappa_range: tuple[float, float] = PRIOR_KAPPA,
) -> np.ndarray:
    """Transform the unit hypercube [0, 1]^5 to physical parameter space.

    Mapping:

    - ``u[0]`` -> ``cos(theta_c)`` uniform in [-1, 1] -> ``theta_c``
    - ``u[1]`` -> ``phi_c`` uniform in [0, 2*pi]
    - ``u[2]`` -> ``theta_r`` log-uniform in *theta_r_range_deg*
    - ``u[3]`` -> ``|A|`` log-uniform in *amplitude_range*,
      sign determined by whether ``u[3] < 0.5`` (negative) or >= 0.5
      (positive) via a split of the unit interval
    - ``u[4]`` -> ``kappa`` uniform in *kappa_range*

    Parameters
    ----------
    unit_cube : np.ndarray, shape ``(5,)``
        Point in the unit hypercube.
    theta_r_range_deg : tuple
        ``(min, max)`` angular radius in degrees.
    amplitude_range : tuple
        ``(min, max)`` absolute amplitude.
    kappa_range : tuple
        ``(min, max)`` profile shape index.

    Returns
    -------
    params : np.ndarray, shape ``(5,)``
        ``(theta_c, phi_c, theta_r, A, kappa)`` in physical units.
        ``theta_c`` in radians, ``phi_c`` in radians,
        ``theta_r`` in radians, ``A`` dimensionless, ``kappa`` dimensionless.
    """
    u = np.asarray(unit_cube, dtype=np.float64)
    params = np.empty(5, dtype=np.float64)

    # theta_c: isotropic on sphere -> cos(theta_c) uniform in [-1, 1]
    cos_theta_c = 2.0 * u[0] - 1.0
    params[0] = np.arccos(np.clip(cos_theta_c, -1.0, 1.0))

    # phi_c: uniform in [0, 2*pi]
    params[1] = 2.0 * np.pi * u[1]

    # theta_r: log-uniform in [theta_r_min, theta_r_max] (convert to radians)
    ln_min = np.log(np.radians(theta_r_range_deg[0]))
    ln_max = np.log(np.radians(theta_r_range_deg[1]))
    params[2] = np.exp(ln_min + u[2] * (ln_max - ln_min))

    # A: log-uniform in |A|, random sign
    # Split u[3] into sign (first half) and magnitude (second half)
    # Use u[3] to encode both: map [0, 0.5) -> negative, [0.5, 1] -> positive
    # Magnitude from a remapped variable
    ln_a_min = np.log(amplitude_range[0])
    ln_a_max = np.log(amplitude_range[1])
    if u[3] < 0.5:
        # Negative amplitude
        u_mag = 2.0 * u[3]  # remap [0, 0.5) -> [0, 1)
        abs_a = np.exp(ln_a_min + u_mag * (ln_a_max - ln_a_min))
        params[3] = -abs_a
    else:
        # Positive amplitude
        u_mag = 2.0 * (u[3] - 0.5)  # remap [0.5, 1] -> [0, 1]
        abs_a = np.exp(ln_a_min + u_mag * (ln_a_max - ln_a_min))
        params[3] = abs_a

    # kappa: uniform in [kappa_min, kappa_max]
    params[4] = kappa_range[0] + u[4] * (kappa_range[1] - kappa_range[0])

    return params


# ====================================================================
# Pixel-domain likelihood
# ====================================================================

def pixel_log_likelihood(
    params: np.ndarray,
    data_map: np.ndarray,
    noise_rms: Union[float, np.ndarray],
    mask: np.ndarray,
    nside: int = 64,
) -> float:
    """Compute the pixel-domain log-likelihood for the collision model.

    Implements:

        ln L = -(1/2) sum_{i in unmasked}
               [(T_obs_i - T_model_i)^2 / sigma_i^2
                + ln(2 pi sigma_i^2)]

    where ``T_model_i = T_LCDM_i + DeltaT_i(params)`` and the LCDM
    component is absorbed into the residual so that ``T_obs`` is the
    data and ``T_model`` is the collision template alone (the LCDM
    component cancels in the likelihood ratio).

    Parameters
    ----------
    params : np.ndarray, shape ``(5,)``
        ``(theta_c, phi_c, theta_r, A, kappa)`` in physical units
        (radians for angles, dimensionless for A and kappa).
    data_map : np.ndarray, shape ``(N_pix,)``
        Observed map in micro-Kelvin (or residual after LCDM subtraction).
    noise_rms : float or np.ndarray
        Noise RMS per pixel.  Scalar for uniform noise, array of shape
        ``(N_pix,)`` for inhomogeneous noise.
    mask : np.ndarray, shape ``(N_pix,)``
        Binary mask (1 = valid).
    nside : int
        HEALPix resolution.

    Returns
    -------
    ln_L : float
        Log-likelihood.
    """
    theta_c, phi_c, theta_r, A, kappa = params

    npix = 12 * nside * nside
    valid = mask.astype(bool)

    # Generate collision template
    template = _generate_collision_template(
        theta_c, phi_c, theta_r, A, kappa, nside
    )

    # Residual: data - collision template
    residual = data_map[:npix] - template[:npix]

    # Noise variance
    if np.isscalar(noise_rms):
        noise_var = float(noise_rms) ** 2
        ln_L = -0.5 * np.sum(residual[valid] ** 2 / noise_var)
        # Normalisation constant (constant for fixed noise, but include
        # for correctness)
        n_valid = int(np.sum(valid))
        ln_L -= 0.5 * n_valid * np.log(2.0 * np.pi * noise_var)
    else:
        noise_v = np.asarray(noise_rms, dtype=np.float64) ** 2
        nv = noise_v[:npix]
        safe = valid & (nv > 0)
        ln_L = -0.5 * np.sum(residual[safe] ** 2 / nv[safe])
        ln_L -= 0.5 * np.sum(np.log(2.0 * np.pi * nv[safe]))

    return float(ln_L)


def _generate_collision_template(
    theta_c: float,
    phi_c: float,
    theta_r: float,
    A: float,
    kappa: float,
    nside: int,
) -> np.ndarray:
    """Generate a collision template map.

    Implements Eq. (2.2):
        DeltaT(theta) = A * T_CMB * z(theta)^kappa * Theta_H(theta_r - theta)

    where z = (cos(alpha) - cos(theta_r)) / (1 - cos(theta_r))
    and alpha is the angular distance from (theta_c, phi_c).

    Parameters
    ----------
    theta_c, phi_c : float
        Centre colatitude and longitude in radians.
    theta_r : float
        Angular radius in radians.
    A : float
        Dimensionless amplitude.
    kappa : float
        Profile shape index.
    nside : int
        HEALPix resolution.

    Returns
    -------
    template : np.ndarray, shape ``(12 * nside**2,)``, dtype float64
    """
    npix = 12 * nside * nside
    template = np.zeros(npix, dtype=np.float64)

    # Centre unit vector
    cx = np.sin(theta_c) * np.cos(phi_c)
    cy = np.sin(theta_c) * np.sin(phi_c)
    cz = np.cos(theta_c)

    cos_theta_r = np.cos(theta_r)
    denom = 1.0 - cos_theta_r
    if abs(denom) < 1e-15:
        return template

    # Pixel unit vectors
    pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))
    px = np.sin(pix_theta) * np.cos(pix_phi)
    py = np.sin(pix_theta) * np.sin(pix_phi)
    pz = np.cos(pix_theta)

    cos_alpha = cx * px + cy * py + cz * pz
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    # Pixels inside the collision disk
    inside = cos_alpha > cos_theta_r
    if not np.any(inside):
        return template

    z = (cos_alpha[inside] - cos_theta_r) / denom
    z = np.clip(z, 0.0, 1.0)

    template[inside] = A * T_CMB_UK * (z ** kappa)
    return template


# ====================================================================
# Null model evidence (analytic for Gaussian field + Gaussian noise)
# ====================================================================

def _compute_null_evidence_analytic(
    observed_map: np.ndarray,
    cl_theory: np.ndarray,
    noise_sigma: Union[float, np.ndarray],
    mask: np.ndarray,
    nside: int = 64,
    lmax: int = 191,
) -> float:
    """Compute analytic log-evidence for the null LCDM model.

    For Gaussian CMB + Gaussian noise:

        ln(Z_0) = -(1/2) sum_{l,m} [|a_lm^obs|^2 / (C_l + N_l)
                  + ln(2 pi (C_l + N_l))]

    In practice, we use the pseudo-C_l of the masked map as a proxy.

    Parameters
    ----------
    observed_map : np.ndarray
        Observed map at target Nside.
    cl_theory : np.ndarray
        LCDM power spectrum, shape ``(lmax + 1,)``.
    noise_sigma : float or np.ndarray
        Noise RMS per pixel.
    mask : np.ndarray
        Binary mask.
    nside : int
        HEALPix resolution.
    lmax : int
        Maximum multipole.

    Returns
    -------
    ln_Z0 : float
    """
    npix = 12 * nside * nside
    obs = np.asarray(observed_map, dtype=np.float64)[:npix]
    msk = np.asarray(mask, dtype=np.float64)[:npix]
    f_sky = max(float(np.mean(msk)), 1e-10)

    # Compute observed a_lm (on masked map)
    masked_map = obs * msk
    cl_obs = hp.anafast(masked_map, lmax=lmax) / f_sky

    # Noise power spectrum
    if np.isscalar(noise_sigma):
        omega_pix = 4.0 * np.pi / npix
        n_l = (float(noise_sigma) ** 2) * omega_pix
    else:
        # Average noise variance times pixel solid angle
        omega_pix = 4.0 * np.pi / npix
        n_l = float(np.mean(np.asarray(noise_sigma) ** 2)) * omega_pix

    cl_th = np.asarray(cl_theory, dtype=np.float64)
    n = min(len(cl_th), lmax + 1, len(cl_obs))

    ell = np.arange(n, dtype=np.float64)
    c_total = cl_th[:n] + n_l

    # Avoid issues at l=0, l=1 where C_l may be zero
    valid = (ell >= 2) & (c_total > 0)

    ln_z0 = 0.0
    for l_idx in range(n):
        if not valid[l_idx]:
            continue
        l_val = ell[l_idx]
        n_modes = 2 * l_val + 1  # number of m values
        ct = c_total[l_idx]
        co = cl_obs[l_idx]
        ln_z0 += -0.5 * n_modes * (co / ct + np.log(2.0 * np.pi * ct))

    return float(ln_z0)


# ====================================================================
# Full evidence computation
# ====================================================================

def compute_bayes_factor(
    data_map: np.ndarray,
    noise_rms: Union[float, np.ndarray],
    mask: np.ndarray,
    nside: int = 64,
    nlive: int = 500,
    dlogz: float = 0.5,
    cl_theory: Optional[np.ndarray] = None,
    lmax: int = 191,
    sampling_method: str = "rwalk",
    theta_r_range_deg: tuple[float, float] = PRIOR_THETA_R_DEG,
    amplitude_range: tuple[float, float] = PRIOR_AMPLITUDE,
    kappa_range: tuple[float, float] = PRIOR_KAPPA,
) -> EvidenceResult:
    """Compute the Bayes factor B_10 via nested sampling.

    Parameters
    ----------
    data_map : np.ndarray
        Observed map in micro-Kelvin at *nside*.
    noise_rms : float or np.ndarray
        Noise RMS per pixel.
    mask : np.ndarray
        Binary mask.
    nside : int
        HEALPix resolution.
    nlive : int
        Number of live points for nested sampling.
    dlogz : float
        Convergence criterion: stop when remaining evidence contribution
        is less than ``exp(dlogz)`` of the current estimate.
    cl_theory : np.ndarray or None
        LCDM power spectrum for null-model evidence.  If ``None``,
        the null evidence is set to 0 (so ``ln_B10 = ln_Z1``).
    lmax : int
        Maximum multipole for null evidence computation.
    sampling_method : str
        Nested-sampling method (``"rwalk"``, ``"slice"``, etc.).
    theta_r_range_deg : tuple
        Prior range for angular radius (degrees).
    amplitude_range : tuple
        Prior range for |A|.
    kappa_range : tuple
        Prior range for kappa.

    Returns
    -------
    result : EvidenceResult
    """
    try:
        import dynesty
    except ImportError:
        logger.error(
            "dynesty is not installed.  Install it with: "
            "pip install dynesty"
        )
        return EvidenceResult(interpretation="dynesty not available")

    t0 = time.time()

    data = np.asarray(data_map, dtype=np.float64)
    msk = np.asarray(mask, dtype=np.float64)

    # Define likelihood and prior for dynesty
    def _loglike(params: np.ndarray) -> float:
        return pixel_log_likelihood(params, data, noise_rms, msk, nside)

    def _prior(u: np.ndarray) -> np.ndarray:
        return bubble_prior_transform(
            u,
            theta_r_range_deg=theta_r_range_deg,
            amplitude_range=amplitude_range,
            kappa_range=kappa_range,
        )

    # Run nested sampling
    logger.info(
        "Starting nested sampling: nlive=%d, dlogz=%.2f, method=%s",
        nlive, dlogz, sampling_method,
    )

    sampler = dynesty.NestedSampler(
        _loglike,
        _prior,
        ndim=5,
        nlive=nlive,
        sample=sampling_method,
    )
    sampler.run_nested(dlogz=dlogz)
    ns_results = sampler.results

    ln_z1 = float(ns_results.logz[-1])
    ln_z1_err = float(ns_results.logzerr[-1])

    # Extract posterior samples and weights
    try:
        from dynesty.utils import resample_equal
        samples = ns_results.samples
        weights = np.exp(ns_results.logwt - ns_results.logz[-1])
        weights /= weights.sum()
        eq_samples = resample_equal(samples, weights)
    except Exception:
        eq_samples = ns_results.samples
        weights = np.ones(len(eq_samples)) / len(eq_samples)

    # Null model evidence
    if cl_theory is not None:
        ln_z0 = _compute_null_evidence_analytic(
            data, cl_theory, noise_rms, msk, nside, lmax,
        )
    else:
        ln_z0 = 0.0
        logger.warning("No theoretical C_l provided; ln_Z0 set to 0.")

    ln_bf = ln_z1 - ln_z0
    # Prevent overflow in exp
    if ln_bf > 500:
        bf = np.inf
    elif ln_bf < -500:
        bf = 0.0
    else:
        bf = np.exp(ln_bf)

    interp = interpret_bayes_factor(ln_bf)

    n_calls = int(np.sum(ns_results.ncall)) if hasattr(ns_results.ncall, '__len__') else int(ns_results.ncall)
    wall_time = time.time() - t0

    result = EvidenceResult(
        ln_Z0=ln_z0,
        ln_Z1=ln_z1,
        ln_Z1_err=ln_z1_err,
        ln_bayes_factor=ln_bf,
        bayes_factor=bf,
        interpretation=interp,
        posterior_samples=eq_samples,
        posterior_weights=weights,
        n_likelihood_calls=n_calls,
        wall_time_seconds=wall_time,
    )

    logger.info(
        "Nested sampling complete: ln(Z1)=%.2f +/- %.2f, "
        "ln(Z0)=%.2f, ln(B10)=%.2f (%s), "
        "%d likelihood calls in %.1f s",
        ln_z1, ln_z1_err, ln_z0, ln_bf, interp, n_calls, wall_time,
    )
    return result


# ====================================================================
# Jeffreys scale interpretation
# ====================================================================

def interpret_bayes_factor(log_B10: float) -> str:
    """Interpret the log Bayes factor on the Jeffreys scale.

    Parameters
    ----------
    log_B10 : float
        Natural logarithm of the Bayes factor B_10.

    Returns
    -------
    interpretation : str
        One of:
        - ``"Favors LCDM"`` (ln B10 < 0)
        - ``"Not worth more than a bare mention"`` (0 <= ln B10 < 1)
        - ``"Substantial evidence for collision"`` (1 <= ln B10 < 2.5)
        - ``"Strong evidence for collision"`` (2.5 <= ln B10 < 5)
        - ``"Decisive evidence for collision"`` (ln B10 >= 5)
    """
    if log_B10 < 0:
        return "Favors LCDM"
    elif log_B10 < 1.0:
        return "Not worth more than a bare mention"
    elif log_B10 < 2.5:
        return "Substantial evidence for collision"
    elif log_B10 < 5.0:
        return "Strong evidence for collision"
    else:
        return "Decisive evidence for collision"


# ====================================================================
# Quick approximate Bayes factor
# ====================================================================

def approximate_bayes_factor(
    p_coll: float,
    prior_odds: float = 1.0,
) -> float:
    """Quick approximate Bayes factor from the calibrated network output.

    B_10 ~ [p_coll / (1 - p_coll)] * [pi(H0) / pi(H1)]

    This assumes the network output is a well-calibrated posterior
    probability under equal prior odds.

    Parameters
    ----------
    p_coll : float
        Calibrated collision probability from the network.
    prior_odds : float
        Prior odds ratio pi(H0) / pi(H1).  Default 1.0 (equal priors).

    Returns
    -------
    B_10 : float
        Approximate Bayes factor.
    """
    p = np.clip(p_coll, 1e-15, 1.0 - 1e-15)
    return float((p / (1.0 - p)) * prior_odds)
