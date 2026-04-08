"""
Bubble collision signal injection into CMB temperature maps.

Implements the parametric temperature perturbation profile from the
eternal inflation / bubble collision framework:

    DeltaT/T(theta) = A * z(theta)^kappa * Theta_H(theta_r - theta)

where z(theta) = (cos(theta) - cos(theta_r)) / (1 - cos(theta_r))
is a normalised radial coordinate within the collision disk.

References
----------
- Gobbetti & Kleban (2012), JCAP 1205, 025.
- Feeney, Johnson, Mortlock & Peiris (2011), PRL 107, 071301.
- Chang, Kleban & Levi (2009), JCAP 0904, 025.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import healpy as hp
import numpy as np

logger = logging.getLogger(__name__)

# CMB monopole temperature in microKelvin
T_CMB_UK: float = 2.7255e6


@dataclass
class BubbleParams:
    """Parameters describing a single bubble collision signature.

    Attributes
    ----------
    theta_c : float
        Colatitude of the collision centre in radians.
    phi_c : float
        Longitude of the collision centre in radians.
    theta_r : float
        Angular radius of the collision disk in radians.
    amplitude : float
        Dimensionless amplitude A of the temperature perturbation
        (DeltaT / T).  May be positive (hot spot) or negative (cold spot).
    kappa : float
        Profile shape index (power-law exponent).
    """

    theta_c: float
    phi_c: float
    theta_r: float
    amplitude: float
    kappa: float

    def to_array(self) -> np.ndarray:
        """Return parameters as a float32 array of length 5."""
        return np.array(
            [self.theta_c, self.phi_c, self.theta_r, self.amplitude, self.kappa],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BubbleParams":
        """Construct from an array [theta_c, phi_c, theta_r, amplitude, kappa]."""
        return cls(
            theta_c=float(arr[0]),
            phi_c=float(arr[1]),
            theta_r=float(arr[2]),
            amplitude=float(arr[3]),
            kappa=float(arr[4]),
        )


def sample_bubble_params(
    rng: np.random.Generator,
    theta_r_range_deg: tuple[float, float] = (5.0, 60.0),
    amplitude_range: tuple[float, float] = (3e-5, 5e-4),
    kappa_values: tuple[int, ...] = (2, 3, 4),
) -> BubbleParams:
    """Draw random bubble collision parameters from the prior distributions.

    Sampling distributions (following the statistical framework):

    - ``cos(theta_c)`` ~ Uniform(-1, 1)  (isotropic on the sphere)
    - ``phi_c`` ~ Uniform(0, 2*pi)
    - ``theta_r`` ~ LogUniform(theta_r_range_deg), in degrees then
      converted to radians
    - ``|A|`` ~ LogUniform(amplitude_range), sign ~ Bernoulli(0.5)
    - ``kappa`` ~ discrete Uniform over *kappa_values*

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator for reproducibility.
    theta_r_range_deg : tuple of float
        (min, max) angular radius range in degrees.
    amplitude_range : tuple of float
        (min, max) absolute amplitude range (dimensionless DeltaT/T).
    kappa_values : tuple of int
        Allowed discrete values for the profile shape index.

    Returns
    -------
    params : BubbleParams
        Randomly sampled collision parameters.
    """
    # Position: uniform on the sphere
    cos_theta = rng.uniform(-1.0, 1.0)
    theta_c = np.arccos(cos_theta)
    phi_c = rng.uniform(0.0, 2.0 * np.pi)

    # Angular radius: log-uniform in degrees, then convert to radians
    log_theta_r_min = np.log(theta_r_range_deg[0])
    log_theta_r_max = np.log(theta_r_range_deg[1])
    theta_r_deg = np.exp(rng.uniform(log_theta_r_min, log_theta_r_max))
    theta_r = np.radians(theta_r_deg)

    # Amplitude: log-uniform magnitude, random sign
    log_amp_min = np.log(amplitude_range[0])
    log_amp_max = np.log(amplitude_range[1])
    abs_amplitude = np.exp(rng.uniform(log_amp_min, log_amp_max))
    sign = rng.choice([-1.0, 1.0])
    amplitude = sign * abs_amplitude

    # Shape index: discrete uniform over allowed values
    kappa = float(rng.choice(kappa_values))

    return BubbleParams(
        theta_c=theta_c,
        phi_c=phi_c,
        theta_r=theta_r,
        amplitude=amplitude,
        kappa=kappa,
    )


def generate_bubble_template(
    params: BubbleParams,
    nside: int = 64,
    t_cmb_uk: float = T_CMB_UK,
) -> np.ndarray:
    """Generate a HEALPix map of the bubble collision temperature perturbation.

    Implements the parametric profile (Eq. 2.2 of the theoretical framework):

        DeltaT/T(theta) = A * z(theta)^kappa * Theta_H(theta_r - theta)

    where
        z(theta) = (cos(theta) - cos(theta_r)) / (1 - cos(theta_r))

    and theta is the angular distance from the collision centre.  The
    template is returned in microKelvin (i.e., DeltaT = A * T_CMB * z^kappa).

    Parameters
    ----------
    params : BubbleParams
        Collision parameters.
    nside : int
        HEALPix Nside resolution.
    t_cmb_uk : float
        CMB monopole temperature in microKelvin (default 2.7255e6).

    Returns
    -------
    template : np.ndarray, shape (12 * nside**2,), dtype float32
        Temperature perturbation map in microKelvin.
    """
    npix = hp.nside2npix(nside)
    template = np.zeros(npix, dtype=np.float64)

    # Get the angular coordinates of all pixels
    pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))

    # Compute angular distance from the collision centre for each pixel
    # using the spherical law of cosines:
    #   cos(alpha) = cos(theta_c)*cos(theta_p)
    #                + sin(theta_c)*sin(theta_p)*cos(phi_c - phi_p)
    cos_alpha = (
        np.cos(params.theta_c) * np.cos(pix_theta)
        + np.sin(params.theta_c)
        * np.sin(pix_theta)
        * np.cos(params.phi_c - pix_phi)
    )
    # Clamp for numerical safety
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)

    # Identify pixels inside the collision disk
    inside = alpha < params.theta_r

    if not np.any(inside):
        logger.debug(
            "No pixels inside collision disk (theta_r=%.2f deg, nside=%d)",
            np.degrees(params.theta_r),
            nside,
        )
        return template.astype(np.float32)

    # Compute the normalised radial coordinate z(theta)
    cos_theta_r = np.cos(params.theta_r)
    denominator = 1.0 - cos_theta_r
    if abs(denominator) < 1e-15:
        logger.warning("theta_r ~ 0; degenerate collision disk.")
        return template.astype(np.float32)

    z = (cos_alpha[inside] - cos_theta_r) / denominator
    # Clamp z to [0, 1] for numerical safety
    z = np.clip(z, 0.0, 1.0)

    # Profile: DeltaT = A * T_CMB * z^kappa
    template[inside] = params.amplitude * t_cmb_uk * np.power(z, params.kappa)

    return template.astype(np.float32)


def inject_bubble(
    cmb_map: np.ndarray,
    params: BubbleParams,
    nside: int = 64,
    t_cmb_uk: float = T_CMB_UK,
) -> np.ndarray:
    """Inject a bubble collision signal into a CMB map.

    The collision template is generated and added to the input CMB map.

    Parameters
    ----------
    cmb_map : np.ndarray, shape (12 * nside**2,)
        Input CMB temperature map in microKelvin.
    params : BubbleParams
        Collision parameters.
    nside : int
        HEALPix Nside resolution parameter.
    t_cmb_uk : float
        CMB monopole temperature in microKelvin.

    Returns
    -------
    injected_map : np.ndarray, shape (12 * nside**2,), dtype float32
        CMB + collision signal map in microKelvin.
    """
    template = generate_bubble_template(params, nside=nside, t_cmb_uk=t_cmb_uk)
    injected = cmb_map.astype(np.float64) + template.astype(np.float64)
    return injected.astype(np.float32)


def generate_bubble_on_grid(
    theta_r_deg: float,
    amplitude: float,
    kappa: float,
    nside: int = 64,
    center_theta: float = 0.0,
    center_phi: float = 0.0,
    t_cmb_uk: float = T_CMB_UK,
) -> np.ndarray:
    """Generate a bubble template at specific parameter values.

    Convenience function for injection-recovery studies where parameters
    are drawn on a regular grid rather than sampled from the prior.

    Parameters
    ----------
    theta_r_deg : float
        Angular radius in degrees.
    amplitude : float
        Dimensionless amplitude A (with sign).
    kappa : float
        Profile shape index.
    nside : int
        HEALPix Nside resolution.
    center_theta : float
        Colatitude of centre in radians.  Default is the north pole.
    center_phi : float
        Longitude of centre in radians.  Default is 0.
    t_cmb_uk : float
        CMB monopole temperature in microKelvin.

    Returns
    -------
    template : np.ndarray, shape (12 * nside**2,), dtype float32
        Collision template in microKelvin.
    """
    params = BubbleParams(
        theta_c=center_theta,
        phi_c=center_phi,
        theta_r=np.radians(theta_r_deg),
        amplitude=amplitude,
        kappa=kappa,
    )
    return generate_bubble_template(params, nside=nside, t_cmb_uk=t_cmb_uk)
