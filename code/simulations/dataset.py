"""
PyTorch Dataset and DataLoader for CMB map simulations.

Provides a unified interface for generating or loading CMB temperature
maps (with optional bubble collision injection and noise), applying
preprocessing (masking, monopole/dipole removal, standardisation), and
feeding them to training / evaluation pipelines via PyTorch DataLoaders.

Two modes of operation are supported:

1. **On-the-fly generation** -- maps are simulated during ``__getitem__``
   calls.  Memory-efficient but slower.
2. **Pre-generated HDF5** -- maps are loaded from an HDF5 file produced
   by :func:`create_dataset_hdf5`.  Fast random access at the cost of
   disk space.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import healpy as hp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .bubble_simulator import BubbleParams, inject_bubble, sample_bubble_params
from .cmb_simulator import compute_theoretical_cl, generate_cmb_map
from .noise_simulator import generate_noise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split seed ranges (statistical framework, Section 1.7)
# ---------------------------------------------------------------------------
SPLIT_SEED_RANGES: dict[str, tuple[int, int]] = {
    "train": (0, 39_999),
    "val": (40_000, 44_999),
    "test": (45_000, 49_999),
    "null": (50_000, 59_999),
    "injection": (60_000, 69_999),
}

SPLIT_SIZES: dict[str, int] = {
    "train": 40_000,
    "val": 5_000,
    "test": 5_000,
    "null": 10_000,
    "injection": 10_000,
}

# Noise type mixture probabilities (architecture doc, Section 2.4)
DEFAULT_NOISE_MIX: dict[str, float] = {
    "white": 0.7,
    "ffp10": 0.2,
    "ffp10_full": 0.1,
}

# Mapping noise type names to integer codes for HDF5 storage
NOISE_TYPE_CODES: dict[str, int] = {"white": 0, "ffp10": 1, "ffp10_full": 2}
NOISE_CODE_TO_TYPE: dict[int, str] = {v: k for k, v in NOISE_TYPE_CODES.items()}

# Graph cache (module-level so it persists across Dataset instances)
_graph_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


# ============================================================================
# Preprocessing utilities
# ============================================================================

def load_mask(
    mask_path: Union[str, Path],
    nside_target: int = 64,
) -> np.ndarray:
    """Load and degrade a Planck mask to the target Nside.

    The mask is degraded by ``healpy.ud_grade`` and then thresholded at
    0.5 to produce a strict binary mask.

    Parameters
    ----------
    mask_path : str or Path
        Path to the FITS mask file.
    nside_target : int
        Target HEALPix resolution.

    Returns
    -------
    mask : np.ndarray, shape (12 * nside_target**2,), dtype float32
        Binary mask (0.0 or 1.0).
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        logger.warning(
            "Mask file %s not found; returning an all-sky (all-ones) mask.",
            mask_path,
        )
        return np.ones(hp.nside2npix(nside_target), dtype=np.float32)

    mask = hp.read_map(str(mask_path), dtype=np.float64)
    nside_mask = hp.get_nside(mask)

    if nside_mask != nside_target:
        mask = hp.ud_grade(mask, nside_target)

    # Threshold to binary
    mask = (mask >= 0.5).astype(np.float32)
    return mask


def preprocess_map(
    raw_map: np.ndarray,
    mask: np.ndarray,
    remove_monopole_dipole: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Apply mask, remove monopole/dipole, and standardise.

    Parameters
    ----------
    raw_map : np.ndarray
        Input map in microKelvin.
    mask : np.ndarray
        Binary mask (1 = valid, 0 = masked).
    remove_monopole_dipole : bool
        Whether to subtract the l = 0 (monopole) and l = 1 (dipole)
        components from unmasked pixels.

    Returns
    -------
    processed_map : np.ndarray, dtype float32
        Standardised map (zero mean, unit variance over unmasked pixels).
        Masked pixels are set to 0.
    mu : float
        Mean of unmasked pixels before standardisation.
    sigma : float
        Standard deviation of unmasked pixels before standardisation.
    """
    out = raw_map.astype(np.float64).copy()

    # Identify valid pixels
    valid = mask.astype(bool)

    # Remove monopole and dipole (l = 0, 1) from unmasked pixels
    if remove_monopole_dipole:
        try:
            out = hp.remove_dipole(out, gal_cut=0, nest=False, copy=True)
        except Exception:
            # healpy.remove_dipole can fail on partial skies;
            # fall back to simple mean subtraction
            out[valid] -= np.mean(out[valid])

    # Apply mask: zero out masked pixels
    out[~valid] = 0.0

    # Compute statistics over unmasked pixels
    valid_vals = out[valid]
    mu = float(np.mean(valid_vals))
    sigma = float(np.std(valid_vals))

    # Standardise
    if sigma > 0:
        out[valid] = (out[valid] - mu) / sigma
    else:
        logger.warning("Zero variance in unmasked pixels; skipping standardisation.")
        out[valid] = out[valid] - mu

    # Ensure masked pixels are exactly 0
    out[~valid] = 0.0

    return out.astype(np.float32), mu, sigma


def rotate_map_alm(
    input_map: np.ndarray,
    nside: int,
    lmax: int,
    euler_angles: tuple[float, float, float],
) -> np.ndarray:
    """Rotate a HEALPix map using spherical harmonic space rotation.

    The map is decomposed into a_lm, the a_lm are rotated by the given
    Euler angles, and then synthesised back into pixel space.

    Parameters
    ----------
    input_map : np.ndarray
        Input HEALPix map.
    nside : int
        HEALPix resolution.
    lmax : int
        Maximum multipole for the a_lm decomposition.
    euler_angles : tuple of float
        ``(alpha, beta, gamma)`` ZYZ Euler angles in radians.

    Returns
    -------
    rotated_map : np.ndarray, dtype float32
        Rotated HEALPix map.
    """
    alm = hp.map2alm(input_map.astype(np.float64), lmax=lmax)
    rotator = hp.Rotator(rot=euler_angles, deg=False, eulertype="ZYZ")
    alm_rot = rotator.rotate_alm(alm, lmax=lmax)
    rotated = hp.alm2map(alm_rot, nside, lmax=lmax)
    return rotated.astype(np.float32)


# ============================================================================
# HEALPix graph construction (for DeepSphere / graph convolutions)
# ============================================================================

def build_healpix_graph(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the adjacency structure for graph convolutions on HEALPix.

    Each pixel is connected to its nearest neighbours (8 neighbours for
    most pixels, 7 for some edge pixels).  Edge weights are set to the
    inverse angular distance between connected pixel centres.

    The result is cached in a module-level dictionary so that repeated
    calls for the same *nside* are free.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    edge_index : np.ndarray, shape (2, n_edges), dtype int64
        COO-format edge indices ``[source_pixels; target_pixels]``.
    edge_weight : np.ndarray, shape (n_edges,), dtype float32
        Edge weights (inverse angular distance in radians).
    """
    if nside in _graph_cache:
        return _graph_cache[nside]

    npix = hp.nside2npix(nside)
    logger.info("Building HEALPix graph for Nside=%d (%d pixels)", nside, npix)

    src_list: list[int] = []
    dst_list: list[int] = []
    weight_list: list[float] = []

    # Pixel angular positions (theta, phi) -- needed for edge weights
    pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))
    # Convert to unit vectors for fast angular distance computation
    pix_vec = hp.ang2vec(pix_theta, pix_phi)

    for ipix in range(npix):
        neighbours = hp.get_all_neighbours(nside, ipix)
        # hp.get_all_neighbours returns -1 for missing neighbours
        neighbours = neighbours[neighbours >= 0]

        for jpix in neighbours:
            # Angular distance via dot product of unit vectors
            cos_dist = np.dot(pix_vec[ipix], pix_vec[jpix])
            cos_dist = np.clip(cos_dist, -1.0, 1.0)
            ang_dist = np.arccos(cos_dist)
            # Avoid division by zero for coincident pixels
            w = 1.0 / max(ang_dist, 1e-10)

            src_list.append(ipix)
            dst_list.append(jpix)
            weight_list.append(w)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(weight_list, dtype=np.float32)

    _graph_cache[nside] = (edge_index, edge_weight)
    logger.info(
        "HEALPix graph: %d edges (avg %.1f per pixel)",
        edge_index.shape[1],
        edge_index.shape[1] / npix,
    )
    return edge_index, edge_weight


# ============================================================================
# Default configuration builder
# ============================================================================

def _default_config() -> dict[str, Any]:
    """Return a default configuration dictionary for simulations."""
    return {
        "nside": 64,
        "lmax": 191,
        "fwhm_arcmin": 80.0,
        "use_pixel_window": True,
        "signal_fraction": 0.5,
        "noise_type": "white",
        "noise_sigma_uK": 45.0,
        "noise_mix": DEFAULT_NOISE_MIX,
        "ffp10_dir": "data/planck/ffp10/",
        "mask_path": "data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
        "bubble": {
            "theta_r_range_deg": [5.0, 60.0],
            "amplitude_range": [3e-5, 5e-4],
            "kappa_values": [2, 3, 4],
        },
        "batch_size": 64,
        "num_workers": 4,
        "n_augmentation_rotations": 2,
        "cl_cache_path": "data/cache/planck2018_cl.npy",
    }


def _merge_config(user_config: Optional[dict] = None) -> dict[str, Any]:
    """Merge user-supplied config with defaults."""
    cfg = _default_config()
    if user_config is not None:
        cfg.update(user_config)
    return cfg


# ============================================================================
# PyTorch Dataset
# ============================================================================

class CMBMapDataset(Dataset):
    """PyTorch Dataset for CMB maps with optional bubble collision injection.

    Supports two modes:

    1. **Pre-generated** -- loads maps from an HDF5 file on disk.
    2. **On-the-fly** -- generates maps during ``__getitem__`` calls.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``, ``"null"``,
        ``"injection"``.
    config : dict
        Configuration dictionary with simulation parameters.  Missing
        keys are filled with defaults.
    pregenerated_path : str or Path, optional
        Path to an HDF5 file produced by :func:`create_dataset_hdf5`.
        If ``None``, maps are generated on-the-fly.
    transform : callable, optional
        An additional transform applied to the preprocessed map tensor
        before returning.
    augment : bool
        Whether to apply random rotation augmentation (typically enabled
        only for training).
    """

    def __init__(
        self,
        split: str,
        config: Optional[dict] = None,
        pregenerated_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()

        if split not in SPLIT_SEED_RANGES:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Supported: {list(SPLIT_SEED_RANGES.keys())}"
            )

        self.split = split
        self.config = _merge_config(config)
        self.transform = transform
        self.augment = augment

        self.nside: int = self.config["nside"]
        self.lmax: int = self.config["lmax"]
        self.npix: int = hp.nside2npix(self.nside)
        self.fwhm_arcmin: float = self.config["fwhm_arcmin"]
        self.signal_fraction: float = self.config["signal_fraction"]

        # Seed range for this split
        self.seed_start, self.seed_end = SPLIT_SEED_RANGES[split]
        self._n_samples: int = self.seed_end - self.seed_start + 1

        # For "null" split, signal fraction is 0 (no collisions)
        if split == "null":
            self.signal_fraction = 0.0
        elif split == "injection":
            self.signal_fraction = 1.0

        # Pre-generated HDF5 mode
        self._h5_file = None
        self._pregenerated_path = (
            Path(pregenerated_path) if pregenerated_path is not None else None
        )

        # Load mask
        self._mask = load_mask(
            self.config.get(
                "mask_path",
                "data/planck/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
            ),
            nside_target=self.nside,
        )

        # Pre-compute power spectrum (shared across all maps)
        self._cl: Optional[np.ndarray] = None

        logger.info(
            "CMBMapDataset: split=%s, n_samples=%d, augment=%s, "
            "signal_fraction=%.2f, pregenerated=%s",
            split,
            self._n_samples,
            augment,
            self.signal_fraction,
            self._pregenerated_path is not None,
        )

    @property
    def cl(self) -> np.ndarray:
        """Lazily compute and cache the theoretical power spectrum."""
        if self._cl is None:
            self._cl = compute_theoretical_cl(
                lmax=self.lmax,
                cache_path=self.config.get("cl_cache_path"),
            )
        return self._cl

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        """Return a single preprocessed map with label and metadata.

        Returns
        -------
        map_tensor : torch.Tensor, shape (1, npix), dtype float32
            Preprocessed and standardised map.
        label : int
            0 = null (no collision), 1 = signal (collision injected).
        metadata : dict
            Keys: ``"seed"``, ``"noise_type"``, ``"bubble_params"``
            (BubbleParams or None), ``"mu"``, ``"sigma"``.
        """
        if self._pregenerated_path is not None:
            return self._getitem_hdf5(idx)
        return self._getitem_onthefly(idx)

    # ------------------------------------------------------------------
    # On-the-fly generation
    # ------------------------------------------------------------------

    def _getitem_onthefly(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        """Generate a map on-the-fly for the given index."""
        seed = self.seed_start + idx
        rng = np.random.default_rng(seed)

        # Determine label (signal or null)
        label = int(rng.random() < self.signal_fraction)

        # Generate CMB realisation
        # Use a derived seed for the CMB so that noise gets a different seed
        cmb_seed = int(rng.integers(0, 2**31))
        cmb_map = generate_cmb_map(
            cl=self.cl,
            nside=self.nside,
            fwhm_arcmin=self.fwhm_arcmin,
            seed=cmb_seed,
            use_pixel_window=self.config.get("use_pixel_window", True),
        )

        # Inject bubble collision if signal
        bubble_params: Optional[BubbleParams] = None
        if label == 1:
            bubble_cfg = self.config.get("bubble", {})
            bubble_params = sample_bubble_params(
                rng=rng,
                theta_r_range_deg=tuple(
                    bubble_cfg.get("theta_r_range_deg", [5.0, 60.0])
                ),
                amplitude_range=tuple(
                    bubble_cfg.get("amplitude_range", [3e-5, 5e-4])
                ),
                kappa_values=tuple(
                    bubble_cfg.get("kappa_values", [2, 3, 4])
                ),
            )
            cmb_map = inject_bubble(
                cmb_map, bubble_params, nside=self.nside
            )

        # Add noise
        noise_type = self._choose_noise_type(rng)
        noise_seed = int(rng.integers(0, 2**31))
        noise_map = generate_noise(
            noise_type=noise_type,
            nside=self.nside,
            seed=noise_seed,
            sigma_uK=self.config.get("noise_sigma_uK", 45.0),
            ffp10_dir=self.config.get("ffp10_dir", "data/planck/ffp10/"),
        )
        raw_map = cmb_map + noise_map

        # Apply augmentation (random rotation before masking)
        if self.augment:
            euler = (
                rng.uniform(0, 2 * np.pi),
                np.arccos(rng.uniform(-1, 1)),
                rng.uniform(0, 2 * np.pi),
            )
            raw_map = rotate_map_alm(raw_map, self.nside, self.lmax, euler)

        # Preprocess: mask, remove monopole/dipole, standardise
        processed_map, mu, sigma = preprocess_map(raw_map, self._mask)

        # Build tensor: (1, npix)
        map_tensor = torch.from_numpy(processed_map).unsqueeze(0)

        if self.transform is not None:
            map_tensor = self.transform(map_tensor)

        metadata: dict[str, Any] = {
            "seed": seed,
            "noise_type": noise_type,
            "bubble_params": bubble_params,
            "mu": mu,
            "sigma": sigma,
        }
        return map_tensor, label, metadata

    def _choose_noise_type(self, rng: np.random.Generator) -> str:
        """Select a noise type according to the mixture probabilities."""
        noise_mix = self.config.get("noise_mix", DEFAULT_NOISE_MIX)
        types = list(noise_mix.keys())
        probs = np.array([noise_mix[t] for t in types], dtype=np.float64)
        probs /= probs.sum()
        return str(rng.choice(types, p=probs))

    # ------------------------------------------------------------------
    # Pre-generated HDF5 mode
    # ------------------------------------------------------------------

    def _getitem_hdf5(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        """Load a map from the pre-generated HDF5 file."""
        import h5py  # lazy import

        if self._h5_file is None:
            self._h5_file = h5py.File(str(self._pregenerated_path), "r")

        raw_map = self._h5_file["maps"][idx].astype(np.float32)
        label = int(self._h5_file["labels"][idx])
        seed = int(self._h5_file["seeds"][idx])
        noise_code = int(self._h5_file["noise_types"][idx])
        noise_type = NOISE_CODE_TO_TYPE.get(noise_code, "white")

        bp_arr = self._h5_file["bubble_params"][idx]
        if np.any(np.isnan(bp_arr)):
            bubble_params = None
        else:
            bubble_params = BubbleParams.from_array(bp_arr)

        # Preprocess (mask, monopole/dipole removal, standardise)
        processed_map, mu, sigma = preprocess_map(raw_map, self._mask)

        # Augmentation for HDF5 mode
        if self.augment:
            rng = np.random.default_rng(seed + idx)
            euler = (
                rng.uniform(0, 2 * np.pi),
                np.arccos(rng.uniform(-1, 1)),
                rng.uniform(0, 2 * np.pi),
            )
            processed_map = rotate_map_alm(
                processed_map, self.nside, self.lmax, euler
            )
            # Re-standardise after rotation
            valid = self._mask.astype(bool)
            vals = processed_map[valid]
            mu2 = float(np.mean(vals))
            sigma2 = float(np.std(vals))
            if sigma2 > 0:
                processed_map[valid] = (processed_map[valid] - mu2) / sigma2
            processed_map[~valid] = 0.0

        map_tensor = torch.from_numpy(processed_map).unsqueeze(0)

        if self.transform is not None:
            map_tensor = self.transform(map_tensor)

        metadata: dict[str, Any] = {
            "seed": seed,
            "noise_type": noise_type,
            "bubble_params": bubble_params,
            "mu": mu,
            "sigma": sigma,
        }
        return map_tensor, label, metadata

    def __del__(self) -> None:
        """Ensure the HDF5 file handle is closed on deletion."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# ============================================================================
# HDF5 pre-generation
# ============================================================================

def create_dataset_hdf5(
    output_path: Union[str, Path],
    n_samples: int,
    config: Optional[dict] = None,
    seed: int = 0,
    split: str = "train",
) -> Path:
    """Pre-generate a set of simulated CMB maps and save to HDF5.

    Parameters
    ----------
    output_path : str or Path
        Output HDF5 file path.
    n_samples : int
        Number of maps to generate.
    config : dict, optional
        Simulation configuration.  Missing keys filled with defaults.
    seed : int
        Base seed; each map uses ``seed + i`` as its unique seed.
    split : str
        Dataset split name (used for HDF5 metadata).

    Returns
    -------
    output_path : Path
        Path to the created HDF5 file.

    Notes
    -----
    HDF5 schema::

        /maps          (N, npix) float32  -- raw maps before preprocessing
        /labels        (N,)      int8     -- 0 or 1
        /seeds         (N,)      int64    -- random seeds used
        /noise_types   (N,)      int8     -- 0=white, 1=ffp10, 2=ffp10_full
        /bubble_params (N, 5)    float32  -- (theta_c, phi_c, theta_r, A, kappa)
    """
    import h5py  # lazy import

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _merge_config(config)
    nside = cfg["nside"]
    npix = hp.nside2npix(nside)
    signal_fraction = cfg.get("signal_fraction", 0.5)
    if split == "null":
        signal_fraction = 0.0
    elif split == "injection":
        signal_fraction = 1.0

    # Pre-compute the power spectrum
    cl = compute_theoretical_cl(
        lmax=cfg["lmax"],
        cache_path=cfg.get("cl_cache_path"),
    )

    logger.info(
        "Creating HDF5 dataset: %s, n=%d, split=%s", output_path, n_samples, split
    )

    with h5py.File(str(output_path), "w") as f:
        # Create datasets
        ds_maps = f.create_dataset(
            "maps", shape=(n_samples, npix), dtype="float32",
            chunks=(1, npix), compression="gzip", compression_opts=4,
        )
        ds_labels = f.create_dataset("labels", shape=(n_samples,), dtype="int8")
        ds_seeds = f.create_dataset("seeds", shape=(n_samples,), dtype="int64")
        ds_noise = f.create_dataset("noise_types", shape=(n_samples,), dtype="int8")
        ds_bp = f.create_dataset(
            "bubble_params", shape=(n_samples, 5), dtype="float32"
        )

        for i in range(n_samples):
            map_seed = seed + i
            rng = np.random.default_rng(map_seed)

            label = int(rng.random() < signal_fraction)

            # CMB
            cmb_seed = int(rng.integers(0, 2**31))
            cmb_map = generate_cmb_map(
                cl=cl,
                nside=nside,
                fwhm_arcmin=cfg["fwhm_arcmin"],
                seed=cmb_seed,
                use_pixel_window=cfg.get("use_pixel_window", True),
            )

            # Bubble
            bp_arr = np.full(5, np.nan, dtype=np.float32)
            if label == 1:
                bubble_cfg = cfg.get("bubble", {})
                bp = sample_bubble_params(
                    rng=rng,
                    theta_r_range_deg=tuple(
                        bubble_cfg.get("theta_r_range_deg", [5.0, 60.0])
                    ),
                    amplitude_range=tuple(
                        bubble_cfg.get("amplitude_range", [3e-5, 5e-4])
                    ),
                    kappa_values=tuple(
                        bubble_cfg.get("kappa_values", [2, 3, 4])
                    ),
                )
                cmb_map = inject_bubble(cmb_map, bp, nside=nside)
                bp_arr = bp.to_array()

            # Noise
            noise_mix = cfg.get("noise_mix", DEFAULT_NOISE_MIX)
            types_list = list(noise_mix.keys())
            probs = np.array([noise_mix[t] for t in types_list], dtype=np.float64)
            probs /= probs.sum()
            noise_type = str(rng.choice(types_list, p=probs))
            noise_seed = int(rng.integers(0, 2**31))
            noise_map = generate_noise(
                noise_type=noise_type,
                nside=nside,
                seed=noise_seed,
                sigma_uK=cfg.get("noise_sigma_uK", 45.0),
                ffp10_dir=cfg.get("ffp10_dir", "data/planck/ffp10/"),
            )

            raw_map = cmb_map + noise_map

            ds_maps[i] = raw_map
            ds_labels[i] = label
            ds_seeds[i] = map_seed
            ds_noise[i] = NOISE_TYPE_CODES.get(noise_type, 0)
            ds_bp[i] = bp_arr

            if (i + 1) % 1000 == 0 or i == n_samples - 1:
                logger.info("  Generated %d / %d maps", i + 1, n_samples)

        # Store metadata as HDF5 attributes
        f.attrs["nside"] = nside
        f.attrs["lmax"] = cfg["lmax"]
        f.attrs["fwhm_arcmin"] = cfg["fwhm_arcmin"]
        f.attrs["split"] = split
        f.attrs["creation_date"] = datetime.utcnow().isoformat()
        try:
            import yaml  # type: ignore
            config_str = yaml.dump(cfg)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        except ImportError:
            config_hash = "unknown"
        f.attrs["config_hash"] = config_hash

    logger.info("Dataset saved to %s", output_path)
    return output_path


# ============================================================================
# DataLoader factory
# ============================================================================

def _collate_fn(
    batch: list[tuple[torch.Tensor, int, dict]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Custom collate function that handles the metadata dict.

    Parameters
    ----------
    batch : list of (map_tensor, label, metadata)
        Items from ``CMBMapDataset.__getitem__``.

    Returns
    -------
    maps : torch.Tensor, shape (B, 1, npix)
    labels : torch.Tensor, shape (B,), dtype long
    metadata_list : list of dict
    """
    maps = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata_list = [item[2] for item in batch]
    return maps, labels, metadata_list


def get_data_loaders(
    config: Optional[dict] = None,
    splits: Optional[list[str]] = None,
    pregenerated_dir: Optional[Union[str, Path]] = None,
) -> dict[str, DataLoader]:
    """Create PyTorch DataLoaders for the requested splits.

    Parameters
    ----------
    config : dict, optional
        Simulation and training configuration.
    splits : list of str, optional
        Which splits to create.  Defaults to ``["train", "val", "test"]``.
    pregenerated_dir : str or Path, optional
        Directory containing pre-generated HDF5 files named
        ``simulations_{split}.h5``.  If ``None``, generation is on-the-fly.

    Returns
    -------
    loaders : dict
        Mapping from split name to ``torch.utils.data.DataLoader``.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    cfg = _merge_config(config)
    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)

    loaders: dict[str, DataLoader] = {}

    for split in splits:
        pregen_path = None
        if pregenerated_dir is not None:
            candidate = Path(pregenerated_dir) / f"simulations_{split}.h5"
            if candidate.exists():
                pregen_path = candidate

        augment = split == "train"

        dataset = CMBMapDataset(
            split=split,
            config=cfg,
            pregenerated_path=pregen_path,
            augment=augment,
        )

        shuffle = split == "train"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        loaders[split] = loader

    return loaders
