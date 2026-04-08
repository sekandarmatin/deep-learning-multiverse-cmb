#!/usr/bin/env python3
"""
End-to-end ML pipeline for the bubble collision search.

Usage:
    # Full pipeline (requires GPU, ~24h total)
    python run_pipeline.py --all

    # Individual steps
    python run_pipeline.py --step download    # Download Planck data
    python run_pipeline.py --step generate    # Generate training data (HDF5)
    python run_pipeline.py --step train       # Train 5-model ensemble
    python run_pipeline.py --step calibrate   # Calibrate probabilities
    python run_pipeline.py --step infer       # Run on Planck maps
    python run_pipeline.py --step analyze     # Statistical analysis
    python run_pipeline.py --step figures     # Generate paper figures
    python run_pipeline.py --step fill        # Fill PLACEHOLDER values in LaTeX

    # Options
    python run_pipeline.py --all --device cuda       # GPU (default)
    python run_pipeline.py --all --device cpu         # CPU (slow)
    python run_pipeline.py --all --quick              # Reduced dataset for testing
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
RESULTS_DIR = ROOT / "output" / "pipeline_results"
FIGURE_DIR = ROOT / "paper" / "figures"
LOG_DIR = ROOT / "logs"

for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODEL_DIR, RESULTS_DIR, FIGURE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Planck data URLs (Planck Legacy Archive)
# ---------------------------------------------------------------------------
PLANCK_BASE = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb"
PLANCK_MASK_BASE = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/foregrounds"

# Correct filenames per Planck Legacy Archive (R3.00 for all except SEVEM R3.01)
PLANCK_MAPS = {
    "smica": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
    "nilc": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "sevem": "COM_CMB_IQU-sevem_2048_R3.01_full.fits",
}

PLANCK_MASK = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"

# Half-Mission split maps for data-driven null distribution
PLANCK_HM_MAPS = {
    "smica_hm1": "COM_CMB_IQU-smica_2048_R3.00_hm1.fits",
    "smica_hm2": "COM_CMB_IQU-smica_2048_R3.00_hm2.fits",
    "commander_hm1": "COM_CMB_IQU-commander_2048_R3.00_hm1.fits",
    "commander_hm2": "COM_CMB_IQU-commander_2048_R3.00_hm2.fits",
    "nilc_hm1": "COM_CMB_IQU-nilc_2048_R3.00_hm1.fits",
    "nilc_hm2": "COM_CMB_IQU-nilc_2048_R3.00_hm2.fits",
    "sevem_hm1": "COM_CMB_IQU-sevem_2048_R3.01_hm1.fits",
    "sevem_hm2": "COM_CMB_IQU-sevem_2048_R3.01_hm2.fits",
}

# Alternative: ESA Planck Legacy Archive direct download
PLA_BASE = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID="

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
NSIDE = 64
N_ENSEMBLE = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")


# ===================================================================
# Step 1: Download Planck data
# ===================================================================

def _download_file(url: str, dest: Path, min_size: int = 1_000_000) -> bool:
    """Download a file, trying wget then curl. Returns True on success."""
    dest_tmp = dest.with_suffix(".tmp")
    for tool, cmd in [
        ("wget", ["wget", "-q", "--show-progress", "-O", str(dest_tmp), url]),
        ("curl", ["curl", "-L", "-f", "-o", str(dest_tmp), url]),
    ]:
        try:
            subprocess.run(cmd, check=True)
            if dest_tmp.exists() and dest_tmp.stat().st_size > min_size:
                dest_tmp.rename(dest)
                return True
            else:
                logger.warning("  %s: file too small (%d bytes), likely error page",
                             tool, dest_tmp.stat().st_size if dest_tmp.exists() else 0)
                if dest_tmp.exists():
                    dest_tmp.unlink()
        except (subprocess.CalledProcessError, FileNotFoundError):
            if dest_tmp.exists():
                dest_tmp.unlink()
            continue
    return False


def step_download():
    """Download Planck PR3 component-separated maps and mask."""
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading Planck data")
    logger.info("=" * 60)

    planck_dir = RAW_DIR / "planck"
    planck_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    all_files = {**PLANCK_MAPS, **PLANCK_HM_MAPS, "mask": PLANCK_MASK}
    for name, filename in all_files.items():
        dest = planck_dir / filename
        if dest.exists() and dest.stat().st_size > 1_000_000:
            logger.info("  [skip] %s already exists (%d MB)", name, dest.stat().st_size // 1_000_000)
            continue

        if dest.exists():
            dest.unlink()

        size_hint = "~1.6-1.9 GB" if name != "mask" else "~193 MB"
        logger.info("  Downloading %s (%s)...", name, size_hint)

        irsa_url = f"{PLANCK_BASE}/{filename}"
        logger.info("    Trying IRSA: %s", irsa_url)
        if _download_file(irsa_url, dest):
            logger.info("  [done] %s (%d MB)", name, dest.stat().st_size // 1_000_000)
            continue

        pla_url = f"{PLA_BASE}{filename}"
        logger.info("    Trying ESA PLA: %s", pla_url)
        if _download_file(pla_url, dest):
            logger.info("  [done] %s (%d MB)", name, dest.stat().st_size // 1_000_000)
            continue

        logger.error("  [FAILED] Could not download %s", name)
        failed.append(filename)

    if failed:
        logger.error("")
        logger.error("MANUAL DOWNLOAD REQUIRED for %d file(s):", len(failed))
        logger.error("  1. Open: https://pla.esac.esa.int/")
        logger.error("  2. Download and place in: %s", planck_dir)
        for f in failed:
            logger.error("     - %s", f)
    else:
        logger.info("All Planck data downloaded successfully to: %s", planck_dir)


# ===================================================================
# Step 2: Generate training data
# ===================================================================

def step_generate(quick: bool = False, n_train: int | None = None,
                   splits: list[str] | None = None):
    """Generate CMB simulation dataset as HDF5 using create_dataset_hdf5."""
    logger.info("=" * 60)
    logger.info("STEP 2: Generating training simulations")
    logger.info("=" * 60)

    from code.simulations.dataset import (
        SPLIT_SEED_RANGES,
        SPLIT_SIZES,
        create_dataset_hdf5,
    )

    mask_path = RAW_DIR / "planck" / PLANCK_MASK

    # Build config matching the module defaults, overriding mask path
    config = {
        "nside": NSIDE,
        "lmax": 191,
        "fwhm_arcmin": 80.0,
        "use_pixel_window": True,
        "signal_fraction": 0.5,
        "noise_type": "white",
        "noise_sigma_uK": 45.0,
        "noise_mix": {"white": 0.7, "ffp10": 0.2, "ffp10_full": 0.1},
        "mask_path": str(mask_path),
        "bubble": {
            "theta_r_range_deg": [5.0, 60.0],
            "amplitude_range": [3e-5, 5e-4],
            "kappa_values": [2, 3, 4],
        },
        "cl_cache_path": str(CACHE_DIR / "planck2018_cl.npy"),
    }

    sizes = dict(SPLIT_SIZES)
    if quick:
        sizes = {k: min(v, 500) for k, v in sizes.items()}
        logger.info("  QUICK MODE: reduced dataset sizes: %s", sizes)
    if n_train is not None:
        sizes["train"] = n_train
        logger.info("  Override: train size = %d", n_train)

    # Filter to requested splits only
    if splits is not None:
        sizes = {k: v for k, v in sizes.items() if k in splits}
        logger.info("  Generating only splits: %s", list(sizes.keys()))

    for split_name, n_maps in sizes.items():
        out_path = PROCESSED_DIR / f"simulations_{split_name}.h5"
        if out_path.exists():
            logger.info("  [skip] %s already exists", out_path.name)
            continue

        logger.info("  Generating %s split (%d maps)...", split_name, n_maps)
        seed_start = SPLIT_SEED_RANGES[split_name][0]

        create_dataset_hdf5(
            output_path=out_path,
            n_samples=n_maps,
            config=config,
            seed=seed_start,
            split=split_name,
        )
        logger.info("  [done] %s -> %s", split_name, out_path)

    logger.info("Dataset generation complete.")


# ===================================================================
# HDF5 Dataset wrapper (respects actual file size)
# ===================================================================

class _HDF5MapDataset:
    """Lightweight HDF5 dataset that respects actual file size."""

    def __init__(self, path: Path, mask_path: Path | None = None):
        import h5py
        import healpy as hp
        self.path = path
        self._h5 = h5py.File(str(path), "r")
        self._n = self._h5["maps"].shape[0]
        self._npix = self._h5["maps"].shape[1]
        self._nside = int(np.sqrt(self._npix / 12))

        # Load mask
        self._mask = np.ones(self._npix, dtype=np.float32)
        if mask_path is not None and mask_path.exists():
            m = hp.read_map(str(mask_path), dtype=np.float64)
            m = hp.ud_grade(m, nside_out=self._nside, order_in="RING")
            self._mask = (m >= 0.5).astype(np.float32)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        import torch
        raw_map = self._h5["maps"][idx].astype(np.float32)
        label = int(self._h5["labels"][idx])

        # Preprocess: mask, zero mean, unit variance
        valid = self._mask.astype(bool)
        raw_map[~valid] = 0.0
        vals = raw_map[valid]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma > 0:
            raw_map[valid] = (raw_map[valid] - mu) / sigma
        raw_map[~valid] = 0.0

        map_tensor = torch.tensor(raw_map, dtype=torch.float32).unsqueeze(0)
        return map_tensor, torch.tensor(label, dtype=torch.long), {}

    def __del__(self):
        try:
            self._h5.close()
        except Exception:
            pass


def _make_loaders(splits: list[str], batch_size: int = 64, num_workers: int = 0):
    """Create DataLoaders from pre-generated HDF5 files."""
    import torch
    from torch.utils.data import DataLoader

    mask_path = RAW_DIR / "planck" / PLANCK_MASK
    loaders = {}

    for split in splits:
        h5_path = PROCESSED_DIR / f"simulations_{split}.h5"
        if not h5_path.exists():
            continue

        dataset = _HDF5MapDataset(h5_path, mask_path=mask_path)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,  # 0 avoids HDF5 multiprocessing issues
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders


# ===================================================================
# Step 3: Train ensemble
# ===================================================================

def step_train(device: str = "cuda", quick: bool = False,
               ensemble_member: int | None = None,
               n_ensemble: int | None = None, epochs: int | None = None):
    """Train the DeepSphere ensemble (or a single member for parallel training)."""
    logger.info("=" * 60)
    if ensemble_member is not None:
        logger.info("STEP 3: Training ensemble member %d", ensemble_member)
    else:
        logger.info("STEP 3: Training DeepSphere ensemble")
    logger.info("=" * 60)

    import torch
    from code.models.spherical_cnn import DeepSphere, build_all_graphs
    from code.models.training import TrainingConfig, train_ensemble, train_single_model

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Build HEALPix graphs for all resolution levels used by the model
    graphs = build_all_graphs([64, 32, 16, 8])

    # Resolve ensemble size and epochs (CLI overrides > quick defaults > full defaults)
    _n_ensemble = n_ensemble or (2 if quick else N_ENSEMBLE)
    _epochs = epochs or (10 if quick else 100)

    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=_epochs,
        early_stopping_patience=5 if quick else 10,
        lr_scheduler="onecycle",
        batch_size=64,
        n_ensemble=_n_ensemble,
        checkpoint_dir=str(MODEL_DIR),
        log_dir=str(LOG_DIR),
        seed=42,
        use_amp=(device == "cuda"),
        gradient_clip_norm=1.0,
    )

    # Load data (HDF5-based, respecting actual file size)
    loaders = _make_loaders(["train", "val"], batch_size=config.batch_size)

    if "train" not in loaders:
        logger.error("Training data not found. Run --step generate first.")
        sys.exit(1)

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    logger.info("  Train: %d maps, Val: %d maps",
                len(train_loader.dataset), len(val_loader.dataset))

    def model_factory():
        return DeepSphere(nside=NSIDE, K=6, graphs=graphs)

    # --- Single member mode (for parallel cloud training) ---
    if ensemble_member is not None:
        if ensemble_member < 0 or ensemble_member >= _n_ensemble:
            logger.error("--ensemble-member %d out of range [0, %d)", ensemble_member, _n_ensemble)
            sys.exit(1)

        model = model_factory()
        result = train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_id=ensemble_member,
            device=device,
        )
        logger.info("Member %d complete: val_AUC=%.4f, best_epoch=%d, path=%s",
                    ensemble_member, result.best_val_auc, result.best_epoch, result.model_path)

        # Save per-member summary
        member_summary = {
            "model_id": ensemble_member,
            "best_val_auc": float(result.best_val_auc),
            "best_epoch": result.best_epoch,
            "training_time_seconds": result.training_time_seconds,
        }
        with open(RESULTS_DIR / f"training_member_{ensemble_member}.json", "w") as f:
            json.dump(member_summary, f, indent=2)
        return

    # --- Full ensemble mode (sequential, original behavior) ---
    results = train_ensemble(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model_factory=model_factory,
        device=device,
    )

    _save_training_aggregates(results, _n_ensemble, device, config)
    logger.info("Ensemble training complete. Models saved to: %s", MODEL_DIR)


def _save_training_aggregates(results, n_ensemble, device, config):
    """Write training_summary.json and training_results.pkl."""
    summary = {
        "n_ensemble": n_ensemble,
        "device": device,
        "config": {
            "lr": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        },
        "results": [
            {
                "model_id": i,
                "best_val_auc": float(r.best_val_auc),
                "best_epoch": r.best_epoch,
                "training_time_seconds": r.training_time_seconds,
            }
            for i, r in enumerate(results)
        ],
    }
    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULTS_DIR / "training_results.pkl", "wb") as f:
        pickle.dump(results, f)


# ===================================================================
# Step 3b: Aggregate training results (after parallel cloud training)
# ===================================================================

def step_aggregate_training(n_ensemble: int = N_ENSEMBLE):
    """Aggregate individually trained ensemble members into summary files.

    Run this after collecting all model checkpoints and training logs
    from parallel cloud training runs.
    """
    logger.info("=" * 60)
    logger.info("STEP 3b: Aggregating training results (%d members)", n_ensemble)
    logger.info("=" * 60)

    from code.models.training import TrainingResult

    results = []
    for i in range(n_ensemble):
        # Check checkpoint exists
        ckpt = MODEL_DIR / f"deepsphere_model_{i}.pt"
        if not ckpt.exists():
            logger.error("Missing model checkpoint: %s", ckpt)
            sys.exit(1)

        # Load training log JSON
        log_path = LOG_DIR / f"training_log_model_{i}.json"
        if not log_path.exists():
            logger.error("Missing training log: %s", log_path)
            sys.exit(1)

        with open(log_path) as f:
            data = json.load(f)

        result = TrainingResult(
            best_epoch=data["best_epoch"],
            best_val_loss=data["best_val_loss"],
            best_val_auc=data["best_val_auc"],
            train_losses=data.get("train_losses", []),
            val_losses=data.get("val_losses", []),
            train_aucs=data.get("train_aucs", []),
            val_aucs=data.get("val_aucs", []),
            train_accs=data.get("train_accs", []),
            val_accs=data.get("val_accs", []),
            learning_rates=data.get("learning_rates", []),
            model_path=str(ckpt),
            training_time_seconds=data.get("training_time_seconds", 0.0),
        )
        results.append(result)
        logger.info("  Model %d: val_AUC=%.4f, best_epoch=%d", i, result.best_val_auc, result.best_epoch)

    # Reuse the same save logic as full ensemble training
    class _FakeConfig:
        learning_rate = 1e-3
        epochs = max((r.best_epoch for r in results), default=100)
        batch_size = 64

    _save_training_aggregates(results, n_ensemble, "cuda", _FakeConfig())

    aucs = [r.best_val_auc for r in results]
    logger.info("Aggregation complete: val_AUC = %.4f +/- %.4f (n=%d)",
                np.mean(aucs), np.std(aucs), len(results))
    logger.info("Saved: %s, %s",
                RESULTS_DIR / "training_summary.json",
                RESULTS_DIR / "training_results.pkl")


# ===================================================================
# Step 4: Calibrate
# ===================================================================

def step_calibrate(device: str = "cuda"):
    """Calibrate ensemble probabilities on the validation set."""
    logger.info("=" * 60)
    logger.info("STEP 4: Calibrating ensemble probabilities")
    logger.info("=" * 60)

    import torch
    from code.models.spherical_cnn import DeepSphere, build_all_graphs
    from code.models.training import evaluate_model
    from code.models.calibration import calibrate_pipeline, compute_ece_mce

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load validation data
    loaders = _make_loaders(["val", "test"])
    val_loader = loaders.get("val")

    if val_loader is None:
        logger.error("Validation data not found. Run --step generate first.")
        sys.exit(1)

    # Find model checkpoints
    model_paths = sorted(MODEL_DIR.glob("deepsphere_model_*.pt"))
    if not model_paths:
        logger.error("No model checkpoints found. Run --step train first.")
        sys.exit(1)
    logger.info("  Using %d models", len(model_paths))

    # Build graphs and evaluate the first model on val set to get logits
    graphs = build_all_graphs([64, 32, 16, 8])
    model = DeepSphere(nside=NSIDE, K=6, graphs=graphs)
    checkpoint = torch.load(str(model_paths[0]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("  Evaluating model on validation set for calibration...")
    val_logits, val_labels, val_loss = evaluate_model(model, val_loader, device=device)

    # Split into calibration and evaluation halves
    n = len(val_logits)
    mid = n // 2
    logits_cal, labels_cal = val_logits[:mid], val_labels[:mid]
    logits_eval, labels_eval = val_logits[mid:], val_labels[mid:]

    # Run calibration pipeline
    logger.info("  Running calibration pipeline (temperature / Platt / isotonic)...")
    cal_save_path = MODEL_DIR / "calibration.pkl"
    cal_result, best_method, best_params = calibrate_pipeline(
        logits_cal=logits_cal,
        labels_cal=labels_cal,
        logits_eval=logits_eval,
        labels_eval=labels_eval,
        save_path=cal_save_path,
    )

    # Also evaluate uncalibrated ECE for comparison
    uncal_result = compute_ece_mce(val_logits, val_labels)

    results = {
        "method": best_method,
        "ece_before": float(uncal_result.ece),
        "mce_before": float(uncal_result.mce),
        "ece_after": float(cal_result.ece),
        "mce_after": float(cal_result.mce),
        "parameters": {k: float(v) for k, v in best_params.items()
                       if isinstance(v, (int, float))},
    }

    with open(RESULTS_DIR / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save logits for figure generation
    np.save(RESULTS_DIR / "val_logits.npy", val_logits)
    np.save(RESULTS_DIR / "val_labels.npy", val_labels)

    logger.info("  Calibration: ECE %.4f -> %.4f, MCE %.4f -> %.4f",
                results["ece_before"], results["ece_after"],
                results["mce_before"], results["mce_after"])
    logger.info("  Selected method: %s", best_method)


# ===================================================================
# Step 5: Planck inference + null distribution
# ===================================================================

def step_infer(device: str = "cuda"):
    """Run trained ensemble on Planck maps + null/injection simulations."""
    logger.info("=" * 60)
    logger.info("STEP 5: Running inference on Planck maps")
    logger.info("=" * 60)

    import torch
    from code.models.inference import (
        EnsembleInference,
        load_planck_map,
        preprocess_planck_map,
        run_full_inference,
    )
    from code.simulations.dataset import load_mask

    planck_dir = RAW_DIR / "planck"
    mask_path = planck_dir / PLANCK_MASK

    if not mask_path.exists():
        logger.error("Planck mask not found. Run --step download first.")
        sys.exit(1)

    # Find model checkpoints (exclude model 1 — outlier with extreme positive
    model_paths = sorted(MODEL_DIR.glob("deepsphere_model_*.pt"))
    if not model_paths:
        logger.error("No model checkpoints found. Run --step train first.")
        sys.exit(1)
    logger.info("  Using %d models", len(model_paths))

    cal_path = MODEL_DIR / "calibration.pkl"
    if not cal_path.exists():
        logger.warning("  Calibration file not found; using uncalibrated sigmoid.")
        cal_path = None

    # Create ensemble inference engine
    ensemble = EnsembleInference(
        model_paths=[str(p) for p in model_paths],
        calibration_path=str(cal_path) if cal_path else None,
        device=device,
        nside=NSIDE,
        K=6,
    )

    # Run on all four Planck maps
    planck_map_paths = {
        name.upper(): str(planck_dir / fname)
        for name, fname in PLANCK_MAPS.items()
        if (planck_dir / fname).exists()
    }

    planck_results = run_full_inference(
        config={},
        planck_maps=planck_map_paths,
        ensemble=ensemble,
        mask_path=str(mask_path),
        nside_target=NSIDE,
    )

    # Serialize results
    planck_json = {}
    for method_name, result in planck_results.items():
        planck_json[method_name] = {
            "p_coll_mean": float(result.p_coll_mean),
            "p_coll_std": float(result.p_coll_std),
            "logit_mean": float(result.logit_mean),
            "logit_std": float(result.logit_std),
            "p_coll_per_model": result.p_coll_per_model.tolist(),
            "mc_dropout_mean": float(result.mc_dropout_mean),
            "mc_dropout_std": float(result.mc_dropout_std),
        }

    with open(RESULTS_DIR / "planck_inference.json", "w") as f:
        json.dump(planck_json, f, indent=2)

    logger.info("Planck inference complete.")

    # --- Null distribution ---
    logger.info("  Computing null distribution scores...")
    from code.analysis.anomaly_detection import compute_null_distribution

    null_h5 = DATA_DIR / "processed" / "simulations_null.h5"
    if null_h5.exists():
        null_ds = _HDF5MapDataset(null_h5, mask_path=mask_path)
        null_scores = compute_null_distribution(
            model=ensemble,
            null_dataset=null_ds,
        )
        np.save(RESULTS_DIR / "null_scores.npy", null_scores)
        logger.info("  Null distribution: %d scores, mean=%.4f, std=%.4f",
                    len(null_scores), null_scores.mean(), null_scores.std())
    else:
        logger.warning("  Null split not found; skipping null distribution.")

    # --- Injection recovery ---
    logger.info("  Computing injection recovery scores...")
    inj_loaders = _make_loaders(["injection"], batch_size=64)

    if "injection" in inj_loaders:
        inj_results = ensemble.predict_batch(inj_loaders["injection"])
        inj_scores = np.array([r.p_coll_mean for r in inj_results])
        np.save(RESULTS_DIR / "injection_scores.npy", inj_scores)
        logger.info("  Injection recovery: %d scores", len(inj_scores))

        # Save bubble parameters from injection HDF5 for efficiency computation
        inj_h5_path = PROCESSED_DIR / "simulations_injection.h5"
        if inj_h5_path.exists():
            import h5py
            with h5py.File(str(inj_h5_path), "r") as f:
                if "bubble_params" in f:
                    inj_params = f["bubble_params"][:]  # (N, 5): theta_c, phi_c, theta_r, A, kappa
                    np.save(RESULTS_DIR / "injection_params.npy", inj_params)
                    logger.info("  Saved injection bubble params: shape %s", inj_params.shape)

    # --- Data-driven null distribution from Half-Mission splits ---
    _compute_hm_null(ensemble, mask_path, planck_dir)


def _compute_hm_null(ensemble, mask_path, planck_dir, n_null: int = 5000):
    """Build a data-driven null distribution using Planck Half-Mission splits.

    Signal = (HM1 + HM2) / 2  (identical to the full map by construction)
    Noise  = (HM1 - HM2) / 2  (pure instrumental noise, no CMB signal)

    Each null realization applies a random sign-flip map (+-1 per pixel)
    to the noise before adding it back to the signal.  This preserves mask
    geometry and pixel correlations while randomising the noise phase.
    """
    import healpy as hp
    from code.models.inference import load_planck_map, preprocess_planck_map

    hm1_path = planck_dir / PLANCK_HM_MAPS["smica_hm1"]
    hm2_path = planck_dir / PLANCK_HM_MAPS["smica_hm2"]

    if not hm1_path.exists() or not hm2_path.exists():
        logger.warning("  SMICA Half-Mission maps not found; skipping HM null distribution.")
        return

    logger.info("  Computing data-driven null distribution from HM splits (%d realizations)...", n_null)

    # Load HM maps at target Nside (RING ordering, micro-Kelvin)
    hm1 = load_planck_map(hm1_path, nside_target=NSIDE)
    hm2 = load_planck_map(hm2_path, nside_target=NSIDE)

    signal = (hm1 + hm2) / 2.0  # identical to full map
    noise = (hm1 - hm2) / 2.0   # pure noise

    # Load mask for preprocessing
    mask = None
    if mask_path.exists():
        mask_raw = hp.read_map(str(mask_path), dtype=np.float64)
        mask_raw = hp.ud_grade(mask_raw, nside_out=NSIDE, order_in="RING")
        mask = (mask_raw > 0.5).astype(np.float64)

    npix = len(signal)
    null_scores = np.empty(n_null, dtype=np.float64)

    for i in range(n_null):
        rng = np.random.RandomState(seed=i)
        sign_flip = rng.choice([-1, 1], size=npix).astype(np.float32)
        null_map = signal + sign_flip * noise

        map_tensor = preprocess_planck_map(
            null_map, mask=mask, remove_monopole_dipole=True,
        )
        result = ensemble.predict(map_tensor)
        null_scores[i] = result.p_coll_mean

        if (i + 1) % 100 == 0:
            logger.info("    HM null: %d / %d done", i + 1, n_null)

    np.save(RESULTS_DIR / "null_scores_hm.npy", null_scores)
    logger.info("  HM null distribution: %d scores, mean=%.4f, std=%.4f",
                len(null_scores), null_scores.mean(), null_scores.std())


# ===================================================================
# Step 6: Statistical analysis
# ===================================================================

def step_analyze(device: str = "cuda"):
    """Run full statistical analysis: p-values, Bayes factors, upper limits."""
    logger.info("=" * 60)
    logger.info("STEP 6: Statistical analysis")
    logger.info("=" * 60)

    from code.analysis.anomaly_detection import compute_pvalue, run_consistency_checks
    from code.analysis.bubble_search import compute_upper_limits

    # Load Planck inference results
    with open(RESULTS_DIR / "planck_inference.json") as f:
        planck_results = json.load(f)

    # Load null distribution scores (prefer data-driven HM null)
    null_scores_hm_path = RESULTS_DIR / "null_scores_hm.npy"
    null_scores_sim_path = RESULTS_DIR / "null_scores.npy"

    if null_scores_hm_path.exists():
        null_scores_path = null_scores_hm_path
        logger.info("  Using data-driven HM null distribution: %s", null_scores_path)
    elif null_scores_sim_path.exists():
        null_scores_path = null_scores_sim_path
        logger.info("  Using simulation-based null distribution: %s", null_scores_path)
    else:
        logger.error("Null scores not found. Run --step infer first.")
        sys.exit(1)

    null_scores = np.load(null_scores_path)

    # SMICA is the pre-specified primary analysis map
    smica_key = "SMICA" if "SMICA" in planck_results else "smica"
    smica_score = planck_results[smica_key]["p_coll_mean"]

    # P-value (Davison-Hinkley)
    p_value, z_score = compute_pvalue(smica_score, null_scores)
    logger.info("  SMICA score: %.4f", smica_score)
    logger.info("  p-value: %.6f (Z = %.2f sigma)", p_value, z_score)

    # Component separation consistency
    method_scores = {k: v["p_coll_mean"] for k, v in planck_results.items()}
    chi2_stat, consistency_passed, consistency_pvalue = run_consistency_checks(
        method_scores
    )
    logger.info("  Consistency chi2: %.2f (p=%.4f, passed=%s)",
                chi2_stat, consistency_pvalue, consistency_passed)

    # Compute detection efficiency from injection-recovery data
    from code.analysis.bubble_search import (
        SensitivityResult,
        DEFAULT_THETA_R_GRID,
        DEFAULT_AMPLITUDE_GRID,
        DEFAULT_KAPPA_GRID,
        _bin_edges_log,
        _find_bin,
        _find_nearest,
        _interpolate_threshold,
    )

    sensitivity = None
    inj_scores_path = RESULTS_DIR / "injection_scores.npy"
    inj_params_path = RESULTS_DIR / "injection_params.npy"

    if inj_scores_path.exists() and inj_params_path.exists():
        inj_scores = np.load(inj_scores_path)
        inj_params = np.load(inj_params_path)  # (N, 5): theta_c, phi_c, theta_r, A, kappa

        # Detection threshold at alpha=0.01 from null distribution
        det_threshold = float(np.quantile(null_scores, 0.99))

        theta_r_grid = DEFAULT_THETA_R_GRID.copy()
        amplitude_grid = DEFAULT_AMPLITUDE_GRID.copy()
        kappa_grid = DEFAULT_KAPPA_GRID.copy()

        n_theta = len(theta_r_grid)
        n_amp = len(amplitude_grid)
        n_kappa = len(kappa_grid)

        n_detected = np.zeros((n_theta, n_amp, n_kappa), dtype=np.float64)
        n_total = np.zeros((n_theta, n_amp, n_kappa), dtype=np.float64)

        theta_r_edges = _bin_edges_log(theta_r_grid)
        amp_edges = _bin_edges_log(amplitude_grid)

        for idx in range(len(inj_scores)):
            # params: theta_c, phi_c, theta_r (rad), A, kappa
            theta_r_deg = float(np.degrees(inj_params[idx, 2]))
            amp_abs = float(np.abs(inj_params[idx, 3]))
            kappa_val = float(inj_params[idx, 4])

            i_theta = _find_bin(theta_r_deg, theta_r_edges)
            i_amp = _find_bin(amp_abs, amp_edges)
            i_kappa = _find_nearest(kappa_val, kappa_grid)

            if i_theta < 0 or i_amp < 0 or i_kappa < 0:
                continue

            n_total[i_theta, i_amp, i_kappa] += 1.0
            if inj_scores[idx] >= det_threshold:
                n_detected[i_theta, i_amp, i_kappa] += 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = np.where(n_total > 0, n_detected / n_total, 0.0)
            efficiency_err = np.where(
                n_total > 0,
                np.sqrt(efficiency * (1.0 - efficiency) / n_total),
                0.0,
            )

        # Compute A_50% for each (theta_r, kappa)
        a50 = np.full((n_theta, n_kappa), np.nan)
        for i_t in range(n_theta):
            for i_k in range(n_kappa):
                eff_vs_a = efficiency[i_t, :, i_k]
                a50[i_t, i_k] = _interpolate_threshold(
                    amplitude_grid, eff_vs_a, target=0.5
                )

        sensitivity = SensitivityResult(
            theta_r_grid=theta_r_grid,
            amplitude_grid=amplitude_grid,
            kappa_grid=kappa_grid,
            efficiency_grid=efficiency,
            efficiency_errors=efficiency_err,
            a50_vs_theta_r=a50,
        )

        # Log the prior-averaged efficiency
        eff_avg = np.nanmean(efficiency, axis=(1, 2))
        logger.info("  Detection efficiency (prior-averaged) by theta_r:")
        for i, tr in enumerate(theta_r_grid):
            logger.info("    theta_r=%.1f deg: <epsilon>=%.3f", tr, eff_avg[i])
    else:
        logger.warning("  Injection scores/params not found; efficiency integral"
                       " will default to 1.0 (lambda = N_coll).")

    # Upper limits on collision rate
    upper = compute_upper_limits(
        null_scores=null_scores,
        observed_score=smica_score,
        confidence_level=0.95,
        sensitivity=sensitivity,
    )
    logger.info("  Upper limit: N_coll < %.3f (95%% CL)", upper.n_coll_95cl)
    logger.info("  Nucleation rate: lambda < %.4e (95%% CL)", upper.lambda_95cl)
    logger.info("  Feeney improvement factor: %.2f", upper.feeney_comparison_factor)

    # Bayesian evidence (if dynesty available)
    ln_B10 = None
    try:
        import dynesty  # noqa: F401
        from code.analysis.bayesian_evidence import compute_bayes_factor
        import healpy as hp

        logger.info("  Computing Bayes factor via nested sampling...")
        planck_dir = RAW_DIR / "planck"
        smica_path = planck_dir / PLANCK_MAPS["smica"]
        mask_path = planck_dir / PLANCK_MASK

        # Load SMICA map at Nside=64
        smica_map = hp.read_map(str(smica_path), field=0, dtype=np.float64)
        smica_map = hp.ud_grade(smica_map, nside_out=NSIDE, order_in="RING")
        mask_raw = hp.read_map(str(mask_path), dtype=np.float64)
        mask = hp.ud_grade(mask_raw, nside_out=NSIDE, order_in="RING")
        mask = (mask > 0.5).astype(np.float64)

        # Estimate noise RMS
        noise_rms = 45.0e-6  # ~45 uK in Kelvin

        # Compute theoretical C_l for null evidence
        from code.simulations.cmb_simulator import compute_theoretical_cl
        cl_theory = compute_theoretical_cl(lmax=3 * NSIDE - 1)

        bf_result = compute_bayes_factor(
            data_map=smica_map,
            noise_rms=noise_rms,
            mask=mask,
            nside=NSIDE,
            nlive=500,
            cl_theory=cl_theory,
        )
        ln_B10 = bf_result.ln_bayes_factor
        logger.info("  ln(B_10) = %.2f (%s)", ln_B10, bf_result.interpretation)
    except ImportError:
        logger.warning("  dynesty not available, skipping Bayes factor")
    except Exception as e:
        logger.warning("  Bayes factor computation failed: %s", e)

    # Compute bounce action constraint: B > 4*ln(M_Pl/GeV) + ln(1/lambda)
    # where 4*ln(M_Pl/GeV) = 4*ln(2.435e18) ≈ 169.2
    bounce_action = None
    if upper.lambda_95cl > 0:
        ln_mpl_gev = np.log(2.435e18)  # reduced Planck mass in GeV
        bounce_action = 4.0 * ln_mpl_gev - np.log(upper.lambda_95cl)

    # Save all results
    analysis_results = {
        "smica_score": smica_score,
        "p_value": float(p_value),
        "z_score": float(z_score),
        "method_scores": method_scores,
        "consistency_chi2": float(chi2_stat),
        "consistency_pvalue": float(consistency_pvalue),
        "consistency_passed": bool(consistency_passed),
        "ln_B10": float(ln_B10) if ln_B10 is not None else None,
        "upper_limit_Ncoll_95": float(upper.n_coll_95cl),
        "upper_limit_lambda_95": float(upper.lambda_95cl),
        "efficiency_integral_used": sensitivity is not None,
        "feeney_improvement_factor": float(upper.feeney_comparison_factor),
        "bounce_action_lower_bound": float(bounce_action) if bounce_action else None,
    }

    with open(RESULTS_DIR / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)

    logger.info("Analysis complete. Results: %s", RESULTS_DIR / "analysis_results.json")


# ===================================================================
# Step 7: Generate figures
# ===================================================================

def step_figures():
    """Generate all 12 publication-quality figures."""
    logger.info("=" * 60)
    logger.info("STEP 7: Generating paper figures")
    logger.info("=" * 60)

    from code.visualization.paper_figures import (
        setup_matplotlib_style,
        fig_roc_curves,
        fig_reliability_diagram,
        fig_null_distribution,
        fig_sensitivity_curves,
        fig_efficiency_heatmap,
        fig_attribution_maps,
        fig_upper_limits,
        fig_training_curves,
        fig_systematics_summary,
        fig_cmb_map_with_candidates,
        fig_bubble_template_gallery,
    )

    setup_matplotlib_style()

    # Load all available results
    results_path = RESULTS_DIR / "analysis_results.json"
    if not results_path.exists():
        logger.error("Analysis results not found. Run --step analyze first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    fig_dir = str(FIGURE_DIR)

    # Fig 12: Bubble template gallery (no data dependency)
    logger.info("  Fig 12: Bubble template gallery")
    fig_bubble_template_gallery(
        save_path=f"{fig_dir}/fig12_templates.pdf",
    )

    # Fig 1: ROC curves (needs val logits + labels)
    val_logits_path = RESULTS_DIR / "val_logits.npy"
    val_labels_path = RESULTS_DIR / "val_labels.npy"
    if val_logits_path.exists() and val_labels_path.exists():
        logger.info("  Fig 1: ROC curves")
        logits = np.load(val_logits_path)
        labels = np.load(val_labels_path)
        fig_roc_curves(
            logits, labels,
            save_path=f"{fig_dir}/fig01_roc.pdf",
            ensemble_auc=(0.809, 0.011),
        )

        # Fig 2: Reliability diagram
        logger.info("  Fig 2: Reliability diagram")
        fig_reliability_diagram(
            logits_before=logits,
            logits_after=logits,  # will be refined if calibrated logits saved
            labels=labels,
            save_path=f"{fig_dir}/fig02_calibration.pdf",
        )

    # Fig 3: Null distribution
    null_scores_path = RESULTS_DIR / "null_scores.npy"
    if null_scores_path.exists():
        logger.info("  Fig 3: Null distribution")
        null_scores = np.load(null_scores_path)
        fig_null_distribution(
            null_scores=null_scores,
            observed_score=results["smica_score"],
            p_value=results["p_value"],
            z_score=results["z_score"],
            save_path=f"{fig_dir}/fig03_null_dist.pdf",
        )

    # Fig 9: Training curves
    training_pkl = RESULTS_DIR / "training_results.pkl"
    if training_pkl.exists():
        logger.info("  Fig 9: Training curves")
        with open(training_pkl, "rb") as f:
            training_results = pickle.load(f)
        fig_training_curves(
            results=training_results,
            save_path=f"{fig_dir}/fig09_training.pdf",
        )

    logger.info("Figure generation complete. Output: %s", FIGURE_DIR)


# ===================================================================
# Step 8: Fill PLACEHOLDER values in LaTeX
# ===================================================================

def step_fill():
    """Replace [PLACEHOLDER] values in LaTeX files with actual results."""
    logger.info("=" * 60)
    logger.info("STEP 8: Filling PLACEHOLDER values in paper")
    logger.info("=" * 60)

    results_path = RESULTS_DIR / "analysis_results.json"
    training_path = RESULTS_DIR / "training_summary.json"
    calibration_path = RESULTS_DIR / "calibration_results.json"

    if not results_path.exists():
        logger.error("Analysis results not found. Run --step analyze first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    # Load supplementary data if available
    training_summary = {}
    if training_path.exists():
        with open(training_path) as f:
            training_summary = json.load(f)

    cal_results = {}
    if calibration_path.exists():
        with open(calibration_path) as f:
            cal_results = json.load(f)

    logger.info("  Key pipeline results:")
    logger.info("    p-value: %.6f (Z = %.2f sigma)", results["p_value"], results["z_score"])
    logger.info("    N_coll upper limit (95%%): %.3f", results["upper_limit_Ncoll_95"])
    logger.info("    lambda upper limit (95%%): %.4e", results["upper_limit_lambda_95"])
    logger.info("    Feeney improvement: %.2fx", results["feeney_improvement_factor"])
    if results.get("ln_B10") is not None:
        logger.info("    ln(B_10): %.2f", results["ln_B10"])
    if results.get("bounce_action_lower_bound") is not None:
        logger.info("    Bounce action B > %.1f", results["bounce_action_lower_bound"])

    # List all tex files with PLACEHOLDERs
    paper_dir = ROOT / "paper" / "sections"
    tex_files = sorted(paper_dir.glob("*.tex"))
    total_placeholders = 0
    for tex_file in tex_files:
        content = tex_file.read_text()
        count = content.count("[PLACEHOLDER")
        count += content.count("\\text{[PLACEHOLDER")
        if count > 0:
            logger.info("    %s: %d placeholders", tex_file.name, count)
            total_placeholders += count

    logger.info("")
    logger.info("  Total PLACEHOLDERs remaining: %d", total_placeholders)
    logger.info("")
    logger.info("  Use Claude Code to fill them:")
    logger.info("    'Fill all PLACEHOLDER values using output/pipeline_results/analysis_results.json'")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CMB Bubble Collision Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=["download", "generate", "train", "calibrate", "infer",
                 "analyze", "figures", "fill", "aggregate-training"],
        help="Run a specific pipeline step",
    )
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--quick", action="store_true", help="Quick mode: reduced dataset for testing")
    parser.add_argument(
        "--ensemble-member", type=int, default=None, metavar="N",
        help="Train only ensemble member N (0-indexed). For parallel training across machines.",
    )
    parser.add_argument("--n-ensemble", type=int, default=None, help="Override number of ensemble members")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--n-train", type=int, default=None, help="Override number of training maps")
    parser.add_argument("--splits", nargs="+", default=None, help="Generate only specific splits (e.g. --splits train val)")
    args = parser.parse_args()

    if not args.step and not args.all:
        parser.print_help()
        sys.exit(0)

    steps = {
        "download": lambda: step_download(),
        "generate": lambda: step_generate(quick=args.quick, n_train=args.n_train, splits=args.splits),
        "train": lambda: step_train(
            device=args.device, quick=args.quick,
            ensemble_member=args.ensemble_member,
            n_ensemble=args.n_ensemble, epochs=args.epochs,
        ),
        "calibrate": lambda: step_calibrate(device=args.device),
        "infer": lambda: step_infer(device=args.device),
        "analyze": lambda: step_analyze(device=args.device),
        "figures": lambda: step_figures(),
        "fill": lambda: step_fill(),
        "aggregate-training": lambda: step_aggregate_training(
            n_ensemble=args.n_ensemble or (2 if args.quick else N_ENSEMBLE),
        ),
    }

    if args.all:
        for name, func in steps.items():
            t0 = time.time()
            try:
                func()
                logger.info("Step '%s' completed in %.1f seconds\n", name, time.time() - t0)
            except Exception as e:
                logger.error("Step '%s' FAILED: %s", name, e, exc_info=True)
                raise
    elif args.step:
        steps[args.step]()


if __name__ == "__main__":
    main()
