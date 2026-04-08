"""
Training pipeline for the DeepSphere ensemble.

Provides a complete training loop with:

- ``BCEWithLogitsLoss`` for binary classification
- Adam optimiser with weight decay
- OneCycleLR (or CosineAnnealingLR) learning-rate scheduler
- Early stopping on validation loss
- Model checkpointing (best validation AUC)
- Deep-ensemble training (N independent models, different seeds)
- GPU support with automatic mixed precision
- Logging of per-epoch metrics (loss, AUC, accuracy)

References
----------
- Lakshminarayanan et al. (2017), "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles"
- Architecture document, Section 2.6
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ===================================================================
# Configuration and result dataclasses
# ===================================================================

@dataclass
class TrainingConfig:
    """Training hyper-parameters.

    Attributes
    ----------
    learning_rate : float
        Peak learning rate for Adam.
    weight_decay : float
        L2 regularisation coefficient.
    epochs : int
        Maximum number of training epochs.
    early_stopping_patience : int
        Stop if validation loss has not improved for this many epochs.
    lr_scheduler : str
        ``"onecycle"`` or ``"cosine"``.
    batch_size : int
        Mini-batch size.
    n_ensemble : int
        Number of models in the deep ensemble.
    checkpoint_dir : str
        Directory to save model checkpoints.
    log_dir : str
        Directory for training logs (JSON).
    seed : int
        Base random seed (ensemble member *i* uses ``seed + i``).
    use_amp : bool
        Enable automatic mixed-precision training.
    gradient_clip_norm : float
        Max gradient norm for clipping (0 = disabled).
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    lr_scheduler: str = "onecycle"
    batch_size: int = 64
    n_ensemble: int = 5
    checkpoint_dir: str = "data/models/"
    log_dir: str = "logs/"
    seed: int = 42
    use_amp: bool = False
    gradient_clip_norm: float = 1.0


@dataclass
class TrainingResult:
    """Result from a single training run.

    Attributes
    ----------
    best_epoch : int
        Epoch with the best validation AUC.
    best_val_loss : float
        Validation loss at the best epoch.
    best_val_auc : float
        Validation AUC-ROC at the best epoch.
    train_losses : list of float
        Per-epoch training loss.
    val_losses : list of float
        Per-epoch validation loss.
    train_aucs : list of float
        Per-epoch training AUC-ROC.
    val_aucs : list of float
        Per-epoch validation AUC-ROC.
    train_accs : list of float
        Per-epoch training accuracy.
    val_accs : list of float
        Per-epoch validation accuracy.
    learning_rates : list of float
        Learning rate at the end of each epoch.
    model_path : str
        Path to the saved best-model checkpoint.
    training_time_seconds : float
        Wall-clock training time in seconds.
    """

    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_auc: float = 0.0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_aucs: list[float] = field(default_factory=list)
    val_aucs: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    model_path: str = ""
    training_time_seconds: float = 0.0


# ===================================================================
# Seed & device helpers
# ===================================================================

def _set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device(device: str) -> torch.device:
    """Resolve device string to ``torch.device``."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


# ===================================================================
# Single-epoch helpers
# ===================================================================

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    gradient_clip_norm: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    all_logits : np.ndarray, shape (N,)
    all_labels : np.ndarray, shape (N,)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        # Unpack: dataset may return (map, label) or (map, label, metadata)
        if len(batch) == 3:
            maps, labels, _ = batch
        else:
            maps, labels = batch[0], batch[1]

        maps = maps.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(maps)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits.squeeze(-1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(maps)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        # Step scheduler per batch for OneCycleLR
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        bs = maps.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    return avg_loss, np.concatenate(all_logits), np.concatenate(all_labels)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run validation.

    Returns
    -------
    avg_loss : float
    all_logits : np.ndarray
    all_labels : np.ndarray
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        if len(batch) == 3:
            maps, labels, _ = batch
        else:
            maps, labels = batch[0], batch[1]

        maps = maps.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(maps)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze(-1)
        loss = criterion(logits, labels)

        bs = maps.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    return avg_loss, np.concatenate(all_logits), np.concatenate(all_labels)


def _compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute AUC-ROC and accuracy from logits and labels.

    Parameters
    ----------
    logits : np.ndarray
    labels : np.ndarray

    Returns
    -------
    auc : float
    acc : float
    """
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels.astype(int), probs)
    except ValueError:
        # Single class present in labels
        auc = 0.5

    acc = accuracy_score(labels.astype(int), preds)
    return auc, acc


# ===================================================================
# Main training functions
# ===================================================================

def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_id: int = 0,
    device: str = "cuda",
    resume: bool = False,
) -> TrainingResult:
    """Train a single DeepSphere model.

    Uses ``BCEWithLogitsLoss``, Adam optimiser, learning-rate scheduler,
    early stopping on validation loss, and checkpoints the best model
    (by validation AUC).

    Parameters
    ----------
    model : torch.nn.Module
        Initialised DeepSphere model.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    config : TrainingConfig
        Training hyper-parameters.
    model_id : int
        Identifier for this model in the ensemble (affects seed offset
        via ``config.seed + model_id``).
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    result : TrainingResult
        Training metrics and path to best model checkpoint.
    """
    seed = config.seed + model_id
    _set_seed(seed)
    dev = _get_device(device)

    model = model.to(dev)

    # ----- Loss, optimiser, scheduler ---------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = config.epochs * len(train_loader)
    if config.lr_scheduler == "onecycle":
        scheduler: Any = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )
    elif config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )
    else:
        scheduler = None

    # ----- AMP scaler -------------------------------------------------
    scaler = (
        torch.cuda.amp.GradScaler() if config.use_amp and dev.type == "cuda"
        else None
    )

    # ----- Checkpoint directory ---------------------------------------
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"deepsphere_model_{model_id}.pt"

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ----- Training loop ----------------------------------------------
    result = TrainingResult()
    best_state: Optional[dict] = None
    patience_counter = 0
    start_epoch = 1
    t0 = time.time()

    # ----- Resume from checkpoint if requested ------------------------
    resume_ckpt = ckpt_dir / f"deepsphere_resume_{model_id}.pt"
    if resume and resume_ckpt.exists():
        ckpt = torch.load(str(resume_ckpt), map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        patience_counter = ckpt.get("patience_counter", 0)
        result.best_val_auc = ckpt.get("best_val_auc", 0.0)
        result.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        result.best_epoch = ckpt.get("best_epoch", 0)
        result.train_losses = ckpt.get("train_losses", [])
        result.val_losses = ckpt.get("val_losses", [])
        result.train_aucs = ckpt.get("train_aucs", [])
        result.val_aucs = ckpt.get("val_aucs", [])
        result.train_accs = ckpt.get("train_accs", [])
        result.val_accs = ckpt.get("val_accs", [])
        result.learning_rates = ckpt.get("learning_rates", [])
        if "best_model_state_dict" in ckpt:
            best_state = ckpt["best_model_state_dict"]
        logger.info(
            "[Model %d] Resumed from epoch %d (val_auc=%.4f, patience=%d)",
            model_id, start_epoch - 1, result.best_val_auc, patience_counter,
        )

    logger.info(
        "Training model %d (seed=%d, device=%s, lr=%g, epochs=%d)",
        model_id, seed, dev, config.learning_rate, config.epochs,
    )

    for epoch in range(start_epoch, config.epochs + 1):
        # Train
        train_loss, train_logits, train_labels = _train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, dev,
            scaler, config.gradient_clip_norm,
        )
        train_auc, train_acc = _compute_metrics(train_logits, train_labels)

        # Validate
        val_loss, val_logits, val_labels = _validate(
            model, val_loader, criterion, dev,
        )
        val_auc, val_acc = _compute_metrics(val_logits, val_labels)

        # Step scheduler per epoch for Cosine
        if scheduler is not None and config.lr_scheduler == "cosine":
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.train_aucs.append(train_auc)
        result.val_aucs.append(val_auc)
        result.train_accs.append(train_acc)
        result.val_accs.append(val_acc)
        result.learning_rates.append(current_lr)

        logger.info(
            "[Model %d] Epoch %3d/%d  "
            "train_loss=%.4f  val_loss=%.4f  "
            "train_auc=%.4f  val_auc=%.4f  "
            "train_acc=%.4f  val_acc=%.4f  lr=%.2e",
            model_id, epoch, config.epochs,
            train_loss, val_loss,
            train_auc, val_auc,
            train_acc, val_acc,
            current_lr,
        )

        # Check improvement
        if val_auc > result.best_val_auc:
            result.best_val_auc = val_auc
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Save resume checkpoint every epoch
        resume_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "patience_counter": patience_counter,
            "best_val_auc": result.best_val_auc,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "train_aucs": result.train_aucs,
            "val_aucs": result.val_aucs,
            "train_accs": result.train_accs,
            "val_accs": result.val_accs,
            "learning_rates": result.learning_rates,
            "config": asdict(config),
            "model_id": model_id,
            "seed": seed,
        }
        if scheduler is not None:
            resume_state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            resume_state["scaler_state_dict"] = scaler.state_dict()
        if best_state is not None:
            resume_state["best_model_state_dict"] = best_state
        torch.save(resume_state, ckpt_dir / f"deepsphere_resume_{model_id}.pt")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(
                "[Model %d] Early stopping at epoch %d (patience=%d)",
                model_id, epoch, config.early_stopping_patience,
            )
            break

    # ----- Save best model --------------------------------------------
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "config": asdict(config),
                "model_id": model_id,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "best_val_auc": result.best_val_auc,
            },
            ckpt_path,
        )
        result.model_path = str(ckpt_path)
        logger.info(
            "[Model %d] Saved best checkpoint to %s (epoch %d, val_auc=%.4f)",
            model_id, ckpt_path, result.best_epoch, result.best_val_auc,
        )

    result.training_time_seconds = time.time() - t0

    # ----- Save training log ------------------------------------------
    log_path = log_dir / f"training_log_model_{model_id}.json"
    with open(log_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def train_ensemble(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_factory: Callable[[], nn.Module],
    device: str = "cuda",
) -> list[TrainingResult]:
    """Train a deep ensemble of N models with different random seeds.

    Parameters
    ----------
    config : TrainingConfig
        Training hyper-parameters.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    model_factory : callable
        Function that returns a new, initialised DeepSphere model.
        Called once per ensemble member.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    results : list of TrainingResult
        One per ensemble member (length ``config.n_ensemble``).
    """
    results: list[TrainingResult] = []

    for i in range(config.n_ensemble):
        logger.info(
            "===== Training ensemble member %d / %d =====",
            i + 1, config.n_ensemble,
        )
        model = model_factory()
        result = train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_id=i,
            device=device,
        )
        results.append(result)

        # Free GPU memory between ensemble members
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    aucs = [r.best_val_auc for r in results]
    logger.info(
        "Ensemble training complete: val_AUC = %.4f +/- %.4f  (min=%.4f, max=%.4f)",
        np.mean(aucs), np.std(aucs), np.min(aucs), np.max(aucs),
    )

    return results


# ===================================================================
# Evaluation
# ===================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model (will be set to eval mode).
    data_loader : DataLoader
        Evaluation data.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    logits : np.ndarray, shape ``(N,)``
        Raw model output (pre-sigmoid).
    labels : np.ndarray, shape ``(N,)``
        Ground-truth labels.
    loss : float
        Mean BCE loss over the dataset.
    """
    dev = _get_device(device)
    model = model.to(dev)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    running_loss = 0.0
    n_samples = 0

    for batch in data_loader:
        if len(batch) == 3:
            maps, labels, _ = batch
        else:
            maps, labels = batch[0], batch[1]

        maps = maps.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True).float()

        logits = model(maps)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.squeeze(-1)
        loss = criterion(logits, labels)

        bs = maps.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    return np.concatenate(all_logits), np.concatenate(all_labels), avg_loss


def compute_overfitting_gap(
    train_loss: float,
    val_loss: float,
    threshold: float = 0.05,
) -> tuple[float, bool]:
    """Compute the over-fitting gap.

    Parameters
    ----------
    train_loss : float
    val_loss : float
    threshold : float
        Flag if ``gap > threshold`` (Statistical Framework Section 8.1).

    Returns
    -------
    gap : float
        ``val_loss - train_loss``.
    flagged : bool
        ``True`` if the gap exceeds the threshold.
    """
    gap = val_loss - train_loss
    flagged = gap > threshold
    if flagged:
        logger.warning(
            "Overfitting gap %.4f exceeds threshold %.4f", gap, threshold,
        )
    return gap, flagged
