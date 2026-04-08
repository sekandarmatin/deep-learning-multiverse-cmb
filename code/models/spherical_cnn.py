"""
DeepSphere: Graph-based spherical CNN on HEALPix grid.

Implements the DeepSphere architecture for binary classification of CMB maps
(bubble collision vs. no collision) using Chebyshev graph convolutions on the
HEALPix sphere, following the specification in the theoretical framework
(Section 4.4) and architecture document (Section 2.5).

Architecture
------------
    Layer 1: GraphConv(K=20, F=32)  + BN + ReLU   [Nside=64, 49152 px]
    Layer 2: GraphConv(K=20, F=32)  + BN + ReLU   [Nside=64]
    Pool:    HEALPix avg pool                      [Nside=32, 12288 px]
    Layer 3: GraphConv(K=20, F=64)  + BN + ReLU   [Nside=32]
    Layer 4: GraphConv(K=20, F=64)  + BN + ReLU   [Nside=32]
    Pool:    HEALPix avg pool                      [Nside=16, 3072 px]
    Layer 5: GraphConv(K=20, F=128) + BN + ReLU   [Nside=16]
    Layer 6: GraphConv(K=20, F=128) + BN + ReLU   [Nside=16]
    Pool:    HEALPix avg pool                      [Nside=8, 768 px]
    Layer 7: GraphConv(K=20, F=256) + BN + ReLU   [Nside=8]
    GAP:     Global Average Pooling                [256-dim]
    FC:      256 -> 128 + ReLU + Dropout(0.5)
    FC:      128 -> 1   (classification head)
    FC:      128 -> 5   (regression head, optional)

References
----------
- Perraudin et al. (2019), "DeepSphere: Efficient spherical CNN [...]"
- Defferrard et al. (2020), "DeepSphere: a graph-based spherical CNN"
- Gobbetti & Kleban (2012), "Analyzing cosmic bubble collisions"
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default architecture constants (from architecture document Table 2.5)
# ---------------------------------------------------------------------------
DEFAULT_CHANNELS: list[int] = [1, 32, 32, 64, 64, 128, 128, 256]
DEFAULT_POOLING_NSIDES: list[int] = [64, 64, 32, 32, 16, 16, 8]
DEFAULT_FC_DIMS: list[int] = [256, 128]
DEFAULT_K: int = 6
DEFAULT_DROPOUT: float = 0.5
DEFAULT_NSIDE: int = 64
REGRESSION_OUTPUT_DIM: int = 5  # (theta_c, phi_c, theta_r, A, kappa)


# ===================================================================
# Graph construction utilities
# ===================================================================

def build_healpix_graph(nside: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the HEALPix graph structure for a given Nside.

    Each pixel is connected to its nearest neighbours (8 for most pixels,
    7 for some corner pixels). Edge weights are set to
    ``1 / angular_distance`` between pixel centres.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (must be a power of 2).

    Returns
    -------
    edge_index : torch.LongTensor, shape (2, n_edges)
        COO-format edge indices.
    edge_weight : torch.FloatTensor, shape (n_edges,)
        Edge weights (inverse angular distance in radians).
    """
    npix = hp.nside2npix(nside)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    # Pre-compute all pixel vectors for angular distance calculation
    pixel_vectors = np.array(hp.pix2vec(nside, np.arange(npix))).T  # (npix, 3)

    for ipix in range(npix):
        neighbours = hp.get_all_neighbours(nside, ipix)
        # hp.get_all_neighbours returns -1 for missing neighbours
        neighbours = neighbours[neighbours >= 0]

        for jpix in neighbours:
            sources.append(ipix)
            targets.append(jpix)

            # Angular distance via dot product of unit vectors
            dot = np.clip(
                np.dot(pixel_vectors[ipix], pixel_vectors[jpix]),
                -1.0,
                1.0,
            )
            ang_dist = np.arccos(dot)
            # Avoid division by zero for coincident points
            weight = 1.0 / max(ang_dist, 1e-12)
            weights.append(weight)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    logger.debug(
        "Built HEALPix graph: Nside=%d, npix=%d, edges=%d",
        nside, npix, edge_index.shape[1],
    )
    return edge_index, edge_weight


def build_all_graphs(
    nsides: list[int] | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build and cache graph structures for all resolution levels.

    Parameters
    ----------
    nsides : list of int, optional
        Resolution levels to build graphs for.
        Defaults to ``[64, 32, 16, 8]``.

    Returns
    -------
    graphs : dict
        Mapping ``Nside -> (edge_index, edge_weight)``.
    """
    if nsides is None:
        nsides = [64, 32, 16, 8]

    graphs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for ns in nsides:
        logger.info("Building HEALPix graph for Nside=%d ...", ns)
        graphs[ns] = build_healpix_graph(ns)
    return graphs


# ===================================================================
# HEALPix pooling
# ===================================================================

class HealpixPool(nn.Module):
    """HEALPix hierarchical pooling (Nside -> Nside/2).

    Exploits the HEALPix **nested** ordering hierarchy: every group of
    4 consecutive child pixels in nested ordering shares the same parent
    pixel at the next coarser resolution.  This layer averages (or
    max-pools) each group of 4.

    Parameters
    ----------
    mode : str
        ``"avg"`` for average pooling, ``"max"`` for max pooling.
    """

    def __init__(self, mode: str = "avg") -> None:
        super().__init__()
        if mode not in ("avg", "max"):
            raise ValueError(f"Pooling mode must be 'avg' or 'max', got '{mode}'")
        self.mode = mode

    def forward(self, x: torch.Tensor, nside: int) -> torch.Tensor:
        """Pool from Nside to Nside/2.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, N_pix)``
            Input feature map in **nested** HEALPix ordering.
        nside : int
            Current Nside (must be a power of 2 and >= 2).

        Returns
        -------
        x_pooled : torch.Tensor, shape ``(B, C, N_pix // 4)``
        """
        if nside < 2:
            raise ValueError(f"Cannot pool below Nside=1, got Nside={nside}")

        B, C, N = x.shape
        expected_npix = 12 * nside * nside
        if N != expected_npix:
            raise ValueError(
                f"Expected {expected_npix} pixels for Nside={nside}, got {N}"
            )

        # Reshape: group every 4 consecutive pixels (nested ordering)
        x = x.view(B, C, N // 4, 4)

        if self.mode == "avg":
            x_pooled = x.mean(dim=-1)
        else:  # max
            x_pooled = x.max(dim=-1).values

        return x_pooled

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


# ===================================================================
# Graph convolution block
# ===================================================================

class SphericalGraphConvBlock(nn.Module):
    """Single graph convolution block: ChebConv + BatchNorm + ReLU.

    Uses Chebyshev polynomial spectral graph convolutions from
    ``torch_geometric.nn.ChebConv``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    K : int
        Chebyshev filter order.
    edge_index : torch.Tensor
        Graph edge indices for this resolution level, shape ``(2, E)``.
    edge_weight : torch.Tensor or None
        Graph edge weights, shape ``(E,)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.conv = ChebConv(in_channels, out_channels, K=K)
        self.bn = nn.BatchNorm1d(out_channels)

        # Store graph topology as buffers (move with .to(device))
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C_in, N_pix)``

        Returns
        -------
        out : torch.Tensor, shape ``(B, C_out, N_pix)``
        """
        B, C_in, N = x.shape

        # ChebConv expects (num_nodes, in_channels) per graph.
        # We process the batch by stacking all graphs along the node
        # dimension and offsetting edge indices accordingly.
        # Reshape: (B, C, N) -> (B*N, C)
        x_flat = x.permute(0, 2, 1).reshape(B * N, C_in)

        # Build batched edge index: offset each graph's edges by i*N
        edge_index_batched = self._batch_edge_index(B, N)
        edge_weight_batched = (
            self.edge_weight.repeat(B)
            if self.edge_weight is not None
            else None
        )

        # Apply graph convolution
        out = self.conv(
            x_flat,
            edge_index_batched,
            edge_weight=edge_weight_batched,
        )

        # BatchNorm: (B*N, C_out) -> transpose for BN -> back
        out = self.bn(out.view(B, N, -1).permute(0, 2, 1).contiguous().view(-1, out.shape[-1]))

        # Reshape back and apply ReLU
        out = out.view(B, N, -1).permute(0, 2, 1)
        out = F.relu(out, inplace=True)

        return out

    def _batch_edge_index(self, batch_size: int, n_nodes: int) -> torch.Tensor:
        """Create batched edge index by offsetting node indices.

        Parameters
        ----------
        batch_size : int
            Number of graphs in the batch.
        n_nodes : int
            Number of nodes per graph.

        Returns
        -------
        batched_edge_index : torch.Tensor, shape ``(2, E * batch_size)``
        """
        if batch_size == 1:
            return self.edge_index

        offsets = torch.arange(
            batch_size, device=self.edge_index.device
        ) * n_nodes
        # (B, 1) + (2, E) -> broadcast to (B, 2, E) -> reshape (2, B*E)
        batched = self.edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
        batched = batched.permute(1, 0, 2).reshape(2, -1)
        return batched


# ===================================================================
# Main DeepSphere model
# ===================================================================

class DeepSphere(nn.Module):
    """DeepSphere: Graph-based spherical CNN on the HEALPix grid.

    Architecture follows the theoretical framework Section 4.4 and
    architecture document Section 2.5.  The final sigmoid is **not**
    included in the model; ``BCEWithLogitsLoss`` is used during
    training for numerical stability.

    Parameters
    ----------
    nside : int
        Input HEALPix resolution (default 64).
    K : int
        Chebyshev polynomial order for all graph conv layers.
    channels : list of int or None
        Channel dimensions ``[in, L1, L2, L3, L4, L5, L6, L7]``.
        Defaults to ``[1, 32, 32, 64, 64, 128, 128, 256]``.
    pooling_nsides : list of int or None
        Nside **after** each conv layer (length = 7).
        Pooling is applied whenever this value drops.
        Defaults to ``[64, 64, 32, 32, 16, 16, 8]``.
    fc_dims : list of int or None
        Fully-connected hidden dimensions.
        Defaults to ``[256, 128]``.
    dropout : float
        Dropout rate for FC layers.
    graphs : dict or None
        Pre-built graph structures ``{Nside: (edge_index, edge_weight)}``.
        If ``None``, graphs are built on-the-fly.
    pool_mode : str
        ``"avg"`` or ``"max"`` pooling mode.
    enable_regression : bool
        If ``True``, add a regression head that predicts the 5 bubble
        collision parameters ``(theta_c, phi_c, theta_r, A, kappa)``.
    """

    def __init__(
        self,
        nside: int = DEFAULT_NSIDE,
        K: int = DEFAULT_K,
        channels: Optional[list[int]] = None,
        pooling_nsides: Optional[list[int]] = None,
        fc_dims: Optional[list[int]] = None,
        dropout: float = DEFAULT_DROPOUT,
        graphs: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
        pool_mode: str = "avg",
        enable_regression: bool = False,
    ) -> None:
        super().__init__()

        # ----- Store hyper-parameters ---------------------------------
        self.nside = nside
        self.K = K
        self.channels = channels if channels is not None else list(DEFAULT_CHANNELS)
        self.pooling_nsides = (
            pooling_nsides
            if pooling_nsides is not None
            else list(DEFAULT_POOLING_NSIDES)
        )
        self.fc_dims = fc_dims if fc_dims is not None else list(DEFAULT_FC_DIMS)
        self.dropout_rate = dropout
        self.enable_regression = enable_regression
        self.pool_mode = pool_mode

        n_conv = len(self.channels) - 1  # 7 conv layers
        if len(self.pooling_nsides) != n_conv:
            raise ValueError(
                f"pooling_nsides length ({len(self.pooling_nsides)}) must "
                f"equal the number of conv layers ({n_conv})."
            )

        # ----- Build / retrieve graph structures ----------------------
        unique_nsides = sorted(set([nside] + self.pooling_nsides), reverse=True)
        if graphs is None:
            graphs = build_all_graphs(unique_nsides)
        self._graphs = graphs

        # ----- Graph convolution blocks -------------------------------
        self.conv_blocks = nn.ModuleList()
        current_nside = nside
        for i in range(n_conv):
            # The resolution for this conv layer
            layer_nside = current_nside
            ei, ew = self._graphs[layer_nside]
            block = SphericalGraphConvBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1],
                K=K,
                edge_index=ei,
                edge_weight=ew,
            )
            self.conv_blocks.append(block)

            # Pool if the next resolution is lower
            target_nside = self.pooling_nsides[i]
            while current_nside > target_nside:
                current_nside = current_nside // 2

        # ----- Pooling layer ------------------------------------------
        self.pool = HealpixPool(mode=pool_mode)

        # ----- Fully-connected classification head --------------------
        fc_layers: list[nn.Module] = []
        in_dim = self.channels[-1]
        for out_dim in self.fc_dims[1:]:  # skip first (== channels[-1])
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        self.fc = nn.Sequential(*fc_layers)
        self.classifier = nn.Linear(in_dim, 1)

        # ----- Optional regression head --------------------------------
        if enable_regression:
            self.regressor = nn.Sequential(
                nn.Linear(self.channels[-1], 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, REGRESSION_OUTPUT_DIM),
            )
        else:
            self.regressor = None

        logger.info(
            "DeepSphere initialised: Nside=%d, K=%d, params=%s, channels=%s",
            nside, K, f"{self.count_parameters(self):,}", self.channels,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning raw logit (pre-sigmoid).

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 1, N_pix)``
            Input HEALPix map in nested ordering.

        Returns
        -------
        logit : torch.Tensor, shape ``(B, 1)``
            Classification logit (if ``enable_regression=False``).
        logit, params : tuple of torch.Tensor
            ``(B, 1)`` logit and ``(B, 5)`` parameter estimates
            (if ``enable_regression=True``).
        """
        current_nside = self.nside

        for i, block in enumerate(self.conv_blocks):
            x = block(x)

            # Pool if target Nside is lower than current Nside
            target_nside = self.pooling_nsides[i]
            while current_nside > target_nside:
                x = self.pool(x, current_nside)
                current_nside = current_nside // 2

        # Global Average Pooling: (B, C, N) -> (B, C)
        gap = x.mean(dim=-1)

        # Classification head
        fc_out = self.fc(gap)
        logit = self.classifier(fc_out)

        if self.enable_regression and self.regressor is not None:
            params = self.regressor(gap)
            return logit, params

        return logit

    # ------------------------------------------------------------------
    # Feature maps for Grad-CAM
    # ------------------------------------------------------------------

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate feature maps for Grad-CAM attribution.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, 1, N_pix)``

        Returns
        -------
        features : list of torch.Tensor
            Feature maps after each convolutional block.
        """
        features: list[torch.Tensor] = []
        current_nside = self.nside

        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            features.append(x)

            target_nside = self.pooling_nsides[i]
            while current_nside > target_nside:
                x = self.pool(x, current_nside)
                current_nside = current_nside // 2

        return features

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total trainable parameters.

        Parameters
        ----------
        model : torch.nn.Module

        Returns
        -------
        n_params : int
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"nside={self.nside}, K={self.K}, "
            f"channels={self.channels}, "
            f"dropout={self.dropout_rate}, "
            f"regression={self.enable_regression}"
        )
