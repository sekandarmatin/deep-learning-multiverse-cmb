"""
Simulation pipeline for CMB temperature maps with bubble collision injection.

This package provides end-to-end simulation capabilities:

- **cmb_simulator** -- Gaussian CMB map generation from Planck 2018 best-fit
  power spectra.
- **bubble_simulator** -- Parametric bubble collision template generation
  and signal injection.
- **noise_simulator** -- Planck-like instrumental noise (white or FFP10).
- **dataset** -- PyTorch Dataset / DataLoader wrappers with preprocessing,
  augmentation, and HDF5 storage.

Example
-------
>>> from code.simulations import (
...     compute_theoretical_cl,
...     generate_cmb_map,
...     BubbleParams,
...     sample_bubble_params,
...     inject_bubble,
...     generate_noise,
...     CMBMapDataset,
...     get_data_loaders,
... )
"""

from .bubble_simulator import (
    BubbleParams,
    generate_bubble_on_grid,
    generate_bubble_template,
    inject_bubble,
    sample_bubble_params,
)
from .cmb_simulator import (
    PLANCK2018_BESTFIT,
    compute_theoretical_cl,
    generate_cmb_alm,
    generate_cmb_map,
)
from .dataset import (
    CMBMapDataset,
    build_healpix_graph,
    create_dataset_hdf5,
    get_data_loaders,
    load_mask,
    preprocess_map,
    rotate_map_alm,
)
from .noise_simulator import (
    estimate_noise_rms,
    generate_noise,
    generate_white_noise,
    load_ffp10_noise,
)

__all__ = [
    # cmb_simulator
    "PLANCK2018_BESTFIT",
    "compute_theoretical_cl",
    "generate_cmb_alm",
    "generate_cmb_map",
    # bubble_simulator
    "BubbleParams",
    "sample_bubble_params",
    "generate_bubble_template",
    "generate_bubble_on_grid",
    "inject_bubble",
    # noise_simulator
    "generate_white_noise",
    "load_ffp10_noise",
    "generate_noise",
    "estimate_noise_rms",
    # dataset
    "CMBMapDataset",
    "create_dataset_hdf5",
    "build_healpix_graph",
    "get_data_loaders",
    "load_mask",
    "preprocess_map",
    "rotate_map_alm",
]
