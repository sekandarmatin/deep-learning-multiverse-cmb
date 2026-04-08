"""
Visualisation modules for the CMB bubble collision paper.

Modules
-------
paper_figures
    Publication-quality figures (12 figures) following PRD style
    conventions.  All figures save to PDF and optionally PNG.
"""

from .paper_figures import (
    fig_attribution_maps,
    fig_bubble_template_gallery,
    fig_cmb_map_with_candidates,
    fig_efficiency_heatmap,
    fig_null_distribution,
    fig_posterior_corner,
    fig_reliability_diagram,
    fig_roc_curves,
    fig_sensitivity_curves,
    fig_systematics_summary,
    fig_training_curves,
    fig_upper_limits,
    setup_matplotlib_style,
)

__all__ = [
    "setup_matplotlib_style",
    "fig_roc_curves",
    "fig_reliability_diagram",
    "fig_null_distribution",
    "fig_sensitivity_curves",
    "fig_efficiency_heatmap",
    "fig_attribution_maps",
    "fig_posterior_corner",
    "fig_upper_limits",
    "fig_training_curves",
    "fig_systematics_summary",
    "fig_cmb_map_with_candidates",
    "fig_bubble_template_gallery",
]
