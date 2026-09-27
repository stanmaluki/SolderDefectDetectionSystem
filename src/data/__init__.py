"""Data generation and loading for SolSight."""
from .dataset import (
    DiscreteMultiScaleTrainLoader,
    SolderPatchDataset,
    get_fixed_tier_loader,
)
from .synthetic_generator import (
    generate_dataset_split,
    render_defect_bridging,
    render_defect_cold_joint,
    render_defect_solder_amount,
    render_defect_void,
    render_normal_joint,
)

__all__ = [
    "DiscreteMultiScaleTrainLoader",
    "SolderPatchDataset",
    "get_fixed_tier_loader",
    "generate_dataset_split",
    "render_normal_joint",
    "render_defect_void",
    "render_defect_bridging",
    "render_defect_cold_joint",
    "render_defect_solder_amount",
]
