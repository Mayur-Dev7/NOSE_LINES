"""
Canonical Shared Data Contract for Cattle Muzzle Biometric Reference Templates
=============================================================================
Defines the schema, validators, and serialization helpers for multi-image
consolidated reference templates.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np

SCHEMA_VERSION = "1.0"


def robust_median_and_mad(values: List[float]) -> Tuple[float, float]:
    """
    Computes sample median and Median Absolute Deviation (MAD):
    MAD = median(|x_i - median(X)|)
    Normalized MAD scale estimate for normal distribution: MAD * 1.4826
    """
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def calculate_stability(values: List[float], med: Optional[float] = None, mad: Optional[float] = None) -> float:
    """
    Stability metric in [0.0, 1.0].
    Higher stability = lower dispersion relative to magnitude.
    Formula: 1.0 / (1.0 + (MAD / (|med| + 1e-6)))
    """
    if not values or len(values) < 2:
        return 1.0 if len(values) == 1 else 0.0
    if med is None or mad is None:
        med, mad = robust_median_and_mad(values)
    rel_dispersion = mad / (abs(med) + 1e-6)
    return round(float(1.0 / (1.0 + rel_dispersion)), 3)


def make_empty_template(template_id: str, submitted_count: int = 5) -> Dict[str, Any]:
    """Factory creating an empty canonical reference template structure."""
    return {
        "schema_version": SCHEMA_VERSION,
        "template_id": template_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_images": [],
        "batch_summary": {
            "submitted_count": submitted_count,
            "processed_count": 0,
            "usable_count": 0,
            "independent_view_count": 0,
            "rejected_count": 0,
        },
        "quality_summary": {
            "mean_quality_score": 0.0,
            "median_sharpness": 0.0,
            "usable_ratio": 0.0,
            "duplicate_count": 0,
        },
        "per_image_features": [],
        "alignment_summary": {
            "registration_attempted": False,
            "reliable_alignment": False,
            "method": "none",
            "quality_metric": 0.0,
        },
        "consensus_features": {},
        "recurring_graph_structures": [],
        "feature_stability": {
            "overall_stability_score": 0.0,
            "confidence_level": "insufficient_data",
        },
        "processing_config": {},
        "warnings": [],
    }
