"""
Centralized Configuration for Cow Muzzle Detection, Biometric Matching & Groove Extraction
========================================================================================
Centralizes all hardware constraints, video sampling, Mask R-CNN inference parameters,
super-resolution enhancement, morphological groove extraction, and skeletonization settings.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = str(BASE_DIR / "weights" / "crop11dec.h5")
if not os.path.isfile(WEIGHTS_PATH) and os.path.isfile("/home/adis_mayur/F_CODES/crop11dec.h5"):
    WEIGHTS_PATH = "/home/adis_mayur/F_CODES/crop11dec.h5"

OUTPUT_DIR = str(BASE_DIR / "output")
UPLOADS_DIR = str(BASE_DIR / "uploads")
SAMPLES_DIR = str(BASE_DIR / "samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Hardware Configuration (Enforce CPU 2-Core or Multi-Core Limits)
# -----------------------------------------------------------------------------
DEFAULT_CPU_CORES = 2
DEVICE = "cpu"
FORCE_CPU = False

def enforce_cpu_lock(cores: int = DEFAULT_CPU_CORES):
    """
    Enforces thread limits for TensorFlow, OpenMP, BLAS, and OpenCV.
    """
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    try:
        import cv2
        cv2.setNumThreads(cores)
    except ImportError:
        pass

# -----------------------------------------------------------------------------
# Video Sampling Configuration
# -----------------------------------------------------------------------------
FRAME_INTERVAL = 10               # Sample every 10th frame
MAX_CANDIDATE_FRAMES = 8          # Stop video sampling after evaluating N candidate frames
BLUR_VARIANCE_THRESHOLD = 20.0    # Minimum Laplacian variance (skip frames with blur < 20.0)

# -----------------------------------------------------------------------------
# Mask R-CNN Muzzle Detection Configuration
# -----------------------------------------------------------------------------
DETECTION_CONFIDENCE_THRESHOLD = 0.70   # Minimum Mask R-CNN detection confidence
PRISTINE_CONFIDENCE_THRESHOLD = 0.85    # Early-exit threshold for pristine muzzle detection
IMAGE_MIN_DIM = 800                     # Input mold minimum dimension
IMAGE_MAX_DIM = 1024                    # Input mold maximum dimension (must be divisible by 64: 1024 / 64 = 16)
POST_NMS_ROIS_INFERENCE = 100           # ROIs passed after NMS for inference speedup
CROP_SIZE = (224, 224)                  # Final normalized biometric crop resolution (W, H)
CROP_PADDING_PCT = 0.08                 # 8% bounding box margin expansion

# -----------------------------------------------------------------------------
# Super-Resolution & Grayscale Ridge Enhancement Configuration
# -----------------------------------------------------------------------------
UPSCALE_METHOD = "lanczos4"             # "lanczos4" (sinc 8-point), "bicubic", or "bilinear"
UPSCALE_FACTOR = 2                      # 2x super-resolution (e.g. 224x224 -> 448x448)
APPLY_CLAHE = True                      # Contrast Limited Adaptive Histogram Equalization
APPLY_UNSHARP = True                    # Micro-edge groove unsharp masking
CLAHE_CLIP_LIMIT = 2.0                  # CLAHE clip limit
CLAHE_GRID_SIZE = (8, 8)                # CLAHE local grid tile size

# -----------------------------------------------------------------------------
# Muzzle Dark Groove Line Extraction & Skeletonization Configuration
# -----------------------------------------------------------------------------
# Stage A: Input Selection
GROOVE_USE_UPSCALED = False             # Default to native crop; upscaled is optional comparison

# Stage B: Muzzle Region Isolation
GROOVE_EXCLUDE_NOSTRILS = True          # Detect and suppress dark nostril cavities
GROOVE_MASK_CLOSE_KERNEL = 5            # Morphological closing kernel to fill holes in skin mask
GROOVE_MASK_OPEN_KERNEL = 3             # Morphological opening kernel to prune boundary hairs

# Stage C: Groove Line Enhancement
GROOVE_ENHANCE_METHOD = "blackhat"      # "blackhat" (morphology) or "frangi" (Hessian vesselness)
GROOVE_BLACKHAT_SCALES: List[int] = [9, 15, 21]  # Elliptical kernel diameters for 448x448 reference
GROOVE_BASE_DIM = 448                   # Reference resolution for proportional kernel scaling
GROOVE_FRANGI_SIGMAS = (1.0, 2.5, 4.0)  # Multi-scale Gaussian smoothing sigmas for Frangi filter

# Stage D: Binary Groove Map Generation
GROOVE_THRESHOLD_METHOD = "otsu"        # "otsu" (automatic bimodal) or "adaptive" (local Gaussian)
GROOVE_ADAPTIVE_BLOCK_SIZE = 15         # Neighborhood block size for adaptive thresholding
GROOVE_ADAPTIVE_C = 2                   # Constant subtracted from weighted mean
GROOVE_MIN_COMPONENT_SIZE = 15          # Minimum connected component area (prunes speckle noise)
GROOVE_LINEAR_ASPECT_RATIO = 3.0        # Components with aspect ratio > 3.0 preserved even if small

# Stage E: Thin Groove Skeletonization
GROOVE_SKELETON_ENABLED = True          # Generate 1-pixel-wide topological centerline network
GROOVE_OVERLAY_COLOR: Tuple[int, int, int] = (0, 255, 0) # Green BGR skeleton overlay

# -----------------------------------------------------------------------------
# Biometric Matching Configuration
# -----------------------------------------------------------------------------
ORB_N_FEATURES = 1000                   # Number of ORB keypoints to extract
ORB_RATIO_TEST = 0.7                    # Lowe's distance ratio test threshold (0.7)
DUPLICATE_DECISION_THRESHOLD = 35       # >= 35 good matches declares SAME CATTLE duplicate
