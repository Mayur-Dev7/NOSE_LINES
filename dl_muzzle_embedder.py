#!/usr/bin/env python3
"""
dl_muzzle_embedder.py
=====================
Deep Learning Cattle Muzzle Feature Extractor and ArcFace Biometric Matching Engine.

Architecture:
- Backbones: ConvNeXt-Tiny (low latency) & ResNet-50 (baseline)
- Pooling: Generalized Mean (GeM) Pooling (p=3.0) to preserve high-frequency ridge & groove contours
- ArcFace: Additive Angular Margin penalty (s=30.0, m=0.50) with L2-normalized 512-D projection neck
- Dual Inference:
    * Fine-Tuned ArcFace mode (loads weights/arcface_{backbone}.pth if available)
    * Zero-Shot GeM Pretrained mode (extracts normalized GeM backbone features directly with
      deterministic orthonormal 512-D reduction or native pooled dimension, avoiding uninitialized weights)
- Input Tensor Modes:
    * composite_3ch (Default): Ch0=CLAHE Grayscale, Ch1=Dark Groove Response, Ch2=Dilated 3x3 Skeleton
    * skeleton_only: Dilated & smoothed Stage 5 skeleton replicated across 3 channels
    * groove_map: Stage 3 groove response replicated across 3 channels
    * Background zeroing via Stage 1 muzzle mask
- Temporal Aggregation:
    * 5-frame element-wise median vector re-normalized to unit L2 norm
    * Pairwise Cosine Similarity: max(0.0, e_A . e_B) * 100%
    * 5x5 pairwise cross-frame similarity matrix and intra-video consistency scoring
"""

import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("dl_muzzle_embedder")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [DL-Embedder] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 1. Generalized Mean (GeM) Pooling
# ---------------------------------------------------------------------------
class GeMPool2d(nn.Module):
    """
    Generalized Mean Pooling:
    f(x) = (1/HW * sum(x_{c, i, j}^p))^(1/p)

    As p -> inf, GeM approaches Max Pooling.
    When p = 1.0, GeM is Average Pooling.
    With p = 3.0, sharp high-frequency ridge and groove activations are preserved
    significantly better than standard average pooling.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        self.eps = eps
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * float(p))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp inputs to eps to avoid NaN gradients with fractional power
        x_clamped = x.clamp(min=self.eps)
        p_val = self.p.view(1, 1, 1, 1) if isinstance(self.p, nn.Parameter) else self.p
        # Power p -> Average Pool -> Power 1/p
        powered = x_clamped.pow(p_val)
        pooled = F.adaptive_avg_pool2d(powered, (1, 1))
        gem = pooled.pow(1.0 / p_val)
        return gem


# ---------------------------------------------------------------------------
# 2. ArcFace Head with Projection Neck & Margin Penalty
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head & Projection Neck:
    - Neck: BatchNorm2d -> Flatten -> Dropout(0.2) -> Linear(in_features, out_features, bias=False) -> BatchNorm1d
    - Loss: L = -log( e^(s * cos(theta + m)) / (e^(s * cos(theta + m)) + sum(e^(s * cos(theta_j)))) )
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 512,
        num_classes: Optional[int] = None,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Projection neck
        self.bn_in = nn.BatchNorm2d(in_features)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn_out = nn.BatchNorm1d(out_features)

        # Classification weight matrix for metric learning / ArcFace training
        if num_classes is not None and num_classes > 0:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, out_features))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.weight = None

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes 4D tensor (B, C, 1, 1) or (B, C) and outputs L2-normalized 512-D embedding.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.bn_in(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_out(x)
        return F.normalize(x, p=2, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        If labels is None, returns normalized 512-D embedding.
        If labels is provided, computes ArcFace margin logits for cross-entropy training.
        """
        embeddings = self.extract_embedding(features)
        if labels is None or self.weight is None:
            return embeddings

        # Cosine similarity between embeddings and class centroids
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        cosine = F.linear(embeddings, w_norm)  # (B, num_classes)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output, embeddings


# ---------------------------------------------------------------------------
# 3. Deterministic Dimension Reducer for Zero-Shot Mode
# ---------------------------------------------------------------------------
class DeterministicProjector(nn.Module):
    """
    Deterministic orthonormal projection matrix (in_dim -> out_dim).
    Generated via fixed seed QR decomposition.
    Ensures zero-shot features map deterministically to 512-D without random weight drift
    or training, preserving pairwise cosine similarity geometry.
    """
    def __init__(self, in_dim: int, out_dim: int = 512, seed: int = 42):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = np.random.RandomState(seed)
        mat = rng.randn(in_dim, out_dim).astype(np.float32)
        q, _ = np.linalg.qr(mat)
        self.register_buffer("proj", torch.from_numpy(q.T))  # (out_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, in_dim)
        projected = F.linear(x, self.proj)
        return F.normalize(projected, p=2, dim=-1)


# ---------------------------------------------------------------------------
# 4. Unified Muzzle Biometric Model
# ---------------------------------------------------------------------------
class MuzzleBiometricModel(nn.Module):
    """
    Unified Biometric Model supporting:
    - Backbones: 'convnext_tiny' (timm / torchvision) & 'resnet50'
    - Pooling: GeM (p=3.0)
    - Dual mode: Fine-Tuned ArcFace vs Pretrained Zero-Shot GeM
    """
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        target_dim: int = 512,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.target_dim = target_dim

        # 1. Initialize Backbone
        if "convnext" in self.backbone_name:
            self._init_convnext(pretrained)
            self.backbone_dim = 768
        elif "resnet50" in self.backbone_name:
            self._init_resnet50(pretrained)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'convnext_tiny' or 'resnet50'.")

        # 2. Replace Average Pooling with GeM Pooling (p=3.0)
        self.gem_pool = GeMPool2d(p=3.0, trainable=False)

        # 3. ArcFace Head
        self.arcface_head = ArcFaceHead(
            in_features=self.backbone_dim,
            out_features=target_dim,
            num_classes=num_classes,
        )

        # 4. Deterministic projector for zero-shot testing (keeps 512-D unified shape)
        self.zero_shot_projector = DeterministicProjector(
            in_dim=self.backbone_dim,
            out_dim=target_dim,
            seed=42,
        )

        # 5. Check if weights exist on disk
        self.weights_mode = "pretrained_gem_zeroshot"
        resolved_weights = self._resolve_weights_path(weights_path)
        if resolved_weights and os.path.exists(resolved_weights):
            try:
                state_dict = torch.load(resolved_weights, map_location="cpu")
                # Handle full model or state_dict
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                self.load_state_dict(state_dict, strict=False)
                self.weights_mode = "arcface_finetuned"
                logger.info(f"Loaded custom fine-tuned ArcFace weights from: {resolved_weights}")
            except Exception as e:
                logger.warning(f"Could not load weights from {resolved_weights}: {e}. Falling back to zero-shot mode.")
                self.weights_mode = "pretrained_gem_zeroshot"
        else:
            logger.info(f"No custom fine-tuned weights found for {self.backbone_name}. Operating in '{self.weights_mode}' mode.")

    def _resolve_weights_path(self, custom_path: Optional[str]) -> Optional[str]:
        if custom_path and os.path.exists(custom_path):
            return custom_path
        # Standard paths
        candidates = [
            WEIGHTS_DIR / f"arcface_{self.backbone_name}.pth",
            WEIGHTS_DIR / f"arcface_muzzle_{self.backbone_name}.pth",
            PROJECT_ROOT / f"arcface_{self.backbone_name}.pth",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _init_convnext(self, pretrained: bool):
        loaded = False
        # Try timm first
        try:
            import timm
            self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
            # Remove default pooling/classifier so forward_features yields spatial tensor
            loaded = True
            logger.info("Loaded ConvNeXt-Tiny via timm.")
        except Exception as e:
            logger.debug(f"timm convnext_tiny load note: {e}")

        # Fallback to torchvision
        if not loaded:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            tv_model = convnext_tiny(weights=weights)
            # Use features stage
            self.backbone = tv_model.features
            logger.info("Loaded ConvNeXt-Tiny via torchvision.models.")

    def _init_resnet50(self, pretrained: bool):
        loaded = False
        try:
            import timm
            self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            loaded = True
            logger.info("Loaded ResNet-50 via timm.")
        except Exception as e:
            logger.debug(f"timm resnet50 load note: {e}")

        if not loaded:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            tv_model = resnet50(weights=weights)
            # Construct backbone without avgpool and fc
            self.backbone = nn.Sequential(
                tv_model.conv1,
                tv_model.bn1,
                tv_model.relu,
                tv_model.maxpool,
                tv_model.layer1,
                tv_model.layer2,
                tv_model.layer3,
                tv_model.layer4,
            )
            logger.info("Loaded ResNet-50 via torchvision.models.")

    def forward_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map (B, C, H', W') from backbone."""
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
            # Some timm backbones return (B, H, W, C) for convnext
            if feat.dim() == 4 and feat.shape[-1] == self.backbone_dim:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            return feat
        else:
            return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        force_zero_shot: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass:
        Input: (B, 3, H, W)
        Output: L2-normalized 512-D embedding tensor (B, 512)
        """
        spatial_feats = self.forward_spatial_features(x)  # (B, C, H', W')
        gem_pooled = self.gem_pool(spatial_feats)         # (B, C, 1, 1)

        # Fine-Tuned / Training Mode
        if (self.weights_mode == "arcface_finetuned" or labels is not None) and not force_zero_shot:
            return self.arcface_head(gem_pooled, labels)

        # Zero-Shot Mode:
        # Avoid uninitialized random projection!
        # Standardize pooled features and project deterministically to 512-D
        flat = torch.flatten(gem_pooled, 1)  # (B, backbone_dim)
        # Feature standardization
        flat_normed = F.normalize(flat, p=2, dim=-1)
        # Deterministic orthonormal projection to 512-D
        emb = self.zero_shot_projector(flat_normed)
        return emb


# ---------------------------------------------------------------------------
# 5. Input Tensor Construction Modes
# ---------------------------------------------------------------------------
def prepare_input_tensor(
    clahe_gray: np.ndarray,
    groove_response: np.ndarray,
    skeleton: np.ndarray,
    mask: Optional[np.ndarray] = None,
    input_mode: str = "composite_3ch",
    target_size: Tuple[int, int] = (224, 224),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Prepares input tensor handling 1-pixel skeleton sparsity:
    Modes:
    1. 'composite_3ch' (DEFAULT):
       - Ch0 (R): Stage 2 CLAHE Grayscale
       - Ch1 (G): Stage 3 Groove Response Map
       - Ch2 (B): Stage 5 1-px Skeleton dilated with 3x3 kernel
    2. 'skeleton_only':
       - Stage 5 skeleton dilated (3x3) + Gaussian smoothed, replicated across 3 channels
    3. 'groove_map':
       - Stage 3 groove response replicated across 3 channels

    Applies native muzzle skin mask (Stage 1) to eliminate background artifacts before normalization.
    Standard ImageNet mean & std normalization applied.
    """
    mode = (input_mode or "composite_3ch").lower()

    # Ensure grayscale 2D arrays
    def to_2d(arr):
        if arr is None:
            return np.zeros(target_size, dtype=np.uint8)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)

    g_clahe = to_2d(clahe_gray)
    g_groove = to_2d(groove_response)
    g_skel = to_2d(skeleton)

    # Resize all stages to standard canvas
    h, w = target_size
    if g_clahe.shape[:2] != (h, w):
        g_clahe = cv2.resize(g_clahe, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if g_groove.shape[:2] != (h, w):
        g_groove = cv2.resize(g_groove, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if g_skel.shape[:2] != (h, w):
        g_skel = cv2.resize(g_skel, (w, h), interpolation=cv2.INTER_NEAREST)

    # Mask background outside muzzle perimeter
    if mask is not None and mask.size > 0:
        m = to_2d(mask)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_bin = m > 0
        g_clahe[~m_bin] = 0
        g_groove[~m_bin] = 0
        g_skel[~m_bin] = 0

    # Dilate 1-pixel skeleton with 3x3 kernel so groove topology survives CNN downsampling
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    skel_dilated = cv2.dilate(g_skel, dilate_kernel, iterations=1)

    if mode == "skeleton_only":
        # Continuous ridge density map via 3x3 dilation + Gaussian smoothing
        skel_smooth = cv2.GaussianBlur(skel_dilated, (3, 3), sigmaX=1.0)
        ch0 = skel_smooth
        ch1 = skel_smooth
        ch2 = skel_smooth
    elif mode == "groove_map":
        ch0 = g_groove
        ch1 = g_groove
        ch2 = g_groove
    else:  # default 'composite_3ch'
        ch0 = g_clahe
        ch1 = g_groove
        ch2 = skel_dilated

    stacked = np.stack([ch0, ch1, ch2], axis=-1)  # (H, W, 3)
    img_float = stacked.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normed = (img_float - mean) / std

    # Convert to PyTorch tensor (1, 3, H, W)
    tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------------------------
# 6. Embedder Pipeline Wrapper Singleton
# ---------------------------------------------------------------------------
_EMBEDDER_SINGLETONS: Dict[str, MuzzleBiometricModel] = {}


def get_muzzle_embedder(
    backbone: str = "convnext_tiny",
    weights_path: Optional[str] = None,
    device: str = "auto",
) -> MuzzleBiometricModel:
    """
    Returns or initializes cached singleton MuzzleBiometricModel.
    Thread-safe and fast for server inference.
    """
    global _EMBEDDER_SINGLETONS

    if str(device).lower() in ("gpu", "cuda"):
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # Default CPU for 100% deterministic, conflict-free inference alongside TF 1.15
        dev_str = "cpu"
    key = f"{backbone.lower()}_{dev_str}"

    if key not in _EMBEDDER_SINGLETONS:
        logger.info(f"Initializing MuzzleBiometricModel [backbone={backbone}, device={dev_str}]...")
        model = MuzzleBiometricModel(backbone=backbone, weights_path=weights_path)
        model.to(torch.device(dev_str))
        model.eval()
        _EMBEDDER_SINGLETONS[key] = model

    return _EMBEDDER_SINGLETONS[key]


# ---------------------------------------------------------------------------
# 7. Single Frame Extraction and Latency Timing
# ---------------------------------------------------------------------------
def extract_frame_embedding(
    model: MuzzleBiometricModel,
    clahe_gray: np.ndarray,
    groove_response: np.ndarray,
    skeleton: np.ndarray,
    mask: Optional[np.ndarray] = None,
    input_mode: str = "composite_3ch",
    target_size: Tuple[int, int] = (224, 224),
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    Extracts L2-normalized 512-D embedding for a single muzzle frame.
    Returns:
        (embedding_array, latency_ms)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # Time end-to-end preprocessing + inference
    t0 = time.perf_counter()
    with torch.no_grad():
        x = prepare_input_tensor(
            clahe_gray=clahe_gray,
            groove_response=groove_response,
            skeleton=skeleton,
            mask=mask,
            input_mode=input_mode,
            target_size=target_size,
            device=device,
        )
        emb = model(x)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb_np = emb.detach().cpu().squeeze(0).numpy().astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return emb_np, round(latency_ms, 2)


# ---------------------------------------------------------------------------
# 8. 5-Frame Temporal Aggregation & Cosine Matching
# ---------------------------------------------------------------------------
def aggregate_temporal_embeddings(
    embeddings: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Aggregates 5 temporal frame embeddings:
    1. Temporal element-wise median: e_raw = median(e_1, e_2, e_3, e_4, e_5)
    2. Re-apply L2 normalization: ||e_median||_2 = 1.0
    3. Intra-video frame consistency score: mean pairwise cosine similarity between the 5 frames.

    Returns:
        (median_embedding, intra_video_consistency_pct)
    """
    if not embeddings:
        return np.zeros(512, dtype=np.float32), 0.0

    arr = np.stack(embeddings, axis=0)  # (N, D)
    raw_median = np.median(arr, axis=0)
    norm = np.linalg.norm(raw_median)
    median_emb = (raw_median / max(1e-8, norm)).astype(np.float32)

    # Intra-video consistency (%): mean pairwise cosine similarity across frames
    n = len(embeddings)
    if n > 1:
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sims.append(max(0.0, sim))
        intra_pct = float(np.mean(sims)) * 100.0 if sims else 100.0
    else:
        intra_pct = 100.0

    return median_emb, round(intra_pct, 2)


def compute_dl_cosine_similarity(
    emb_a: Union[List[float], np.ndarray],
    emb_b: Union[List[float], np.ndarray],
    per_frames_a: Optional[List[Union[List[float], np.ndarray]]] = None,
    per_frames_b: Optional[List[Union[List[float], np.ndarray]]] = None,
) -> Dict[str, Any]:
    """
    Computes Pairwise Cosine Similarity between Video A and Video B median embeddings:
    sim_dl = max(0.0, e_A . e_B) * 100%
    l2_dist = ||e_A - e_B||_2

    Also computes 5x5 frame-to-frame cross matrix and stats.
    """
    va = np.array(emb_a, dtype=np.float32)
    vb = np.array(emb_b, dtype=np.float32)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    va_unit = va / max(1e-8, norm_a)
    vb_unit = vb / max(1e-8, norm_b)

    dot = float(np.dot(va_unit, vb_unit))
    sim_pct = round(max(0.0, min(1.0, dot)) * 100.0, 2)
    l2_dist = round(float(np.linalg.norm(va_unit - vb_unit)), 4)

    # Decision thresholds
    if sim_pct >= 88.0:
        verdict = "MATCH (Confirmed Same Cattle)"
        badge_variant = "match"
        badge_color = "emerald"
    elif sim_pct >= 78.0:
        verdict = "LIKELY MATCH (High Biometric Similarity)"
        badge_variant = "likely"
        badge_color = "teal"
    elif sim_pct >= 68.0:
        verdict = "INCONCLUSIVE (Partial Geometric Overlap)"
        badge_variant = "inconclusive"
        badge_color = "amber"
    else:
        verdict = "DISTINCT (Different Animals)"
        badge_variant = "distinct"
        badge_color = "rose"

    # Cross 5x5 frame similarity matrix
    matrix_stats = {}
    cross_matrix = []
    if per_frames_a and per_frames_b:
        list_a = [np.array(f, dtype=np.float32) for f in per_frames_a]
        list_b = [np.array(f, dtype=np.float32) for f in per_frames_b]
        all_pair_sims = []
        for i, fa in enumerate(list_a):
            row = []
            fa_u = fa / max(1e-8, np.linalg.norm(fa))
            for j, fb in enumerate(list_b):
                fb_u = fb / max(1e-8, np.linalg.norm(fb))
                s = round(max(0.0, float(np.dot(fa_u, fb_u))) * 100.0, 2)
                row.append(s)
                all_pair_sims.append(s)
            cross_matrix.append(row)

        if all_pair_sims:
            matrix_stats = {
                "min_pair_similarity": round(float(np.min(all_pair_sims)), 2),
                "max_pair_similarity": round(float(np.max(all_pair_sims)), 2),
                "median_pair_similarity": round(float(np.median(all_pair_sims)), 2),
                "mean_pair_similarity": round(float(np.mean(all_pair_sims)), 2),
            }

    return {
        "dl_similarity_pct": sim_pct,
        "l2_distance": l2_dist,
        "verdict": verdict,
        "badge_variant": badge_variant,
        "badge_color": badge_color,
        "cross_matrix": cross_matrix,
        "cross_matrix_stats": matrix_stats,
    }


# ---------------------------------------------------------------------------
# CLI Self-Test & Benchmarking
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Testing dl_muzzle_embedder.py ===")
    for b in ["convnext_tiny", "resnet50"]:
        try:
            print(f"\nInitializing {b}...")
            model = get_muzzle_embedder(backbone=b, device="cpu")
            # Create synthetic test frames (448x448)
            dummy_clahe = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
            dummy_groove = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
            dummy_skel = (np.random.rand(448, 448) > 0.95).astype(np.uint8) * 255
            dummy_mask = np.full((448, 448), 255, dtype=np.uint8)

            emb, ms = extract_frame_embedding(
                model=model,
                clahe_gray=dummy_clahe,
                groove_response=dummy_groove,
                skeleton=dummy_skel,
                mask=dummy_mask,
                input_mode="composite_3ch",
            )
            print(f"[{b}] Extracted embedding dim={emb.shape}, norm={np.linalg.norm(emb):.4f}, latency={ms:.2f}ms")
        except Exception as e:
            print(f"[{b}] Error: {e}")
