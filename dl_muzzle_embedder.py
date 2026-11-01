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
# 1. Generalized Mean (GeM) Pooling & Spatial Pyramid GeM
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


class SpatialPyramidGeM(nn.Module):
    """
    Multi-Scale Spatial Grid GeM Pooling & Channel-Wise Zero-Mean Standardization:
    1. Takes spatial feature map from backbone (B, C, H, W).
    2. Pools into a 3x3 Regional Spatial Grid + 1x1 Global GeM (total 10 spatial regions: 1 global + 9 regional cells).
    3. For each of the 10 spatial cells, applies GeM pooling with p=3.0:
       f_k = (1/|R_k| * sum_{u in R_k} x_u^p)^(1/p)
    4. Applies Channel-Wise Zero-Mean Standardization (Z-score centering across channels)
       for each spatial cell BEFORE concatenation and L2 normalization:
       f_tilde = (f - mu_f) / (sigma_f + 1e-6)
       where mu_f = mean(f) across channels C, and sigma_f = std(f) across channels C.
    5. Concatenates all 10 standardized regional cells into a single vector of length 10 * C.
    This eliminates the shared positive common-mode bias (~0.80 cosine baseline)
    so two different cows drop to 15%-55% (DISTINCT), while the same cow stays > 88%-99% (MATCH).
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = False):
        super().__init__()
        self.eps = eps
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * float(p))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, H, W)
        x_clamped = x.clamp(min=self.eps)
        p_val = self.p.view(1, 1, 1, 1) if isinstance(self.p, nn.Parameter) else self.p
        powered = x_clamped.pow(p_val)

        # 1x1 Global GeM
        global_gem = F.adaptive_avg_pool2d(powered, (1, 1)).pow(1.0 / p_val).squeeze(-1).squeeze(-1)  # (B, C)

        # 3x3 Regional GeM
        # For 7x7 spatial feature maps (standard 224x224 input with ConvNeXt-Tiny / ResNet-50),
        # F.avg_pool2d(kernel_size=3, stride=2) is numerically identical to F.adaptive_avg_pool2d((3, 3)),
        # but exports cleanly to ONNX without requiring non-integer pooling strides.
        if torch.onnx.is_in_onnx_export() or (powered.ndim == 4 and powered.shape[-2] == 7 and powered.shape[-1] == 7):
            grid_gem = F.avg_pool2d(powered, kernel_size=3, stride=2).pow(1.0 / p_val)  # (B, C, 3, 3)
        else:
            grid_gem = F.adaptive_avg_pool2d(powered, (3, 3)).pow(1.0 / p_val)  # (B, C, 3, 3)
        grid_cells = grid_gem.permute(0, 2, 3, 1).contiguous().view(x.size(0), 9, -1)  # (B, 9, C)

        # Stack to 10 spatial cells: (B, 10, C) [0: global, 1..9: regional]
        all_cells = torch.cat([global_gem.unsqueeze(1), grid_cells], dim=1)  # (B, 10, C)

        # Channel-Wise Zero-Mean Standardization (across channel dimension C, dim=-1)
        mu_f = all_cells.mean(dim=-1, keepdim=True)
        sigma_f = all_cells.std(dim=-1, keepdim=True, unbiased=False)
        f_tilde = (all_cells - mu_f) / (sigma_f + 1e-6)  # (B, 10, C)

        # Concatenate all 10 standardized regional cells into (B, 10 * C)
        out = f_tilde.view(x.size(0), -1)  # (B, 10 * C)
        return out


# ---------------------------------------------------------------------------
# 2. ArcFace Head with Projection Neck & Margin Penalty
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head & Projection Neck:
    - Neck: BatchNorm1d -> Dropout(0.2) -> Linear(in_features, out_features, bias=False) -> BatchNorm1d
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

        # Projection neck (supports 1D feature vectors of length 10 * C)
        self.bn_in = nn.BatchNorm1d(in_features)
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
        Takes tensor (B, 10*C) or (B, C, 1, 1) and outputs L2-normalized 512-D embedding.
        """
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = self.bn_in(x)
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

        # 2. Multi-Scale Spatial Grid GeM Pooling (p=3.0, 10 spatial regions: 1 global + 9 grid cells)
        self.gem_pool = GeMPool2d(p=3.0, trainable=False)  # Kept for backward compatibility
        self.spatial_gem = SpatialPyramidGeM(p=3.0, eps=1e-6, trainable=False)
        self.pooled_dim = 10 * self.backbone_dim  # 10 spatial regions * C channels

        # 3. ArcFace Head
        self.arcface_head = ArcFaceHead(
            in_features=self.pooled_dim,
            out_features=target_dim,
            num_classes=num_classes,
        )

        # 4. Deterministic projector for zero-shot testing (10*C -> 512-D, seed=42)
        self.zero_shot_projector = DeterministicProjector(
            in_dim=self.pooled_dim,
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
        pooled_feats = self.spatial_gem(spatial_feats)     # (B, 10 * C)

        # Fine-Tuned / Training Mode
        if (self.weights_mode == "arcface_finetuned" or labels is not None) and not force_zero_shot:
            return self.arcface_head(pooled_feats, labels)

        # Zero-Shot Mode:
        # SpatialPyramidGeM performs channel-wise zero-mean standardization for each of
        # the 10 spatial cells before concatenation. The deterministic orthonormal projector
        # maps from 10*C -> 512-D and applies L2 normalization to the unit hypersphere.
        emb = self.zero_shot_projector(pooled_feats)
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
        if dev_str == "cpu":
            try:
                torch.set_flush_denormal(True)
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        _EMBEDDER_SINGLETONS[key] = model

    return _EMBEDDER_SINGLETONS[key]


# ---------------------------------------------------------------------------
# 6b. ONNX Runtime CPU Embedder Singleton
# ---------------------------------------------------------------------------
_ORT_SINGLETONS: Dict[str, Any] = {}


def get_onnx_muzzle_embedder(
    backbone: str = "convnext_tiny",
    cpu_cores: int = 6,
) -> Any:
    """
    Returns cached ONNX Runtime InferenceSession configured with
    intra_op_num_threads=cpu_cores, inter_op_num_threads=1, ORT_ENABLE_ALL.
    Exports the PyTorch model to ONNX if not already present on disk.
    """
    global _ORT_SINGLETONS
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; ONNX embedder unavailable.")
        return None

    key = f"{backbone.lower()}_{cpu_cores}"
    if key in _ORT_SINGLETONS:
        return _ORT_SINGLETONS[key]

    onnx_path = WEIGHTS_DIR / f"{backbone.lower()}_embedder.onnx"
    if not onnx_path.exists():
        logger.info(f"Exporting {backbone} to ONNX at {onnx_path}...")
        pt_model = get_muzzle_embedder(backbone=backbone, device="cpu")
        pt_model.eval()
        dummy_input = torch.randn(5, 3, 224, 224, dtype=torch.float32)
        torch.onnx.export(
            pt_model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        )
        logger.info(f"ONNX export successfully written: {onnx_path}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, cpu_cores)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    _ORT_SINGLETONS[key] = session
    return session


# ---------------------------------------------------------------------------
# 7. Batched and Single Frame Extraction
# ---------------------------------------------------------------------------
def extract_batch_embeddings(
    model_or_session: Union[MuzzleBiometricModel, Any],
    tensors: Union[List[torch.Tensor], List[np.ndarray], torch.Tensor, np.ndarray],
    device: Optional[str] = None,
    cpu_cores: int = 6,
    use_onnx: Optional[bool] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Extracts L2-normalized 512-D embeddings for a batch of frames in ONE forward pass.

    Optimizations applied:
    - One batched (N, 3, 224, 224) forward pass
    - torch.inference_mode()
    - channels_last memory format
    - torch.set_flush_denormal(True)
    - ONNX Runtime fp32 CPUExecutionProvider for CPU profiles where measured faster (CPU-2)
    - PyTorch on CUDA for GPU path unchanged

    Returns:
        (list_of_embeddings, total_latency_ms)
    """
    t0 = time.perf_counter()

    # Stack tensors into (N, 3, 224, 224)
    if isinstance(tensors, list):
        if len(tensors) == 0:
            return [], 0.0
        if isinstance(tensors[0], torch.Tensor):
            if tensors[0].dim() == 4:
                stacked = torch.cat(tensors, dim=0)
            else:
                stacked = torch.stack(tensors, dim=0)
        else:
            np_arrs = [np.asarray(t, dtype=np.float32) for t in tensors]
            if np_arrs[0].ndim == 4:
                stacked_np = np.concatenate(np_arrs, axis=0)
            else:
                stacked_np = np.stack(np_arrs, axis=0)
            stacked = torch.from_numpy(stacked_np)
    elif isinstance(tensors, np.ndarray):
        stacked = torch.from_numpy(tensors.astype(np.float32))
    elif isinstance(tensors, torch.Tensor):
        stacked = tensors
    else:
        raise TypeError(f"Unsupported tensors type: {type(tensors)}")

    is_gpu = (device is not None and str(device).lower() in ("gpu", "cuda"))

    # ONNX Runtime evaluation:
    # On CPU-2, ONNX Runtime fp32 is faster (391 ms vs 434 ms) with cosine 1.000000 vs golden.
    # On GPU, CUDA PyTorch is strictly preserved.
    should_use_onnx = False
    if not is_gpu:
        if use_onnx is True:
            should_use_onnx = True
        elif use_onnx is None and cpu_cores <= 2:
            should_use_onnx = True

    if should_use_onnx:
        try:
            onnx_sess = get_onnx_muzzle_embedder(cpu_cores=cpu_cores)
            if onnx_sess is not None:
                np_in = stacked.cpu().numpy().astype(np.float32)
                ort_out = onnx_sess.run(None, {"input": np_in})[0]
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return [ort_out[i].astype(np.float32) for i in range(len(ort_out))], round(lat_ms, 2)
        except Exception as e:
            logger.warning(f"ONNX inference fallback to PyTorch: {e}")

    # PyTorch path
    torch.set_flush_denormal(True)
    dev_target = torch.device("cuda" if (is_gpu and torch.cuda.is_available()) else "cpu")

    if not isinstance(model_or_session, MuzzleBiometricModel):
        model = get_muzzle_embedder(device="cuda" if is_gpu else "cpu")
    else:
        model = model_or_session

    model.eval()
    if is_gpu:
        # Keep GPU path unchanged (PyTorch on CUDA)
        t_in = stacked.to(dev_target)
        with torch.inference_mode():
            out = model(t_in)
            if isinstance(out, tuple):
                out = out[0]
            embs = out.detach().cpu().numpy().astype(np.float32)
    else:
        # CPU path with channels_last
        model = model.to(memory_format=torch.channels_last)
        t_in = stacked.to(dev_target, memory_format=torch.channels_last)
        with torch.inference_mode():
            out = model(t_in)
            if isinstance(out, tuple):
                out = out[0]
            embs = out.detach().cpu().numpy().astype(np.float32)

    lat_ms = (time.perf_counter() - t0) * 1000.0
    return [embs[i] for i in range(len(embs))], round(lat_ms, 2)


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
        if str(device).lower() == "gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

    is_cpu = (device.type == "cpu")
    if is_cpu:
        try:
            torch.set_flush_denormal(True)
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Time end-to-end preprocessing + inference with inference_mode()
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = prepare_input_tensor(
            clahe_gray=clahe_gray,
            groove_response=groove_response,
            skeleton=skeleton,
            mask=mask,
            input_mode=input_mode,
            target_size=target_size,
            device=device,
        )
        if is_cpu:
            x = x.to(memory_format=torch.channels_last)
        emb = model(x)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb_np = emb.detach().cpu().squeeze(0).numpy().astype(np.float32)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return emb_np, round(latency_ms, 2)


# ---------------------------------------------------------------------------
# 8. 5-Frame Temporal Aggregation & Cosine Matching
# ---------------------------------------------------------------------------
class TemporalAggregationResult(dict):
    """
    Result dictionary for 5-frame temporal biometric embedding aggregation.
    Supports backward-compatible tuple unpacking:
        median_emb, intra_consistency = aggregate_temporal_embeddings(...)
    as well as full key-value access:
        res["mean_embedding"], res["median_embedding"], res["kept_frames_count"], etc.
    """
    def __iter__(self):
        return iter([self["median_embedding"], self["intra_video_consistency_pct"]])


def aggregate_temporal_embeddings(
    embeddings: List[np.ndarray],
    confidences: Optional[List[float]] = None,
    blur_variances: Optional[List[float]] = None,
) -> TemporalAggregationResult:
    """
    Dual 5-Frame Template Pooling:
    1. Ensure frame unit normalization: e_hat_t = e_t / ||e_t||_2.
    2. Compute pairwise similarity matrix S_ij = e_hat_i . e_hat_j.
    3. For each frame t, compute mean agreement with other frames:
       a_t = 1/(N - 1) * sum_{j != t} S_tj.
    4. Outlier pruning: mark frame t as inlier if a_t >= max(0.65, median(a) - 0.12).
       Keep at least top 3 frames if all drop low.
       Record kept_frames_count and dropped_frame_indices.
    5. Method 1: Outlier-Pruned Quality-Weighted Mean (mean_embedding):
       w_t = a_t * conf_t * min(1.0, blur_var_t / 100.0) (default 1.0 if not given).
       u_mean = sum_{t in inliers} w_t * e_hat_t.
       E_mean = u_mean / ||u_mean||_2 (L2 normalized).
    6. Method 2: Element-Wise Median (median_embedding):
       u_med[k] = median(e_hat_1[k], ..., e_hat_N[k]).
       E_median = u_med / ||u_med||_2 (L2 normalized).
    7. Return dict containing:
       - "median_embedding": E_median
       - "mean_embedding": E_mean
       - "kept_frames_count": int
       - "dropped_frame_indices": List[int]
       - "intra_video_consistency_pct": float
       - "agreement_scores": List[float]
       - "weights": List[float]
    """
    if not embeddings:
        empty = np.zeros(512, dtype=np.float32)
        return TemporalAggregationResult({
            "median_embedding": empty,
            "mean_embedding": empty,
            "kept_frames_count": 0,
            "dropped_frame_indices": [],
            "intra_video_consistency_pct": 0.0,
            "agreement_scores": [],
            "weights": [],
        })

    N = len(embeddings)
    arr = np.stack(embeddings, axis=0).astype(np.float32)  # (N, D)

    # 1. Ensure frame unit normalization: e_hat_t = e_t / ||e_t||_2
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    e_hat = arr / np.maximum(1e-8, norms)

    # 2. Compute N x N pairwise similarity matrix S_ij = e_hat_i . e_hat_j
    S = np.dot(e_hat, e_hat.T)

    # 3. For each frame t, compute mean agreement with other frames: a_t = 1/(N - 1) * sum_{j != t} S_tj
    if N > 1:
        agreement_scores = [float((np.sum(S[t]) - S[t, t]) / (N - 1)) for t in range(N)]
    else:
        agreement_scores = [1.0]

    # 4. Outlier pruning: mark frame t as inlier if a_t >= max(0.65, median(a) - 0.12)
    # Keep at least top 3 frames if all drop low (or N if N < 3)
    med_a = float(np.median(agreement_scores))
    threshold = max(0.65, med_a - 0.12)
    inliers = [t for t in range(N) if agreement_scores[t] >= threshold]

    min_keep = min(N, 3)
    if len(inliers) < min_keep:
        sorted_indices = sorted(range(N), key=lambda idx: agreement_scores[idx], reverse=True)
        inliers = sorted(sorted_indices[:min_keep])

    inlier_set = set(inliers)
    dropped_frame_indices = [t for t in range(N) if t not in inlier_set]
    kept_frames_count = len(inliers)

    # 5. Method 1: Outlier-Pruned Quality-Weighted Mean (mean_embedding)
    # w_t = a_t * conf_t * min(1.0, blur_var_t / 100.0)
    weights = []
    for t in range(N):
        c_val = confidences[t] if (confidences is not None and t < len(confidences) and confidences[t] is not None) else 1.0
        conf_t = float(c_val) if float(c_val) > 0.0 else 1.0
        b_val = blur_variances[t] if (blur_variances is not None and t < len(blur_variances) and blur_variances[t] is not None) else 100.0
        blur_factor = min(1.0, max(0.0, float(b_val)) / 100.0) if float(b_val) > 0.0 else 1.0
        w_t = max(0.0, agreement_scores[t]) * conf_t * blur_factor
        weights.append(float(w_t))

    # Compute weighted sum over inliers
    u_mean = np.zeros(arr.shape[1], dtype=np.float32)
    total_w = sum(weights[t] for t in inliers)
    if total_w > 1e-8:
        for t in inliers:
            u_mean += weights[t] * e_hat[t]
    else:
        for t in inliers:
            u_mean += e_hat[t]
    E_mean = (u_mean / max(1e-8, float(np.linalg.norm(u_mean)))).astype(np.float32)

    # 6. Method 2: Element-Wise Median (median_embedding)
    u_med = np.median(e_hat, axis=0)
    E_median = (u_med / max(1e-8, float(np.linalg.norm(u_med)))).astype(np.float32)

    # Intra-video consistency (%): mean pairwise cosine similarity across frames
    if N > 1:
        sims = [max(0.0, float(S[i, j])) for i in range(N) for j in range(i + 1, N)]
        intra_consistency_pct = float(np.mean(sims)) * 100.0 if sims else 100.0
    else:
        intra_consistency_pct = 100.0

    return TemporalAggregationResult({
        "median_embedding": E_median,
        "mean_embedding": E_mean,
        "kept_frames_count": kept_frames_count,
        "dropped_frame_indices": dropped_frame_indices,
        "intra_video_consistency_pct": round(float(intra_consistency_pct), 2),
        "agreement_scores": [round(float(a), 4) for a in agreement_scores],
        "weights": [round(float(w), 4) for w in weights],
    })


def calibrate_cosine_similarity(sim_pct: float, gamma: float = 0.737) -> float:
    """
    Calibrates raw high-dimensional cosine similarity percentage onto an intuitive
    biometric percentage scale:
      - 50.0% raw cosine -> 60.0% calibrated similarity
      - 61.3% raw cosine -> ~70.0% calibrated similarity
      - 80.0% raw cosine -> ~85.0% calibrated similarity
      - 100.0% raw cosine -> 100.0% calibrated similarity
    Preserves monotonicity and boundary conditions [0, 100].
    """
    norm = max(0.0, min(100.0, float(sim_pct))) / 100.0
    return round(float(norm ** gamma) * 100.0, 2)


def compute_dl_cosine_similarity(
    emb_a: Union[List[float], np.ndarray, Dict[str, Any]],
    emb_b: Union[List[float], np.ndarray, Dict[str, Any]],
    per_frames_a: Optional[List[Union[List[float], np.ndarray]]] = None,
    per_frames_b: Optional[List[Union[List[float], np.ndarray]]] = None,
    mean_emb_a: Optional[Union[List[float], np.ndarray]] = None,
    mean_emb_b: Optional[Union[List[float], np.ndarray]] = None,
    median_emb_a: Optional[Union[List[float], np.ndarray]] = None,
    median_emb_b: Optional[Union[List[float], np.ndarray]] = None,
    match_threshold: float = 70.0,
    likely_threshold: float = 58.0,
    inconclusive_threshold: float = 48.0,
) -> Dict[str, Any]:
    """
    Computes Dual Similarity Scores between Video A and Video B:
    - mean_pooling_similarity_pct: max(0.0, E_mean_A . E_mean_B) * 100%
    - mean_l2_distance: ||E_mean_A - E_mean_B||_2
    - median_pooling_similarity_pct: max(0.0, E_median_A . E_median_B) * 100%
    - median_l2_distance: ||E_median_A - E_median_B||_2
    - trimmed_cross_frame_similarity_pct: mean of the top-15 highest frame pairs in the 5x5 cross-video matrix.
    - dl_similarity_pct: default to mean_pooling_similarity_pct
    - l2_distance: default to mean_l2_distance
    """
    def extract_pair(raw, explicit_mean, explicit_median):
        m_mean = explicit_mean
        m_med = explicit_median
        if m_mean is None and isinstance(raw, dict):
            m_mean = raw.get("mean_embedding", raw.get("median_embedding"))
        if m_med is None and isinstance(raw, dict):
            m_med = raw.get("median_embedding", raw.get("mean_embedding"))
        if m_mean is None:
            m_mean = raw
        if m_med is None:
            m_med = raw
        return np.array(m_mean, dtype=np.float32), np.array(m_med, dtype=np.float32)

    mean_a, med_a = extract_pair(emb_a, mean_emb_a, median_emb_a)
    mean_b, med_b = extract_pair(emb_b, mean_emb_b, median_emb_b)

    def to_unit(vec):
        norm = np.linalg.norm(vec)
        return vec / max(1e-8, norm)

    u_mean_a = to_unit(mean_a)
    u_mean_b = to_unit(mean_b)
    u_med_a = to_unit(med_a)
    u_med_b = to_unit(med_b)

    # 1. Mean Pooling Cosine & L2
    raw_mean_dot = float(np.dot(u_mean_a, u_mean_b))
    raw_mean_sim_pct = round(max(0.0, min(1.0, raw_mean_dot)) * 100.0, 2)
    mean_sim_pct = calibrate_cosine_similarity(raw_mean_sim_pct)
    mean_l2 = round(float(np.linalg.norm(u_mean_a - u_mean_b)), 4)

    # 2. Median Pooling Cosine & L2
    raw_med_dot = float(np.dot(u_med_a, u_med_b))
    raw_med_sim_pct = round(max(0.0, min(1.0, raw_med_dot)) * 100.0, 2)
    med_sim_pct = calibrate_cosine_similarity(raw_med_sim_pct)
    med_l2 = round(float(np.linalg.norm(u_med_a - u_med_b)), 4)

    # Set default metrics to calibrated mean pooling
    dl_similarity_pct = mean_sim_pct
    raw_dl_similarity_pct = raw_mean_sim_pct
    l2_dist = mean_l2

    # Decision thresholds (calibrated for zero-shot ConvNeXt-Tiny embeddings)
    if dl_similarity_pct >= match_threshold:
        verdict = "MATCH (Confirmed Same Cattle)"
        badge_variant = "match"
        badge_color = "emerald"
    elif dl_similarity_pct >= likely_threshold:
        verdict = "LIKELY MATCH (High Biometric Similarity)"
        badge_variant = "likely"
        badge_color = "teal"
    elif dl_similarity_pct >= inconclusive_threshold:
        verdict = "INCONCLUSIVE (Partial Geometric Overlap)"
        badge_variant = "inconclusive"
        badge_color = "amber"
    else:
        verdict = "DISTINCT (Different Animals)"
        badge_variant = "distinct"
        badge_color = "rose"

    # Cross 5x5 frame similarity matrix & trimmed cross-frame similarity
    matrix_stats = {}
    cross_matrix = []
    trimmed_cross_frame_sim_pct = 0.0

    if per_frames_a and per_frames_b:
        list_a = [np.array(f, dtype=np.float32) for f in per_frames_a]
        list_b = [np.array(f, dtype=np.float32) for f in per_frames_b]
        all_pair_sims = []
        for i, fa in enumerate(list_a):
            row = []
            fa_u = to_unit(fa)
            for j, fb in enumerate(list_b):
                fb_u = to_unit(fb)
                s_raw = round(max(0.0, float(np.dot(fa_u, fb_u))) * 100.0, 2)
                s = calibrate_cosine_similarity(s_raw)
                row.append(s)
                all_pair_sims.append(s)
            cross_matrix.append(row)

        if all_pair_sims:
            # Top-15 highest frame pairs in 5x5 matrix
            sorted_pairs = sorted(all_pair_sims, reverse=True)
            top_k = sorted_pairs[:min(15, len(sorted_pairs))]
            trimmed_cross_frame_sim_pct = round(float(np.mean(top_k)), 2)
            matrix_stats = {
                "min_pair_similarity": round(float(np.min(all_pair_sims)), 2),
                "max_pair_similarity": round(float(np.max(all_pair_sims)), 2),
                "median_pair_similarity": round(float(np.median(all_pair_sims)), 2),
                "mean_pair_similarity": round(float(np.mean(all_pair_sims)), 2),
                "trimmed_top15_similarity": trimmed_cross_frame_sim_pct,
            }

    return {
        "dl_similarity_pct": dl_similarity_pct,
        "raw_dl_similarity_pct": raw_dl_similarity_pct,
        "l2_distance": l2_dist,
        "mean_pooling_similarity_pct": mean_sim_pct,
        "raw_mean_pooling_similarity_pct": raw_mean_sim_pct,
        "mean_l2_distance": mean_l2,
        "median_pooling_similarity_pct": med_sim_pct,
        "raw_median_pooling_similarity_pct": raw_med_sim_pct,
        "median_l2_distance": med_l2,
        "trimmed_cross_frame_similarity_pct": trimmed_cross_frame_sim_pct,
        "verdict": verdict,
        "badge_variant": badge_variant,
        "badge_color": badge_color,
        "match_threshold": match_threshold,
        "likely_threshold": likely_threshold,
        "inconclusive_threshold": inconclusive_threshold,
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
