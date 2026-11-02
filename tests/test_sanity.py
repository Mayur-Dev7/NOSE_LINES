#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable execve re-exec in muzzle_cropper for test script
os.environ["MRCNN_NVIDIA_CONFIGURED"] = "1"

print("1. Testing TensorFlow import...")
import tensorflow as tf
print("   TF version:", tf.__version__)

print("2. Testing PyTorch import...")
import torch
print("   PyTorch version:", torch.__version__, "CUDA available:", torch.cuda.is_available())

print("3. Testing timm import...")
import timm
print("   timm version:", timm.__version__)

print("4. Testing muzzle_cropper import...")
import muzzle_cropper
print("   muzzle_cropper loaded cleanly!")

print("5. Testing dl_muzzle_embedder import...")
import dl_muzzle_embedder
print("   dl_muzzle_embedder loaded cleanly!")

print("\n--- ALL IMPORTS PASSED WITHOUT CONFLICTS ---")
