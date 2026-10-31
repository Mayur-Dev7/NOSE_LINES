# PHASE 0 AUDIT & BASELINE BENCHMARK REPORT
**Bovine Biometric Verification Pipeline Optimization**  
**Repository**: `/home/adis_mayur/F_CODES/nose_lines`  
**Target Hardware**: AMD Ryzen 5 5600H (6 Physical Cores / 12 Hardware Threads, Base 3.3 GHz, Boost 4.2 GHz), NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6, GA107, Ampere Compute Capability 8.6), 16 GB DDR4 RAM.  
**Environment**: Ubuntu Linux x86_64, Python 3.8.20 venv (`/home/adis_mayur/F_CODES/nose_lines/.venv`).  
**Date**: November 1, 2026  

---

## 1. Executive Summary & Optimization Roadmap

Phase 0 audits the 7-stage bovine biometric verification pipeline across three target profiles:
1. **GPU Profile** (`device='gpu'`, 6 physical cores, CUDA / cuDNN acceleration for Mask R-CNN and ConvNeXt-Tiny)
2. **CPU-6 Profile** (`device='cpu'`, 6 physical cores / 6 worker threads, intra=6 / inter=2)
3. **CPU-2 Profile** (`device='cpu'`, 2 physical cores / 2 worker threads, intra=2 / inter=2)

### Stage Share Ranking (> 5% Threshold for Optimization Eligibility)
Per prompt instructions, optimizations are prioritized strictly by measured payoff, and stages contributing less than 5% of total pipeline latency are skipped:

| Rank | Pipeline Stage | GPU Share (%) | CPU-6 Share (%) | CPU-2 Share (%) | Median Wall-Clock (ms) | Actionable Optimization Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Step 4: Frangi Hessian Grooves** | **32.0%** | **30.3%** | **29.2%** | **3,754 – 3,806 ms** | **Opt 3**: Concurrent execution across persistent spawn process pool (`cv_workers.py`). |
| **2** | **Step 5: Ridge Skeleton & Topology** | **25.6%** | **24.7%** | **23.7%** | **3,037 – 3,076 ms** | **Opt 3 & 4**: Concurrent pool execution + Numba `@njit(cache=True)` spur pruning and crossing numbers. |
| **3** | **Step 2: Mask R-CNN Localization** | **22.4%** | **21.8%** | **21.2%** | **2,657 – 2,730 ms** | **Opt 3 & 6**: Pipelining Video B detection with Video A Steps 3–5 (GPU-only); CPU thread verification. |
| **4** | **Step 1: Frame Sampling & Gating** | **17.1%** | **16.0%** | **15.8%** | **1,993 – 2,033 ms** | **Opt 2**: Single-threaded sequential grab/retrieve sampling; keep top-10 decoded frames in memory. |
| **5** | **Step 6: ConvNeXt Biometric Embedder** | 0.9% *(skip GPU)* | **5.2%** | **7.8%** | **644 – 1,008 ms** | **Opt 5**: Batched forward pass $(5, 3, 224, 224)$, `inference_mode()`, ONNX Runtime CPU provider evaluation. |
| **-** | Step 3: Canonical Alignment & CLAHE | 2.0% | 2.0% | 2.2% | 241 – 279 ms | *Skipped (< 5% share)* |
| **-** | Step 7: Biometric Consensus & Match | 0.0% | 0.0% | 0.0% | 0.9 – 1.0 ms | *Skipped (< 5% share)* |

**Key Architectural Insight**: Steps 4 and 5 (Frangi Hessian Filtering + Zhang-Suen Skeletonization + Graph Topology) account for **53% to 57% of total verification latency (~6.8 seconds)** because the 5 frames are executed sequentially on a single thread. By moving Steps 3–5 into a persistent, pre-forked spawn process pool (`cv_workers.py`), these frames execute concurrently across all physical CPU cores, dropping latency by an estimated ~2.5–3.5 seconds.

---

## 2. Complete Call Graph & Stage Breakdown (Steps 1–7)

```
Two Video Inputs: Video A (muzzleVideo.mp4) & Video B (muzzleVideo2.mp4)
 │
 ├── [Step 1: Frame Sampling & Quality Gating] (video_groove_extractor.py:612-663)
 │    ├── cv2.VideoCapture(v_path)
 │    ├── cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx) -> Random seek across 30 linspace points
 │    ├── cv2.cvtColor(BGR2GRAY) + cv2.Laplacian(gray, cv2.CV_64F).var()
 │    └── Temporal bucketing (10 bins) -> Pick best per bin -> Top 10 sharp candidate frames
 │
 ├── [Step 2: Mask R-CNN Muzzle Localization] (muzzle_cropper.py & video_groove_extractor.py:664-820)
 │    ├── MaskRCNNDetector.process_image() [crop11dec.h5, ResNet-50-FPN, TF 2.13.1]
 │    ├── Early exit: Iterates through candidates; stops as soon as 5 high-conf (>=0.75) frames are found
 │    └── 5% margin padding + square aspect ratio expansion -> Native crop & mask stored in memory
 │
 ├── [Step 3: Canonical Alignment & CLAHE Normalization] (canonical_alignment.py & groove_extractor.py)
 │    ├── Spatial central moments (cv2.moments) -> Bilateral symmetry axis (0 deg horizontal)
 │    ├── 2x Lanczos-4 resampling -> 448x448 canonical resolution
 │    └── cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) contrast enhancement
 │
 ├── [Step 4: Dermatoglyphic Groove Extraction] (groove_extractor.py:240-360)
 │    ├── skimage.filters.frangi (multi-scale Hessian vesselness, sigma in {1.5, 2.5, 3.5, 4.5}, float64)
 │    └── Adaptive Gaussian binarization (cv2.adaptiveThreshold, blockSize=15, C=2)
 │
 ├── [Step 5: Topological Skeletonization & Minutiae] (groove_extractor.py & groove_graph.py)
 │    ├── skimage.morphology.skeletonize (Zhang-Suen iterative morphological thinning)
 │    ├── Pure Python spur pruning (prune_skeleton: loops over endpoints, BFS deque < 8px)
 │    └── Crossing number minutiae classification (pure Python 8-neighbor lookup)
 │
 ├── [Step 6: 512-D Latent Biometric Embedding] (dl_muzzle_embedder.py:425-580)
 │    ├── 3-channel composite tensor: Ch0=CLAHE Gray, Ch1=Groove Response, Ch2=3x3 Dilated Skeleton
 │    ├── ConvNeXt-Tiny forward pass -> SpatialPyramidGeM (p=3.0, 10 cells)
 │    └── Deterministic orthonormal projection -> L2-normalized 512-D unit hypersphere vector
 │
 └── [Step 7: Biometric Consensus & Verification] (video_groove_extractor.py:240-330)
      ├── Temporal 5-frame aggregation (median pooling & mean pooling)
      ├── Pairwise Cosine Similarity (max(0, e_A . e_B) * 100%)
      ├── Bray-Curtis topological magnitude penalty factor
      └── S_calibrated = S_dl * (1 - penalty). Thresholds: >=85% MATCH, >=75% INCONCLUSIVE, <75% NO MATCH
```

---

## 3. Deep Architectural Audit Findings

### 3.1 Mask R-CNN Configuration & Threading
- **Weights File**: `/home/adis_mayur/F_CODES/nose_lines/weights/crop11dec.h5`
- **Backbone**: ResNet-50-FPN
- **Input Resolution**: `IMAGE_MIN_DIM = 640`, `IMAGE_MAX_DIM = 640`, `IMAGE_SHAPE = [640, 640, 3]`
- **Inference ROIs**: `POST_NMS_ROIS_INFERENCE = 50` (optimized from baseline 1000)
- **Batch Size**: `BATCH_SIZE = 1`
- **Max Detections**: `DETECTION_MAX_INSTANCES = 100`
- **Min Confidence**: `DETECTION_MIN_CONFIDENCE = 0.70`
- **TensorFlow Thread Allocation**:
  - `GPU Profile`: `CUDA_VISIBLE_DEVICES="0"`, dynamic GPU memory allocation enabled (`set_memory_growth=True`).
  - `CPU-6 Profile`: `CUDA_VISIBLE_DEVICES=""`, `intra_op_parallelism_threads = 6`, `inter_op_parallelism_threads = 2`.
  - `CPU-2 Profile`: `CUDA_VISIBLE_DEVICES=""`, `intra_op_parallelism_threads = 2`, `inter_op_parallelism_threads = 2`.

### 3.2 Frangi & Skeletonize Implementation & Data Types
- **Frangi Filter**: Direct call to `skimage.filters.frangi(image, sigmas=[1.5, 2.5, 3.5, 4.5], black_ridges=True, mode='reflect', cval=0)` in `groove_extractor.py:270`.
  - Input array converted via `image = np.asarray(image, dtype=np.float64) / 255.0` (`float64`).
  - Frangi filter internally constructs Hessian matrices across all 4 scales using `scipy.ndimage.gaussian_filter`.
- **Skeletonize**: Direct call to `skimage.morphology.skeletonize(binary_grooves > 0)` in `groove_extractor.py:430`.
- **Spur-Pruning & Crossing-Number Loops**:
  - Implemented as pure Python `for` loops in `groove_graph.py:65` (`prune_skeleton`) and `groove_graph.py:168` (`_crossing_number`).
  - Iterates over Python dictionaries and sets pixel by pixel, incurring significant CPython interpreter overhead.

### 3.3 Frame Decoding: Step 1 vs Step 2 Analysis
- **Step 1 Decoding**: Reads 30 candidate frames by repeatedly setting `cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)` followed by `cap.read()`. In H.264 video decoding, random seeks require seeking to the preceding IDR/I-frame and decoding all intermediate P/B-frames, which accounts for the ~2,000 ms latency.
- **Step 2 Decoding**: **Step 2 does NOT re-decode the video**. The decoded candidate `frame_bgr` numpy array is retained in the `detector_candidates` dictionary in RAM during Step 1 (`video_groove_extractor.py:630`) and directly passed into `cropper.process_image(frame_bgr)` in Step 2.

### 3.4 Video Input Specifications
- **Video A (`muzzleVideo.mp4`)**:
  - Dimensions: $720 \times 960$ (Width $\times$ Height)
  - Frame Rate: 6.00 FPS
  - Frame Count: 85 frames
  - Duration: 14.2 seconds
  - Format / Codec: MP4 / AVC H.264 (High @ L3.1)
- **Video B (`muzzleVideo2.mp4`)**:
  - Dimensions: $720 \times 960$
  - Frame Rate: 6.00 FPS
  - Frame Count: 85 frames
  - Duration: 14.2 seconds
  - Format / Codec: MP4 / AVC H.264 (High @ L3.1)

### 3.5 Server Deployment & Execution Mode
- **Server Framework**: FastAPI / Starlette
- **ASGI Server**: `uvicorn`
- **Launch Command**: `uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)`
- **Workers**: 1 worker process, reloader explicitly disabled (`reload=False`).

### 3.6 Software & Library Versions
- **Python**: `3.8.20` (`GCC 11.2.0`)
- **TensorFlow**: `2.13.1`
- **PyTorch**: `2.1.2+cu121`
- **OpenCV (`cv2`)**: `5.0.0`
- **Numba**: `0.58.1`
- **llvmlite**: `0.41.1`
- **ONNX Runtime**: `1.17.0`
- **scikit-image**: `0.21.0`
- **NumPy**: `1.24.3`

### 3.7 OpenMP Runtime Inspection (`/proc/<pid>/maps`)
Inspection of the running server process memory maps reveals **exactly one** OpenMP runtime is loaded:
- Path: `/home/adis_mayur/F_CODES/nose_lines/.venv/lib/python3.8/site-packages/torch/lib/libgomp-a34b3233.so.1` (GNU OpenMP).
- Neither `libiomp5.so` (Intel OpenMP) nor `libomp.so` (LLVM OpenMP) is loaded.
- There is **zero dual-OpenMP runtime contention**.
- Preloading of `libnvblas.so` has been explicitly disabled, preventing cuBLAS/NVBLAS segfaults and non-finite float JSON generation on CPU profiles.

---

## 4. Step 6 Isolation Test Results

To isolate whether the latency in Step 6 (ConvNeXt-Tiny Biometric Embedder) stems from cross-framework thread interference (e.g., TensorFlow 2.13 and PyTorch 2.1 competing for OpenMP thread locks) or pure compute overhead, we benchmarked the embedder under two separate execution environments:
1. **Fresh Isolated Python Process**: Torch only (no TensorFlow, no OpenCV, no Uvicorn imported), batch of 5 tensors $(5, 3, 224, 224)$.
2. **Server Process Post-Step 2**: The current sequential frame-by-frame invocation inside `video_groove_extractor.py` immediately after Mask R-CNN.

### Isolation Test Measurement Table

| Profile | Threads / Hardware | Isolated Process (Batch=5) | Server Process (Sequential Batch=1 $\times$ 5) | Latency Gap (Interference + Overhead) | Root Cause Analysis |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **GPU** | RTX 3050 (4 GB) | **22.93 ms** | **105.74 ms** | 82.81 ms (4.6x) | Kernel launch & host-to-device transfer overhead of 5 individual calls vs 1 batched call. |
| **CPU-6** | 6 Cores (6 Threads) | **233.26 ms** | **644.55 ms** | 411.29 ms (2.8x) | PyTorch intra-op thread barrier and per-layer overhead repeated 5 times. |
| **CPU-2** | 2 Cores (2 Threads) | **411.05 ms** | **1,008.60 ms** | 597.55 ms (2.45x) | CPU memory bandwidth and serial GEMM overhead. |

**Verdict**: The isolated process runs in **233 ms on CPU-6** and **411 ms on CPU-2**. The server process is currently taking **644 ms** and **1,008 ms** primarily because `video_groove_extractor.py` calls the embedder sequentially frame-by-frame (`batch=1`) rather than batching the 5 frames into a single tensor $(5, 3, 224, 224)$. Batching in Optimization 5 will directly unlock this 2.5x–2.8x speedup.

---

## 5. Phase 0 Official Baseline Benchmark Results

The baseline harness (`benchmark_pipeline.py`) was executed under exclusive lock (`flock /tmp/nose_bench.lock`) across all three target hardware profiles. Protocol: 1 warm-up comparison, followed by 3 timed runs. Golden artifacts (embeddings, P_graph JSON, detector parameters, S_calibrated) were preserved in `golden_artifacts/`.

### Profile 1: GPU (RTX 3050 Laptop GPU + 6 CPU Cores)
- **Total Wall-Clock Latency**: **12.09 s (Median)** / **12.02 s (Mean)**

| Stage | Median Latency (ms) | Mean Latency (ms) | Share of Time (%) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Step 4: Frangi Hessian Grooves** | **3,806.2 ms** | 3,785.9 ms | **32.0%** | Target for Optimization |
| **Step 5: Ridge Skeleton & Topology** | **3,037.8 ms** | 3,117.6 ms | **25.6%** | Target for Optimization |
| **Step 2: Mask R-CNN Localization** | **2,657.7 ms** | 2,679.3 ms | **22.4%** | Target for Optimization |
| **Step 1: Frame Sampling & Gating** | **2,031.8 ms** | 2,019.6 ms | **17.1%** | Target for Optimization |
| Step 3: Canonical Alignment & CLAHE | 241.8 ms | 239.5 ms | 2.0% | Skipped (< 5%) |
| Step 6: 512-D Latent Biometric Embedding | 105.7 ms | 104.3 ms | 0.9% | Skipped on GPU (< 5%) |
| Step 7: Biometric Consensus & Match | 0.9 ms | 0.9 ms | 0.0% | Skipped (< 5%) |

### Profile 2: CPU-6 (AMD Ryzen 5 5600H, 6 Physical Cores / 6 Threads)
- **Total Wall-Clock Latency**: **12.56 s (Median)** / **12.59 s (Mean)**

| Stage | Median Latency (ms) | Mean Latency (ms) | Share of Time (%) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Step 4: Frangi Hessian Grooves** | **3,770.5 ms** | 3,770.4 ms | **30.3%** | Target for Optimization |
| **Step 5: Ridge Skeleton & Topology** | **3,076.0 ms** | 3,160.2 ms | **24.7%** | Target for Optimization |
| **Step 2: Mask R-CNN Localization** | **2,710.0 ms** | 2,714.2 ms | **21.8%** | Target for Optimization |
| **Step 1: Frame Sampling & Gating** | **1,993.2 ms** | 1,990.2 ms | **16.0%** | Target for Optimization |
| **Step 6: 512-D Latent Biometric Embedding** | **644.6 ms** | 635.8 ms | **5.2%** | Target for Optimization |
| Step 3: Canonical Alignment & CLAHE | 252.0 ms | 253.7 ms | 2.0% | Skipped (< 5%) |
| Step 7: Biometric Consensus & Match | 1.0 ms | 1.1 ms | 0.0% | Skipped (< 5%) |

### Profile 3: CPU-2 (2 Cores / 2 Worker Threads)
- **Total Wall-Clock Latency**: **12.86 s (Median)** / **12.88 s (Mean)**

| Stage | Median Latency (ms) | Mean Latency (ms) | Share of Time (%) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Step 4: Frangi Hessian Grooves** | **3,754.5 ms** | 3,745.0 ms | **29.2%** | Target for Optimization |
| **Step 5: Ridge Skeleton & Topology** | **3,050.9 ms** | 3,056.0 ms | **23.7%** | Target for Optimization |
| **Step 2: Mask R-CNN Localization** | **2,730.7 ms** | 2,723.7 ms | **21.2%** | Target for Optimization |
| **Step 1: Frame Sampling & Gating** | **2,033.2 ms** | 2,007.8 ms | **15.8%** | Target for Optimization |
| **Step 6: 512-D Latent Biometric Embedding** | **1,008.6 ms** | 1,007.4 ms | **7.8%** | Target for Optimization |
| Step 3: Canonical Alignment & CLAHE | 279.1 ms | 279.0 ms | 2.2% | Skipped (< 5%) |
| Step 7: Biometric Consensus & Match | 1.0 ms | 1.0 ms | 0.0% | Skipped (< 5%) |

---

## 6. Multi-Agent Work Allocation

To implement the target optimizations concurrently without collision, the workload is distributed across three autonomous agents adhering to strict branch separation and regression gating:

```
                          ┌───────────────────────────┐
                          │   Agent 1: Orchestrator   │
                          │   Audit, Benchmarks,      │
                          │   Regression Gate, Merge  │
                          └─────────────┬─────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
  ┌───────────────────────────┐                   ┌───────────────────────────┐
  │   Agent 2: Pipeline Core  │                   │   Agent 3: Runtime & DL   │
  │   - Opt 2: Step 1 Grab    │                   │   - Opt 1: OpenMP Hygiene │
  │   - Opt 3: Spawn Pool     │                   │   - Opt 5: Batched / ONNX │
  │   - Opt 4: Numba Pruning  │                   │   - Opt 6: Mask R-CNN CPU │
  └───────────────────────────┘                   └───────────────────────────┘
```

1. **Agent 1 (Orchestrator / Audit / Integration)**:
   - Authored Phase 0 Report and isolation test.
   - Maintains `benchmark_pipeline.py`, `regression_checker.py`, and `golden_artifacts/`.
   - Executes regression validation under `flock /tmp/nose_bench.lock` before merging each branch into `main`.
   - Records official before/after tables and compiles `OPTIMIZATION_NOTES.md`.

2. **Agent 2 (Optimizations 2, 3, 4)**:
   - Branch: `opt/agent2-sampling-pool-numba`
   - Optimization 2: Single-threaded sequential grab/retrieve sampling in Step 1; keep top-10 decoded frames in memory.
   - Optimization 3: Persistent spawn process pool for Steps 3–5 (`cv_workers.py` importing only numpy/cv2/skimage; pre-forked before CUDA/TF init). Overlap Video B Mask R-CNN with Video A Steps 3–5 on GPU only.
   - Optimization 4: Numba `@njit(cache=True)` compilation of spur pruning and crossing numbers in `groove_graph.py`.

3. **Agent 3 (Optimizations 1, 5, 6)**:
   - Branch: `opt/agent3-threads-embedder-detector`
   - Optimization 1: Thread hygiene (`KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE`, strictly clamp CPU-2 to 2 threads and CPU-6 to 6 threads).
   - Optimization 5: Step 6 batched forward pass $(5, 3, 224, 224)$, `torch.inference_mode()`, `channels_last`, `set_flush_denormal(True)`, evaluate ONNX Runtime fp32 CPU provider under $\ge 0.9999$ cosine gate.
   - Optimization 6: Mask R-CNN CPU tuning (oneDNN verification, graph freeze/optimization).
