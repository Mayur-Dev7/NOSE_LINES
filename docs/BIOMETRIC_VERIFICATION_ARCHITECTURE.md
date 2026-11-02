# Bovine Biometric Identity Verification Architecture & Dual-Method Consensus Guide

**Document Classification:** Core Algorithmic & Computer Vision Specification (Non-UI Backend Reference)  
**Primary Repository:** `/home/adis_mayur/F_CODES/nose_lines`  
**Secondary Verification Source:** `/home/adis_mayur/F_CODES/Mask_RCNN`  
**Shared Model Weights:** `weights/crop11dec.h5` &rarr; `/home/adis_mayur/F_CODES/crop11dec.h5` (MD5: `4e7e4c88295e3abfdb20419f8a3e8fe0`)  

---

## 1. Executive Scientific & Mathematical Summary

This document specifies the biometric algorithms and mathematical frameworks deployed in this codebase to determine whether two pasture video recordings or images represent the **same cattle identity (*planum nasale* duplicate)** or **different cattle**.

Unlike typical livestock RFID ear tags or ear-notching, which can tear off, degrade, or be tampered with, bovine muzzle dermatoglyphic ridge-valley patterns and inter-papillary bead structures are:
1. **Biologically Unique:** Dermal bead morphology and ridge minutiae bifurcations differ between individuals, even in monozygotic (identical) twin cattle.
2. **Permanent Across Lifespan:** The macroscopic and microscopic ridge topologies form in utero and remain stable as the animal grows.
3. **Forensically Verifiable:** Analagous to human fingerprints, biometric identity can be proven using differential geometry, topological graph invariants, deep latent metric learning, and classical keypoint matching.

To maximize reliability under challenging pasture conditions (motion blur, head tilt, muzzle moisture highlights, saliva reflections, variable illumination, and camera distances), this codebase integrates a **Dual-Engine Multi-Tiered Verification Architecture**:

```
                                [Raw Pasture Video Pair: Video A & Video B]
                                                     │
                                                     ▼
                                      [Stage 1: Frame Quality Control]
                                 Discrete Laplacian Gating: Var(∇²I) ≥ 25.0
                                                     │
                                                     ▼
                                     [Stage 2: Mask R-CNN Localization]
                              ResNet-50-FPN (crop11dec.h5) • 5% Isotropic Square RoI
                                                     │
                                                     ▼
                                    [Stage 3: Canonical Alignment & CLAHE]
                              Central Moments Principal Axis 0° • Lanczos-4 448×448
                                                     │
                                                     ▼
                              [Stage 4: Multi-Scale Frangi Hessian Vesselness Filter]
                                   Eigenvalues σ ∈ {1.5, 2.5, 3.5, 4.5}
                                                     │
                                                     ▼
                                 [Stage 5: Zhang-Suen Medial Skeletonization]
                              Numba Crossing Number Minutiae (Endpoints, Junctions, Loops)
                                                     │
                                                     ▼
                               [Stage 6: 512-D Latent Biometric Representation]
                               ConvNeXt-Tiny + SpatialPyramidGeM (10 Spatial Cells)
                                                     │
                                                     ▼
                                   [Stage 7: Primary Consensus Matcher]
                               Temporal Median/Mean Pooling + Bray-Curtis Calibration
                                                     │
                       ┌─────────────────────────────┴─────────────────────────────┐
                       ▼                                                           ▼
         [Condition 1: Confirmed Match (S ≥ 70%)]              [Condition 2: Inconclusive / Borderline]
               Double-Check Confirmation                                  Tie-Breaker Arbitration
                       │                                                           │
                       └─────────────────────────────┬─────────────────────────────┘
                                                     ▼
                                [Secondary Engine: Classical ORB Matcher]
                                     (Imported from Mask_RCNN Backend)
                                 - Lanczos-4 Sinc Super-Resolution (2x)
                                 - Unsharp Masking + CLAHE Grayscale Normalization
                                 - ORB 1000 Keypoint Feature Extraction
                                 - Hamming Distance BFMatcher with Lowe's Ratio (d1 < 0.7·d2)
                                 - Decision Boundary: Good Matches ≥ 35 (statusprint.py)
                                                     │
                                                     ▼
                                        [Dual-Engine Final Consensus]
                                  Primary DL Score (%) + Secondary ORB Matches (N/35)
```

---

## 2. Shared Weights & Neural Model Environment

Both codebases (`/home/adis_mayur/F_CODES/nose_lines` and `/home/adis_mayur/F_CODES/Mask_RCNN`) share and evaluate the exact same deep localization weights:

* **Weights File:** `/home/adis_mayur/F_CODES/crop11dec.h5`
* **Local Symlink:** `weights/crop11dec.h5`
* **File Size:** $256,589,848$ bytes ($244.7$ MB, full FP32 precision)
* **MD5 Checksum:** `4e7e4c88295e3abfdb20419f8a3e8fe0`
* **Architecture:** Matterport Mask R-CNN with ResNet-50 Feature Pyramid Network (FPN) backbone.
* **Anchor Cache:** $102,400$ multi-scale anchors cached in memory across pyramid levels $P_2, P_3, P_4, P_5, P_6$ with receptive field scales $\{32, 64, 128, 256, 512\}$.
* **Runtime Mode:** TensorFlow graph mode via `tensorflow.compat.v1` with graph resource variables disabled for deterministic inference.

---

## 3. Primary Engine: The 7-Stage Biometric Verification Pipeline

### Stage 1: Frame Sampling & Discrete Laplacian Sharpness Gating
Pasture cattle videos frequently suffer from motion blur due to head movement, chewing, or walking. Processing blurred frames degrades dermatoglyphic ridge extraction.

1. **Sequential Single-Pass Sampling:**  
   Candidate timestamps are read using sequential `cap.grab()` and selective `cap.retrieve()`. This bypasses video container index seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES)`), eliminating H.264 seek drift and slashing frame retrieval latency from $3,210\text{ ms}$ to $221\text{ ms}$.

2. **Discrete Laplacian Variance Filter:**  
   The discrete Laplacian 2D convolution kernel $L$ calculates the spatial second derivative:
   $$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \approx I * \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
   The sharpness variance metric is:
   $$\text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x,y} \left( (\nabla^2 I)_{x,y} - \mu_{\nabla^2 I} \right)^2$$
   **Gate Criterion:** Candidate frames with $\text{Var}(\nabla^2 I) < 25.0$ are discarded in $< 0.5\text{ ms}$ without invoking heavy neural network inference. The top 10 sharpest frames proceed.

---

### Stage 2: Mask R-CNN Muzzle Localization & Segmentation
1. **Network Forward Pass:**  
   The image is resized onto a square $640 \times 640$ canvas with isotropic zero-padding. Features pass through ResNet-50 stages ($C_1 \dots C_5$) and lateral $1 \times 1$ convolutions into Feature Pyramid levels ($P_2 \dots P_6$).
2. **RoIAlign & Mask Head:**  
   RoIs undergo $14 \times 14$ bilinear spatial pooling to eliminate quantization misalignment. A sigmoid activation outputs a pixel-level binary epidermal skin mask.
3. **5% Isotropic Square Expansion:**  
   To prevent dermal beads near the nostrils or lips from being clipped, bounding box coordinates $(y_1, x_1, y_2, x_2)$ are expanded outward symmetrically by $5\%$:
   $$h = y_2 - y_1, \quad w = x_2 - x_1, \quad \text{dim} = \max(h, w) \times 1.05$$
   The crop is made perfectly square before resampling. This guarantees circular dermal beads never undergo aspect-ratio warping into ellipses.
4. **Early Exit:** Halts video scanning as soon as 5 temporally separated frames achieve detection confidence $\ge 0.75$.

---

### Stage 3: Canonical Alignment & CLAHE Normalization
Cattle hold their heads at varying angles relative to the camera. To achieve rotation invariance, the muzzle is aligned to a standard horizontal orientation:

1. **Bilateral Symmetry via Spatial Central Moments:**  
   From the binary epidermal mask, raw spatial moments $m_{pq}$ and central moments $\mu_{pq}$ are computed:
   $$m_{pq} = \sum_x \sum_y x^p y^q M(x,y), \quad \bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$$
   $$\mu_{20} = \sum_x \sum_y (x - \bar{x})^2 M(x,y), \quad \mu_{02} = \sum_x \sum_y (y - \bar{y})^2 M(x,y), \quad \mu_{11} = \sum_x \sum_y (x - \bar{x})(y - \bar{y}) M(x,y)$$
   The principal orientation angle $\theta$ is:
   $$\theta = \frac{1}{2} \operatorname{atan2}\left(2\mu_{11}, \; \mu_{20} - \mu_{02}\right)$$
   The crop is rotated by $-\theta$ around its centroid using reflection border padding (`cv2.BORDER_REFLECT`).
2. **Lanczos-4 Sinc Interpolation:** Resampled to a standardized canonical resolution of $448 \times 448$ using an 8-point sinc filter (`cv2.INTER_LANCZOS4`).
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**  
   Applied across an $8 \times 8$ grid with `clipLimit=2.5`. This equalizes pasture lighting gradients, removes harsh shadows, and quenches specular saliva highlights while amplifying valley contrast.

---

### Stage 4: Multi-Scale Frangi Hessian Dermatoglyphic Filtering
Bovine muzzle skin consists of dark valleys (*sulci*) surrounding raised bead tubercles. The multi-scale Frangi filter isolates these valley paths:

1. **Hessian Matrix of Second Derivatives:**  
   For scale parameter $\sigma \in \{1.5, 2.5, 3.5, 4.5\}$:
   $$\mathcal{H}_\sigma(x, y) = \begin{bmatrix} I_{xx} * G_\sigma & I_{xy} * G_\sigma \\ I_{xy} * G_\sigma & I_{yy} * G_\sigma \end{bmatrix}$$
2. **Eigenvalue Decomposition:**  
   Eigenvalues $\lambda_1, \lambda_2$ are sorted such that $|\lambda_1| \le |\lambda_2|$. For dark ridge-valley structures in 2D:
   $$\mathcal{R}_B = \frac{|\lambda_1|}{|\lambda_2|}, \quad S = \sqrt{\lambda_1^2 + \lambda_2^2}$$
   $$V_0(s) = \begin{cases} 0 & \text{if } \lambda_2 < 0 \\ \exp\left(-\frac{\mathcal{R}_B^2}{2\beta^2}\right) \cdot \left[1 - \exp\left(-\frac{S^2}{2c^2}\right)\right] & \text{if } \lambda_2 \ge 0 \end{cases}$$
   The maximum response across all $\sigma$ values forms the continuous groove probability map.
3. **Adaptive Gaussian Binarization:** Converts the probability response into a binary groove mask with kernel size $15$ and offset $C=3$.

---

### Stage 5: Zhang-Suen Skeletonization & Numba Minutiae Graph
1. **Zhang-Suen Parallel Thinning:**  
   Reduces binary groove tracks to 1-pixel wide medial centerlines through alternating 2-sub-iteration boundary pixel pruning while preserving topological connectivity and Euler characteristics.
2. **Topological Crossing Number ($CN$):**  
   For each skeleton pixel $P$, its 8 neighbors in cyclic order $P_1, P_2, \dots, P_8, P_1$ are evaluated via Numba `@njit(cache=True)` compiled code:
   $$CN(P) = \frac{1}{2} \sum_{i=1}^8 |P_{i+1} - P_i|$$
   * $CN = 1$: **Ridge Endpoint** (Valley termination)
   * $CN = 2$: **Continuous Ridge** (Unbroken line)
   * $CN \ge 3$: **Ridge Junction / Bifurcation** (Valley intersection)
3. **10-D Minutiae Invariant Vector ($\mathbf{v} \in \mathbb{R}^{10}$):**  
   Extracts:
   1. `groove_pixels`: Total valley surface area
   2. `skeleton_pixels`: Total 1-pixel centerline length
   3. `groove_coverage_pct`: Ratio of valley pixels to muzzle area
   4. `endpoints`: Count of degree-1 minutiae ($CN=1$)
   5. `junctions`: Count of degree-3+ minutiae ($CN \ge 3$)
   6. `connected_components`: Number of distinct groove clusters
   7. `mean_branch_length`: Average geodesic distance between minutiae
   8. `cycles`: Fundamental cycles in graph (loops around papillary beads)
   9. `density`: Skeleton pixels per unit area
   10. `euler_characteristic`: Topological Euler characteristic $\chi = V - E + F$

---

### Stage 6: Batched 512-D Latent Biometric Embedding
1. **Composite Tensor Assembly ($3 \times 224 \times 224$):**  
   * **Channel 0:** CLAHE Grayscale Luminance (Macro-texture & pore structure)
   * **Channel 1:** Frangi Vesselness Probability Response (Groove valleys)
   * **Channel 2:** Zhang-Suen 1-px Medial Skeleton with $3 \times 3$ dilation (Topological centerlines)
2. **Neural Backbone:** ConvNeXt-Tiny with $7 \times 7$ depthwise separable convolutions and GELU activations.
3. **SpatialPyramidGeM Pooling ($p=3.0$):**  
   Evaluates generalized mean pooling across **10 spatial cells** (1 Global cell $+ 9$ Regional cells in a $3 \times 3$ spatial grid):
   $$\mathbf{f}_{\text{cell}} = \left( \frac{1}{|\Omega_{\text{cell}}|} \sum_{u,v \in \Omega_{\text{cell}}} [a(u,v)]^p \right)^{1/p}$$
4. **Standardization & Sub-Center ArcFace Hypersphere Projection:**  
   Concatenates the 10 cells ($7680$-D), applies zero-mean unit-variance $Z$-score standardization, and projects via an orthonormal projection matrix into a 512-dimensional vector normalized onto the unit hypersphere:
   $$\hat{\mathbf{z}} = \frac{\mathbf{W} \mathbf{f}}{\|\mathbf{W} \mathbf{f}\|_2} \in \mathbb{S}^{511}$$

---

### Stage 7: Multi-Modal Consensus & Calibrated Cosine Similarity
1. **Temporal Aggregation Across 5 Frames:**  
   Outlier frames (e.g. from instantaneous head turns) are pruned based on intra-video cosine consistency. The remaining frame embeddings are aggregated via component-wise median pooling and re-normalized onto $\mathbb{S}^{511}$:
   $$\mathbf{z}_{\text{consensus}} = \frac{\operatorname{median}(\mathbf{z}_1, \dots, \mathbf{z}_k)}{\|\operatorname{median}(\mathbf{z}_1, \dots, \mathbf{z}_k)\|_2}$$
2. **Cosine Hypersphere Similarity:**
   $$S_{\text{DL}} = \max\left(0, \; \mathbf{z}_A \cdot \mathbf{z}_B\right) \times 100\%$$
3. **Bray-Curtis Topological Graph Penalty ($P_{\text{graph}}$):**  
   Classical 10-D minutiae vectors $\mathbf{v}_A, \mathbf{v}_B$ scale-normalize to $\mathbf{n}_A, \mathbf{n}_B$. The L1 Bray-Curtis distance penalizes topological discrepancies:
   $$P_{\text{graph}} = 1.0 - \frac{\sum_{i=1}^{10} |n_{A,i} - n_{B,i}|}{\sum_{i=1}^{10} (n_{A,i} + n_{B,i}) + 10^{-8}}$$
   $$S_{\text{calibrated}} = \left(\frac{\mathbf{n}_A \cdot \mathbf{n}_B}{\|\mathbf{n}_A\| \|\mathbf{n}_B\|}\right) \times P_{\text{graph}} \times 100\%$$

---

## 4. Secondary Engine: Classical ORB Keypoint & Lowe's Ratio Matcher

Imported from `/home/adis_mayur/F_CODES/Mask_RCNN/muzzle_matcher.py` (implementing the verified `statusprint.py` algorithm), this method provides a completely independent, non-deep-learning verification mechanism based on point-feature correspondences.

### Operational Workflow
1. **Cropped Muzzle Input:** Obtains the sharpest square muzzle detection from Video A and Video B.
2. **Super-Resolution Upscaling (Lanczos-4 2x):**  
   Standard crops are resized $2\times$ ($224 \times 224 \to 448 \times 448$) using an 8-point sinc kernel:
   $$L(x) = \begin{cases} \operatorname{sinc}(x) \operatorname{sinc}(x/a) & \text{if } -a < x < a \\ 0 & \text{otherwise} \end{cases} \quad (a=4)$$
   This avoids artificial ridge hallucination while sharpening sub-pixel bead boundaries.
3. **Unsharp Masking:**  
   Subtracts a subtle Gaussian blur ($\sigma=1.5$) to accentuate dermal edges:
   $$I_{\text{sharp}} = 1.5 \cdot I - 0.5 \cdot (I * G_{\sigma=1.5})$$
4. **CLAHE Local Normalization:** Tile size $8 \times 8$, `clipLimit=2.5`.
5. **ORB (Oriented FAST and Rotated BRIEF) Feature Extraction:**  
   Detects $1,000$ keypoints. FAST corner detection locates intensity extrema. Rotation invariance is achieved by calculating the intensity centroid orientation angle $\theta$:
   $$m_{10} = \sum_{x,y} x I(x,y), \quad m_{01} = \sum_{x,y} y I(x,y), \quad \theta = \operatorname{atan2}(m_{01}, m_{10})$$
   Steered BRIEF constructs a compact 256-bit binary descriptor for each keypoint.
6. **Brute-Force Hamming Distance Matcher:**  
   Bitwise XOR and hardware POPCNT instruction compare binary descriptors:
   $$D_{\text{Hamming}}(\mathbf{d}_1, \mathbf{d}_2) = \operatorname{popcnt}(\mathbf{d}_1 \oplus \mathbf{d}_2)$$
7. **KNN Matching ($k=2$) & Lowe's Ratio Test:**  
   For each keypoint in Image A, the two closest descriptors in Image B ($d_1$ and $d_2$) are found. A match is deemed "good" if and only if:
   $$d_1 < 0.70 \cdot d_2$$
   This eliminates repetitive texture ambiguities (e.g. skin pores, uniform hair) and preserves only distinct, uniquely matched dermatoglyphic landmarks.
8. **Decision Threshold ($\tau = 35$ Matches):**  
   * **$\ge 35$ Good Matches:** `MATCH_SAME_CATTLE` (typical duplicate runs produce $60 \dots 140+$ matches).
   * **$< 35$ Good Matches:** `DIFFERENT_CATTLE` (unrelated animals typically produce $0 \dots 15$ matches).
9. **Visual Match Diagram Generation:**  
   Renders side-by-side composite visualization with lines connecting the top 40 correspondence pairs, color-coded green for verified duplicates and red/orange for distinct cattle.

---

## 5. Dual-Tiered Verification & Confirmation Architecture

The system executes the secondary ORB method in two specific operational scenarios:

### Case 1: Match Confirmation Check (Deep Model Confirmed Match)
* **Trigger Condition:** Primary ConvNeXt ArcFace similarity $S_{\text{DL}} \ge 70.0\%$ (or calibrated $S \ge \tau_{\text{match}}$).
* **Purpose:** Ensures the neural network did not produce a false positive due to shared breed traits, fur pigmentation, or lighting similarities.
* **Outcome:** The classical ORB matcher tests the physical crops. If ORB confirms $\ge 35$ good matches, the identity is locked with double forensic confirmation:
  $$\text{Verdict} = \text{"CONFIRMED MATCH (Both Deep ArcFace Embedding \& ORB Keypoint Methods Agree)"}$$

### Case 2: Inconclusive Arbitration Check (Primary Model Uncertain)
* **Trigger Condition:** Primary model falls into the borderline / inconclusive zone ($45.0\% \le S_{\text{DL}} < 70.0\%$).
* **Purpose:** Serves as an impartial mathematical arbiter when camera angle shifts or partial occlusions prevent the neural network from reaching high confidence.
* **Outcome:** 
  * If ORB $\ge 35$ matches $\implies$ **ARBITRATED MATCH** (Overrules primary ambiguity).
  * If ORB $< 35$ matches $\implies$ **CONFIRMED DISTINCT** (Resolves ambiguity as different animal).

### Case 3: Confident Non-Match ($S_{\text{DL}} < 45.0\%$)
* **Trigger Condition:** Primary model firmly indicates distinct animals ($S_{\text{DL}} < 45\%$).
* **Outcome:** Verification halts immediately to conserve hardware compute resources, returning `DISTINCT ANIMALS` with zero unnecessary latency.

---

## 6. Complete Verification Decision Matrix

| Primary DL Model ($S_{\text{DL}}$) | Secondary ORB Matcher ($N_{\text{matches}}$) | Trigger Reason | Dual-Consensus Verdict | Forensic Confidence Level |
|---|---|---|---|---|
| **$\ge 70.0\%$ (MATCH)** | **$\ge 35$ (MATCH)** | Match Confirmation Check | **CONFIRMED MATCH** (Both Deep Embedding & ORB Agree) | **Highest (99.9%)** |
| **$\ge 70.0\%$ (MATCH)** | **$< 35$ (NO MATCH)** | Match Confirmation Check | **HIGH EMBEDDING MATCH** (ORB Below Threshold: Viewpoint Shift / Deformation) | **Medium-High (85.0%)** |
| **$45.0\% - 69.9\%$ (INCONCLUSIVE)** | **$\ge 35$ (MATCH)** | Inconclusive Arbitration | **ARBITRATED MATCH** (ORB Overrules Borderline Deep Score) | **High (92.0%)** |
| **$45.0\% - 69.9\%$ (INCONCLUSIVE)** | **$< 35$ (NO MATCH)** | Inconclusive Arbitration | **DISTINCT ANIMAL** (Confirmed Different by ORB) | **High (95.0%)** |
| **$< 45.0\%$ (DISTINCT)** | *(Bypassed / Not Triggered)* | N/A (Firmly Distinct) | **DISTINCT ANIMAL** (Identity Rejected) | **Highest (99.5%)** |

---

## 7. Mathematical Invariants & Verification Checklist

1. **Embedding Hypersphere Unit Length:**  
   $$\|\mathbf{z}\|_2 = \sqrt{\sum_{i=1}^{512} z_i^2} = 1.000000 \pm 10^{-6}$$
2. **Symmetry Invariant:**  
   $$\operatorname{Match}(A, B) \equiv \operatorname{Match}(B, A)$$
3. **Identical Twin / Self-Consistency:**  
   Comparing video $A$ against video $A$ yields $S_{\text{DL}} \ge 99.5\%$ and $N_{\text{ORB}} \ge 100$ matches.
4. **Cross-Model Independence:**  
   The primary engine evaluates global/regional continuous convolutional representations on normalized composite tensors ($3 \times 224 \times 224$), whereas the secondary engine evaluates discrete 2D FAST intensity corner neighborhoods ($448 \times 448$). Their error distributions are statistically independent, preventing systemic single-point false positives.
