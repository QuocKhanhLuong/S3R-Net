# Asymmetric Spec-HRNet: Comprehensive Technical Specification & Baseline Flow

## 1. Abstract & Objective
The primary objective of Asymmetric Spec-HRNet is to solve the medical image segmentation problem (specifically for ACDC - Cardiac MRI dataset) by resolving the trade-offs between local edge accuracy (Hausdorff Distance - HD95) and global structural consistency limits (Dice Score). We particularly aim to reduce False Positives in the Right Ventricle (RV) boundary, a zone characterized by high ambiguity and low contrast.

## 2. System Baseline Flow
The network ingests spatial and frequency priors via a 3-stream parallel paradigm:
1. **Prior Knowledge Input (`PriorKnowledgeConstructor`)**: Given a raw slice $I_{raw} \in \mathbb{R}^{1 \times H \times W}$, we expand it to $X_{prior} \in \mathbb{R}^{3 \times H \times W}$ consisting of `[Raw, Sobel Edge Magnitude, Local Spatial Variance]`.
2. **Feature Stem**: A single stride-1 Conv3x3 maps $X_{prior}$ to $C=48$ base dimension at 224×224 resolution.
3. **Stream Initiation**: Representations are down-sampled via strided convolutions into three streams operating concurrently: Full-Res (FR: $224^2$), High-Res (HR: $112^2$), and Low-Res (LR: $56^2$).
4. **Deep Supervision**: Intermediate semantic outputs are derived from fused stage 1 and stage 2 outputs to enforce rapid topological learning.

---

## 3. The Asymmetric Tri-Stream Architecture
Traditional multi-resolution nets (e.g., HRNet) use uniform operations across branches. We propose **Asymmetric Specialization**: each resolution branch executes entirely different mathematical operations with **Asymmetric Stage Depths** (2, 4, 6 blocks per stage). 

### 3.1. Stream 1: FR Stream with DCNv3 and HDC Dilation Pyramid
* **Motivation**: The Full-Resolution stream must trace boundaries of thin and irregular anatomical structures (RV walls). Standard Convolutions have rigidly square receptive fields, capturing irrelevant background noise.
* **Core Operation (`DCNv3Block`)**: DCNv3 computes both deformable spatial offsets $\Delta p_k$ and modulation scalars $m_k$. Given input $x \in \mathbb{R}^{C \times H \times W}$ and a target location $p_0$, the operation is:
  $$ y(p_0) = \sum_{g=1}^{G} \sum_{k=1}^{K^2} w_{g, k} \cdot m_{g, k} \cdot x_g(p_0 + p_k + \Delta p_{g, k}) $$
  where $G$ is the number of groups, ensuring computational efficiency.
* **Novel Contribution: HDC Dilation Pyramid**: 
  We modify the offsets natively using Hybrid Dilated Convolution (HDC). Across the 3 stages, we incrementally expand the dilation sequence $d_i$:
  - **Stage 1**: $d \in \{1, 2\}$
  - **Stage 2**: $d \in \{1, 2, 4, 8\}$
  - **Stage 3**: $d \in \{1, 2, 4, 8, 16, 32\}$
  *Mathematical Guarantee*: By ensuring that the greatest common divisor $\gcd(d_i, d_{i+1}) = 1$, HDC aggressively expands the deformable sampling area without generating "gridding artifacts" (dead pixels in the receptive field).

### 3.2. Stream 2: HR Stream with AdaptiveFourierMixer and Mode Pyramid
* **Motivation**: Spectral layers provide a $O(1)$ global receptive field without spatial reduction. However, computing FFT over raw images is highly susceptible to the **Gibbs Phenomenon (ringing artifacts)** near high-contrast edges.
* **Core Operation (`AdaptiveFourierMixer`)**: 
  We compute the 2D Real-Fast Fourier Transform $X_q = \text{RFFT2}(x)$. Instead of mixing all frequencies, we truncate frequencies up to a mode limit $M$:
  $$ \tilde{X_q}(u, v) = X_q(u,v) \odot W_{mode}(u,v) \quad \text{for } u \le M, v \le M $$
  The masked spectrum is mixed purely via linear projections across channels, then inverted back via $\text{IRFFT2}$.
* **Novel Contribution: Discrete Mode Pyramid**:
  Instead of a static frequency threshold, we dynamically throttle the frequency bandwidth allowed in the network per stage:
  - **Stage 1**: $M = H/8 = 14$ modes (Strict low-pass; learns global soft organ structure).
  - **Stage 2**: $M = H/4 = 28$ modes (Permits mid-frequencies).
  - **Stage 3**: $M = H/2 = 56$ modes (Unlocks high frequencies).
  *Result*: This acts as an intrinsic anti-ringing mechanism. The network stabilizes global context using low-frequency signals early, and only later utilizes high-frequency data for edge consolidation.

### 3.3. Stream 3: LR Stream with Mamba-CSGM and Scan Depth Pyramid
* **Motivation**: Traditional Transformers suffer from quadratic cost. Using State-Space Models (Mamba) reduces cost to $O(N)$ for long-range sequence context. Because Mamba scans 2D images as flattened 1D sequences, it historically struggles with "direction hallucination" where topological relationships break.
* **Core Operation (`CrossScanGatedMixer`)**: Uses selective scanning via 1-D casual Depth-Wise Conv.
* **Novel Contribution: Scan Depth Pyramid**:
  To protect topology at early layers, the cross-scan complexity is sequentially released:
  - **Stage 1**: `1-pass` (Forward $H$ only). Avoids heavy 1D restructuring of raw features.
  - **Stage 2**: `2-pass` (Bidirectional $H \uparrow \downarrow$). 
  - **Stage 3**: `4-pass` (Full bidirectional $H \updownarrow + W \leftrightarrow$). Allows absolute spatial correlation mapping once features are highly semantic at the dense 56x56 resolution.

---

## 4. Cross-Stream Structural Interactivity

### 4.1. Stage-Wise Asymmetric Dense Fusion (`TriFuseLayer`)
Standard HRNet enforces `all-to-all` fusion. We implement a **5-path connection matrix**. 
* **The Rule**: `LR -> FR` is intentionally severed.
* **Justification**: The LR stream ($56^2$) produces heavily compressed, highly semantical responses. Fusing this directly back into the precise FR ($224^2$) layer causes edge "blurring" and positional ambiguities. Thus, FR selectively queries only HR, while HR queries everything.

### 4.2. Asymmetric Skip-Attention Denoising
Noise behavior maps linearly to the applied block operation. Thus, before final fusion, each stream receives an algorithm-specific attention denoising layer:
1. **`FRSkipAttention` (Spatial Alignment)**: Employs a CBAM-lite architecture. Since DCNv3 operations are purely spatial, this eliminates classical MRI artifacts (e.g., patient movement ghosts, phase wrapping).
2. **`HRSkipAttention` (Frequency Calibration)**: Acknowledging that FFT models suffer from high-frequency energy leakage, this block computes energy weights along frequencies, attenuating frequencies that carry structural anomalies.
3. **`LRSkipAttention` (Uncertainty Gating)**: Mamba structures produce high hallucinations at ambiguous borders. We query the global variance channel and project an uncertainty score map, multiplying it by the Mamba outputs to suppress sequence hallucination.

### 4.3. Convergence: Frequency-Gated `TriStreamFusion`
To unify the multi-resolution output, the upscaled HR and LR features are not simply concatenated. Instead, the raw FR spatial representation serves as the anchoring prior matrix, computing independent `gate_hr` and `gate_lr` maps using a sigmoid projection:
$$ Out_{final} = FR + (HR_{up} \odot \sigma(Conv_{gate_{hr}}(FR))) + (LR_{up} \odot \sigma(Conv_{gate_{lr}}(FR))) $$
This strictly prohibits low-resolution artifacts from contaminating precise edge predictions in the final segmentation map.

---

## 5. Loss Optimization & Training Objective
Because the Right Ventricle natively suffers from false-positive over-segmentation, a rigorous penalized objective maps directly to our optimization metric `(Dice - 0.7 * HD95)`:

1. **Compound Objective**: 
   $$ \mathcal{L} = 0.5 \cdot \mathcal{L}_{CE} + 0.5 \cdot \mathcal{L}_{Focal(\gamma=2)} + 1.0 \cdot \mathcal{L}_{Dice} + 1.5 \cdot \mathcal{L}_{Boundary} $$
2. **RV Targeted Class Weighting**: 
   Focal and CE apply class weights: `[0.1, 2.5, 1.5, 1.0]` (BG, RV, MYO, LV). Setting RV to 2.5 severely penalizes false-positive confidence maps.
3. **Delayed Boundary Heating ($w_f$)**: 
   Boundary supervision measures Hausdorff distance gradients. It is set to $0.0$ from `epoch=0` to `20` to allow the unguided DCNv3 to find baseline topological offsets, before steadily scaling $w_f \to 1.0$ through `epoch=40` for hard edge refinement.
