# SolSight — Technical Architecture & Machine Learning Write-Up

## Executive Summary

SolSight is an unsupervised, resolution-agnostic automatic optical inspection (AOI) engine for printed circuit board assembly (PCBA) solder joints. By integrating a fully convolutional autoencoder (CAE) with a custom structural dissimilarity (DSSIM) scoring pipeline, SolSight detects micro-voids, solder bridges, cold joints, and abnormal solder quantities without ever training on a single defective sample.

---

## 1. Resolution-Agnostic Design Rationale

Traditional computer vision autoencoders rely on `Flatten()` and `Dense()` (fully-connected) layers at the bottleneck to compress spatial feature maps into fixed-size latent vectors. This architecture imposes rigid input dimension constraints: every incoming image patch must be resized or cropped to identical $H \times W$ dimensions. In electronics manufacturing, solder joints naturally range from tiny 0201/01005 passive terminations (~15–20 pixels under typical camera magnification) to large QFP, BGA, and power transistor pads (128+ pixels). Enforcing standard input dimensions via interpolation introduces artificial blur, ringing artifacts, or distortion on fine solder fillets.

SolSight eliminates this limitation through a **fully convolutional architecture**:
- **No Dense Layers:** The encoder, bottleneck, and decoder consist exclusively of $3\times3$ convolutions, ReLU activations, and $2\times$ nearest-neighbor upsamplers.
- **Factor-Based Downsampling:** Rather than downsampling to a fixed spatial size (e.g. $1\times1$), the network downsamples by a fixed spatial *factor* ($4\times$, via two stride-2 convolutions). A 128px patch reaches a $32\times32$ latent bottleneck, preserving rich 2D spatial arrangement, while a 16px patch reaches a $4\times4$ bottleneck, preserving minimum structural topology.
- **Odd-Size Center Cropping:** For non-power-of-two patches (e.g. 15px or 17px), stride-2 operations round spatial dimensions upward (`ceil(15/2) = 8`, `ceil(8/2) = 4`), which upon $4\times$ upsampling yields 16px. SolSight computes original input dimensions at runtime and center-crops the decoder output to match $(H, W)$ exactly.

---

## 2. Loss & Anomaly Signal: SSIM Over Pixelwise MSE

Conventional autoencoder anomaly detectors optimize Mean Squared Error (MSE):
$$\text{MSE}(x, \hat{x}) = \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W (x_{ij} - \hat{x}_{ij})^2$$

In solder inspection, MSE suffers from two fundamental deficiencies:
1. **Blurry Reconstructions Rewarded:** Pixelwise losses average independent intensity differences, encouraging autoencoders to output blurry, smoothed fillets that minimize average $L_2$ distance while erasing fine high-frequency details.
2. **Poor Domain Alignment:** Solder defects are defined by localized disruptions in **geometry, surface reflection, and specular curvature** rather than uniform color shifts. A cold solder joint, for example, has roughly the same mean grayscale intensity as a normal joint, but its crystalline micro-texture and lack of clean specular highlight are completely invisible to MSE.

SolSight uses **Structural Similarity (SSIM)**:
$$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$
where $\mu, \sigma^2, \sigma_{xy}$ represent local windowed luminance, contrast, and structural covariance. The training loss is defined as:
$$\mathcal{L} = 1 - \text{SSIM}(x, \hat{x})$$
bounded strictly by setting `data_range=1.0` on Sigmoid-normalized activations.

---

## 3. Deliberate Absence of Skip Connections

Modern U-Net architectures utilize skip connections (concatenating encoder feature maps directly into decoder layers) to recover fine spatial details. In defect detection, skip connections are catastrophic: high-frequency defect textures (such as a void or bridge) bypass the compressed bottleneck entirely and are reconstructed with high fidelity.

SolSight deliberately omits all skip connections. Every visual detail must pass through the $4\times$ compressed bottleneck representation. Because the network is trained exclusively on normal golden references, its latent manifold only encodes the geometry and specular characteristics of sound solder joints. When presented with an anomaly, the decoder projects the patch back into the nearest "normal" joint geometry, creating a strong structural mismatch where the defect was present.

---

## 4. Top-5% DSSIM Anomaly Scoring

In production PCBA inspection, a critical defect may be an isolated micro-void or pinhole covering only 1% to 3% of the solder fillet. If anomaly scores were computed by taking the global mean of the dissimilarity map:
$$\text{Score}_{\text{mean}} = \frac{1}{HW}\sum_{i,j} \text{DSSIM}_{ij}$$
the intense localized dissimilarity of the defect is diluted by the 97% of the patch that is completely normal, resulting in false negatives.

SolSight applies **Top-5% Percentile Pooling**:
$$\text{Score}_{\text{top5\%}} = \frac{1}{|\Omega_{0.05}|}\sum_{(i,j) \in \Omega_{0.05}} \text{DSSIM}_{ij}$$
where $\Omega_{0.05}$ denotes the set of coordinates representing the top 5% highest values in the spatial DSSIM map $\text{DSSIM} = \frac{1 - \text{SSIM}}{2}$. This ensures that localized structural disruptions produce an unmistakable peak signal regardless of patch size.

---

## 5. Custom SSIM Implementation with Unreduced Spatial Maps

Standard public libraries (e.g. `pytorch_msssim`) reduce intermediate window calculations into a scalar batch loss. However, anomaly localization and heatmap visualization require the **unreduced 2D spatial similarity tensor** $(B, 1, H, W)$.

SolSight implements a custom 2D convolution-based SSIM engine using depthwise separable Gaussian filters ($\sigma=1.5$). The function outputs both:
1. The unreduced spatial map $(B, 1, H, W)$ for pixel-level defect heatmaps.
2. The scalar mean for backpropagation.

Crucially, the Gaussian window size is computed dynamically at runtime:
$$\text{win\_size} = \max(\min(11, H, W)_{\text{odd}}, 3)$$
with an explicit non-assert exception guard (`if win > min(h, w): raise ValueError(...)`) that survives Python optimization mode (`python -O`).

---

## 6. Analysis of Small Resolutions (15px Constraint)

A common assumption in autoencoder literature is that the SSIM window "collapses to its minimum" at small scales. In SolSight, testing reveals the opposite phenomenon:
At 15px and 16px, $\min(11, 15) = 11$. The window does not shrink to 3px; it remains the full **$11\times11$ window**, covering **$68.8\%$ to $73.3\%$ of the entire patch width**.

**Implications:**
- **Boundary Dominance:** The Gaussian window heavily overlaps the zero-padding at image boundaries.
- **Coarser Localization:** Spatial heatmap peaks at 16px are broad and diffuse rather than pin-sharp.
- **Architectural vs. Detection Reality:** While SolSight architecturally processes 15px and 16px inputs without error, localization sensitivity is naturally lower than at 64px or 128px. SolSight reports this limitation transparently as an inherent property of sliding-window structural metrics on micro-patches.

---

## 7. Resolution Generalization: Discrete Native Tiers & Zero-Shot Transfer

To ensure honest evaluation without interpolation artifacts:
- **Discrete Native Training Tiers:** The model is trained on procedurally rendered patches generated natively at $\{16, 64, 128\}$ pixels. Patches are never downsampled from 128px to 16px, preventing artificial smoothing.
- **Untrained Generalization Proof (32px):** To prove that SolSight learns a truly scale-invariant geometric representation rather than memorizing three fixed dimensions, **32px patches are excluded from training**.
- **Global Threshold Testing:** When evaluated on unseen 32px normal and defective patches, the single trained checkpoint is tested strictly against the global anomaly threshold $T$ calibrated from the trained tiers. High recall and low false-positive rates on 32px confirm genuine zero-shot scale transfer.

---

## 8. Summary of Defect Classes Detected

| Defect Class | Physical Mechanism | SolSight Detection Signature |
|--------------|--------------------|------------------------------|
| **Voids / Blowholes** | Outgassing during reflow leaving dark cavities. | Strong localized drop in SSIM; sharp DSSIM peak in Top-5%. |
| **Solder Bridging** | Excess solder shorting adjacent tracks/pads. | Outer boundary contour disruption; large asymmetric DSSIM mass. |
| **Cold Joints** | Insufficient heat or movement during cooling; granular matte surface. | Global loss of high-contrast specular highlight; elevated DSSIM across fillet. |
| **Excess / Underfill** | Out-of-tolerance solder paste volume. | Mismatch between learned dome radius and observed fillet boundary. |
