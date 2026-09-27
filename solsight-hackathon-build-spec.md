# SolSight — Solder-Joint Defect Detection Component
## Hackathon Build Specification & Technical Reference

**Scope note:** This document covers *only* the solder-joint defect detection component of SolSight (the CAE-based inspection engine). It excludes the trace-validation stream, component/placement stream, PSP hardware capture, conveyor/gate integration, and all commercial/business-side work. Those are out of scope for this event.

**Judging alignment note:** This spec is written against the event's rubric — Problem+value, Functional execution, Quality of AI use, Testing+reliability, Experience+demo, Responsible AI+data. Sections 9 and 10 exist specifically to cover demo and responsible-AI, which a pure ML build plan wouldn't otherwise touch. Section 6 (failure modes) is what feeds honest Testing+reliability scoring — don't treat it as optional cleanup at the end.

**Loss/scoring note:** This spec uses **SSIM-based structural dissimilarity** as the anomaly signal throughout, not raw pixelwise (MSE) reconstruction error. See Section 2.5 for why and how.

---

## 1. What You Are Building

A **resolution-agnostic convolutional autoencoder (CAE)** that:

1. Is trained **unsupervised**, only on defect-free ("golden reference") solder joint image patches.
2. Detects anomalies via **structural dissimilarity (SSIM-based) between input and reconstruction** — a defective joint reconstructs with lower structural similarity because the model has never seen that pattern.
3. Works across a **range of input patch sizes** (from ~15px up to 128px) without hardcoding tensor dimensions — including the SSIM window size — anywhere in the architecture or pipeline.
4. Is trained and evaluated on a **mixed synthetic + real dataset**, with real data reserved primarily for validation.
5. Produces a **usable, demoable inference pipeline**: image patch in → structural-dissimilarity heatmap + anomaly score out.

This is a scoped, working proof of the core inspection mechanism — not the full SolSight system.

---

## 2. Architecture Requirements

### 2.1 Fully convolutional, resolution-agnostic design

- **No `Flatten()` + `Dense()` bottleneck.** Any dense layer forces a fixed input size. The entire network — encoder, bottleneck, decoder — must be composed of convolutional and pooling/upsampling operations only.
- **No hardcoded `H`, `W` values anywhere** — this now also applies to the SSIM computation itself (see 2.5), not just the CAE layers. Every shape and window size must be derived at runtime from `input.shape`.
- **Bottleneck via strided convolutions, not global pooling to a fixed vector.** Downsample by a fixed *factor* (e.g. 4x), not to a fixed *size*. This preserves spatial correspondence needed for the structural-dissimilarity heatmap.
- **Padding strategy:** `padding='same'` throughout so spatial dimensions scale predictably with input.
- **Handle the 15px case explicitly as a design constraint, not an edge case.** Validate with a shape-check script (2.4) *before* writing any training code — and note that at 15px, the SSIM window size (2.5) becomes the binding constraint, not just the CAE's own downsampling.

### 2.2 Reference architecture (parameters, not pixel counts)

```
Encoder:
  ConvBlock(in=3,  out=16, k=3, s=1, pad=same) -> ReLU
  ConvBlock(in=16, out=16, k=3, s=2, pad=same) -> ReLU   # downsample x2
  ConvBlock(in=16, out=32, k=3, s=1, pad=same) -> ReLU
  ConvBlock(in=32, out=32, k=3, s=2, pad=same) -> ReLU   # downsample x2 (x4 total)
  ConvBlock(in=32, out=64, k=3, s=1, pad=same) -> ReLU

Bottleneck:
  ConvBlock(in=64, out=64, k=3, s=1, pad=same) -> ReLU

Decoder (mirror, Upsample+Conv preferred over ConvTranspose to avoid checkerboarding):
  Upsample(scale=2) -> Conv(in=64, out=32, k=3, pad=same) -> ReLU
  Upsample(scale=2) -> Conv(in=32, out=16, k=3, pad=same) -> ReLU
  Conv(in=16, out=3, k=3, pad=same) -> Sigmoid   # reconstruct to [0,1] pixel range — required for SSIM, which assumes a bounded intensity range
```

- Total downsampling factor: **4x** (two stride-2 layers). 8x would collapse a 16px input to a 1×1 bottleneck, losing all spatial information; 4x keeps even small patches at a spatially meaningful 4×4 or larger.
- Channel counts (16→32→64) are a reasonable hackathon-scale starting point. Don't spend time tuning these unless the model is otherwise working.
- **No skip connections, deliberately.** Skip connections let fine detail — including a defect's detail — bypass the compressed bottleneck and get copied straight through, which would let the decoder reconstruct anomalies faithfully and hide them from the dissimilarity map. Forcing every detail through the compressed representation is what makes defects show up as structural dissimilarity in the first place. State this explicitly in your write-up as an intentional trade-off.
- **Sigmoid output is now a hard requirement, not just a nicety.** SSIM's luminance/contrast/structure terms assume inputs are on a consistent, bounded scale — keep both input images and reconstructions normalized to [0,1].

### 2.3 Odd-size handling (important correctness caveat)

With integer stride-2 downsampling, non-power-of-two inputs round up at each stage (e.g. a 15px input becomes `ceil(15/2)=8`, then `ceil(8/2)=4` at the bottleneck), and upsampling back out (4→8→16) yields a **16px reconstruction from a 15px input** — a one-pixel mismatch that will break both the SSIM computation and any pixelwise comparison if unhandled.

Two acceptable fixes, pick one and apply it consistently:
- **Center-crop the decoder output** to match the original input's `H, W` before computing SSIM (simplest, recommended for the hackathon timeframe).
- **Pad the input up** to the nearest multiple of 4 before the encoder, then crop the decoder output back down to the original size afterward.

### 2.4 Shape-check script (do this first, before any training code)

Write a short script that instantiates the model and runs a forward pass at multiple resolutions — include **both powers of two and odd sizes** (e.g. 15, 16, 17, 32, 64, 96, 128px) — asserting the (cropped/padded) output shape matches the input shape at every one. Extend this same script to also compute the SSIM map at each resolution and assert it produces a valid, non-degenerate map (see 2.5 for the window-size logic this is checking). Rerun this any time you touch the architecture or the SSIM window-sizing logic.

### 2.5 Loss, scoring, and thresholding (SSIM-based)

**Why SSIM instead of raw pixelwise (MSE) error:** MSE-based reconstruction error treats every pixel independently and tends to reward blurry, oversmoothed reconstructions — which weakens sensitivity to the kind of localized structural anomalies (voids, bridging, shape deformation) that actually define a solder defect. SSIM instead compares local **luminance, contrast, and structure** within a sliding window, which is a much closer match to "does this region look structurally like a normal solder joint" than raw pixel-difference does. This is the primary loss and scoring mechanism for this build — not an optional add-on.

- **SSIM window size must scale with input size, not be hardcoded.** The conventional default (an 11×11 Gaussian window) is too large relative to a 15px patch. Derive the window size at runtime: `win_size = min(11, H, W)`, rounded down to the nearest odd number, with a floor of 3. This keeps the same SSIM computation valid from 15px up to 128px without branching logic — compute and log this derived value so you can point to it directly as evidence the pipeline is genuinely resolution-agnostic, not just the CAE layers.
- **Training loss:** `loss = 1 - SSIM(input, reconstruction)`, using a differentiable SSIM implementation (e.g. `pytorch_msssim` or an equivalent), averaged over the batch for backprop. Use the dynamically computed window size from above on every call, not a fixed constant.
- **Inference-time dissimilarity map:** compute the *per-pixel/per-window* SSIM map (most SSIM implementations can return the unreduced spatial map rather than only a scalar), then invert it to a dissimilarity map: `dssim_map = (1 - ssim_map) / 2`, giving a `[H, W]` map in `[0, 1]` where higher values indicate more structural difference from what the model considers normal. This is what you visualize as the anomaly heatmap.
- **Anomaly score (single number per patch): use top-k percentile of the dissimilarity map, not the mean.** Formula: `score = mean(top 5% highest-DSSIM regions in dssim_map)`. The rationale carries over directly from a pixelwise approach: a small defect (e.g. a pinhole void covering ~2% of the patch) should not get diluted by a global average across mostly-normal regions. Explain this explicitly in your write-up — it's a deliberate, defect-domain-motivated choice, and is a direct "Quality of AI use" talking point.
- **Threshold selection:** compute anomaly scores on your held-out defect-free validation set, then set threshold as `T = mean(val_scores) + k * std(val_scores)`, with `k` in the 2–3 range as a starting point. Report the resulting false-positive rate honestly.
- **Checkerboard/blocky artifacts:** if the Upsample+Conv decoder produces visible blocky artifacts, a small Gaussian blur applied post-upsample can reduce spurious localized SSIM drops that are decoder artifacts rather than genuine anomalies — check for this before assuming a blocky region is a real defect signal.
- **Multi-scale SSIM (MS-SSIM) as an optional refinement:** if time allows and single-scale SSIM seems to miss larger-scale shape defects (e.g. overall dome deformation) while catching fine texture ones, MS-SSIM (which aggregates SSIM across several downsampled scales) can improve sensitivity across defect sizes. Only pursue this after the single-scale SSIM pipeline is fully working — it adds complexity for a marginal gain within hackathon time constraints.

### 2.6 Multi-scale training approach

- Train with **random resizing/cropping** of patches within the 15–128px range each epoch, if time allows — the strongest proof of resolution-agnosticism, and now also exercises the dynamic SSIM window-sizing logic across scales.
- Simpler, still valid for a hackathon: run a few short training passes at fixed sample resolutions (e.g. 16px, 32px, 64px, 128px) using the *identical* model and loss code, no branching. Show the shape-check script and training curves succeeding at all of them, and log the derived SSIM window size at each resolution as evidence.
- **The demo must show the same model and loss definition running unmodified across at least two very different resolutions** (e.g. 16px and 128px) — this is your core proof point.

### 2.7 Suggested hyperparameters and optional refinements

- Optimizer: Adam, `lr=1e-3`
- Batch size: as large as memory allows at your smallest resolution; reduce for 128px patches if memory-constrained
- Epochs: 20–50 for a hackathon-scale synthetic dataset — watch the SSIM-based training loss curve rather than fixing an epoch count in advance
- Early stopping: monitor validation loss (`1 - SSIM`) on held-out defect-free patches; stop if it plateaus for ~5 epochs
- **Optional, only if the core pipeline works with time to spare:** batch/group normalization (stabilizes training across scales), light dropout or noise injection (discourages trivial memorization of synthetic generator artifacts), a small residual block in the bottleneck (extra capacity), MS-SSIM (2.5). These are refinements, not requirements.

---

## 3. Data Strategy

### 3.1 Synthetic data (primary volume driver)

- Generate golden-reference (defect-free) solder joint patches procedurally: parametric renders of a solder fillet (cone/dome geometry). Suggested minimum volume for a hackathon-scale demo: **500–2,000 patches**, generated via randomized parameters rather than hand-drawn one by one.
- **Randomize more than just geometry.** Vary fillet radius, tilt, highlight position, background color/texture, lighting intensity, and small camera-angle rotations. Randomizing only shape risks the CAE latching onto trivial, non-generalizable cues instead of learning genuine joint structure — and SSIM is specifically sensitive to structural/contrast patterns, so structural diversity in training data matters more here than it would for a plain pixelwise loss.
- **Apply data augmentation aggressively to the synthetic normals** (flip, rotate, scale jitter, brightness/contrast jitter) so the model doesn't overfit to the generator's specific quirks.
- Generate synthetic **defects for validation only** (never for training — this is unsupervised):
  - *Voids:* dark elliptical/irregular blobs subtracted from the fillet region, randomized size/position. Keep some intentionally small — this is exactly the case the top-k DSSIM scoring in 2.5 is designed to catch.
  - *Bridging:* a thin bright band connecting two adjacent joint renders.
  - *Cold joints:* reduce specular highlight intensity and/or add rough noise texture to mimic a matte surface — this is a strong structural-texture change, which SSIM is well-suited to catch.
  - *Insufficient/excess solder:* scale the dome geometry outside the normal parameter range used for golden references.
- Use simple procedural graphics rather than a GAN or diffusion model — out of scope for hackathon time and no accuracy benefit over parametric synthesis for this task.

### 3.2 Real data (validation-focused)

- Pull a small set of real solder joint images from an existing public dataset — e.g. **SolDef_AI on Kaggle (~1,150 labeled real solder joint images)** — enough to check that a model trained purely on synthetic golden references also scores real defect-free joints as normal, and (where labels exist) flags real defects as anomalous.
- Treat real data as your **credibility check**, not primary training volume.
- Attribute the dataset explicitly (name + source) in your write-up.

### 3.3 Train / validate / test split logic

- **Train:** synthetic golden-reference (defect-free) patches only.
- **Validation:** held-out synthetic defect-free + synthetic defects + real defect-free — used to tune the anomaly threshold and measure false-positive rate.
- **Test/demo set:** real defect-free + real defective (if available) + synthetic defects across your resolution range — what you show judges.

---

## 4. Evaluation Metrics (for Testing + Reliability)

Report these honestly, even if imperfect:

- **False-positive rate (FPR)** on held-out defect-free patches (synthetic + real), based on the SSIM-derived anomaly score at your chosen threshold.
- **True-positive rate (recall)** on synthetic defect validation patches, **broken down per defect class** (voids, bridging, cold joints, over/underfill). Note where you'd expect SSIM-based scoring to differ from a plain pixelwise approach — e.g. likely stronger on structural/textural defects (cold joints, bridging) than on very subtle localized voids at small patch sizes.
- **AUROC** if time allows, sweeping the SSIM-derived threshold and plotting TPR vs. FPR.
- **Qualitative failure examples:** show at least one case where the model misses a subtle defect or false-flags a valid joint, with a one-sentence explanation of why.
- **If time allows, automate this:** a small script that sweeps the threshold, computes FPR/TPR at each point, and outputs a summary table or ROC plot.

---

## 5. Qualitative Demo Content

Prepare, in advance, a small fixed set of side-by-side visuals: input patch → reconstruction → SSIM-based dissimilarity heatmap, for a handful of representative cases (a clean joint, a void, a bridge, a cold joint). Annotate what the bright regions in each heatmap correspond to. Include at least one case where the anomaly is barely visible in the heatmap (a near-miss) — this feeds directly into Section 6 and the demo script (Section 9).

---

## 6. Failure Modes and Mitigations

- **Subtle anomalies:** tiny voids or mild texture changes may barely shift the top-5% DSSIM average, leading to low recall on those specific cases. Mitigations: increase defect severity in synthetic generation so subtle cases become more evident for the demo, or add MS-SSIM (2.5) if time allows.
- **High false-positive rate:** if synthetic normals are too uniform (fixed lighting, fixed background), real-world variation (glare, dust, shadow) can register as structural difference. Mitigation: broaden augmentation on synthetic normals (Section 3.1), and/or include some real normal patches in validation to calibrate.
- **Domain gap:** the model only knows what it was trained on. If real joints use alloys, colors, or camera angles the synthetic generator doesn't model, many real normals may score as anomalous. State this limitation directly (Section 10).
- **Extreme small sizes (15px):** the SSIM window shrinks to its floor (3×3) at this scale, which reduces the structural context available for comparison and can make the dissimilarity signal noisier than at larger patch sizes. Note explicitly that "architecturally supports 15px" and "reliably detects defects at 15px" are different claims — the smaller the window, the closer SSIM behaves to a local pixelwise comparison, losing some of its structural advantage.
- **Reconstruction artifacts:** blocky/checkerboard patterns from the upsampling decoder can register as false structural dissimilarity (see 2.5 mitigation).

Treat the model's output as a **cue**, not a final decision — this connects directly to the human-oversight framing in Section 10.

---

## 7. What NOT to Build (don't waste hackathon time on this)

- **Do not build the PSP (phase-shift profilometry) stream.** Requires physical projector/camera hardware you don't have here.
- **Do not build the CAD/Gerber file parser or ROI extraction pipeline.** Real SolSight infrastructure, irrelevant to proving the CAE works — use pre-cropped patches for this event.
- **Do not build the trace-validation (XOR/connectivity) stream.** Separate stream, separate problem.
- **Do not build the physical reject-gate/conveyor integration.** No hardware access, no relevance to the ML task being judged.
- **Do not try to build a GAN, diffusion model, or other generative model for synthetic data.** Parametric/procedural synthesis is sufficient and dramatically faster to implement.
- **Do not chase state-of-the-art anomaly detection literature** (e.g. PatchCore, PaDiM, memory-bank methods) unless the core CAE is finished early with time to spare.
- **Do not spend time building a polished UI/dashboard before the core pipeline works.**
- **Do not hand-pick demo images to only show favorable results.**
- **Do not over-invest in MS-SSIM, hyperparameter tuning, or the optional refinements in Section 2.7.** Get one clean single-scale-SSIM pipeline working end to end first; spend remaining time on evaluation honesty (Section 4) and the demo script (Section 9) instead.

---

## 8. Deliverables Checklist

- [ ] Fully convolutional CAE implementation with zero hardcoded spatial dimensions, no skip connections, sigmoid output
- [ ] Odd-size handling implemented (crop or pad strategy, Section 2.3)
- [ ] SSIM window size derived at runtime from input dimensions (Section 2.5), not hardcoded
- [ ] Shape-check script passing at ≥3 resolutions including at least one odd size (e.g. 15, 64, 128px), also validating SSIM map output at each
- [ ] Procedural synthetic data generator for golden-reference patches + defect variants, with varied lighting/background/rotation augmentation
- [ ] Small real-data validation set integrated (e.g. SolDef_AI), attributed in write-up
- [ ] Training run(s) completed using `1 - SSIM` loss; DSSIM-based anomaly scoring implemented (top-k percentile method)
- [ ] Threshold tuned using validation set; FPR and per-class recall reported honestly
- [ ] At least one qualitative failure case identified and explained (Section 5/6)
- [ ] Demo script/notebook: same architecture and SSIM logic at ≥2 resolutions, dissimilarity heatmaps on real + synthetic defects
- [ ] Short technical write-up covering the resolution-agnostic design rationale, the SSIM-over-MSE decision, the no-skip-connections decision, and how this maps to SolSight's actual multi-defect-class, multi-resolution production requirement
- [ ] Responsible AI + data note written (Section 10)
- [ ] 90-second demo script rehearsed (Section 9)

---

## 9. Demo Script (90 Seconds)

1. **(0:00–0:15) Problem, one sentence.** "Solder joint defects cause field failures in electronics; this detects them without ever training on a single labeled defective example."
2. **(0:15–0:35) Show the model working live at one resolution.** Run inference on a real, unseen solder joint image. Show reconstruction and the SSIM-based dissimilarity heatmap side by side. Point at the heatmap.
3. **(0:35–0:55) Show the resolution-agnostic proof point.** Re-run the *same, unmodified model and SSIM code* on a differently-sized patch (e.g. 16px vs 128px), noting the window size adapted automatically. This is your most distinctive technical claim — don't rush it.
4. **(0:55–1:15) Show one honest failure case** (from Section 6), with a one-sentence explanation of why it happens.
5. **(1:15–1:30) Close on scope and path forward.** "This is the core detection engine; in production it plugs into a CAD-driven capture pipeline not built today — full write-up available."

**Preparation notes:**
- Pre-load all demo images/patches before presenting — do not run live data generation during the demo window.
- Have a backup screen-recording of the same sequence in case live inference has an environment hiccup during judging.
- Rehearse with a timer at least twice before presenting.

---

## 10. Responsible AI + Data Note

- **Data provenance and consent:** All training data is either procedurally generated synthetic imagery (no privacy or consent concerns) or drawn from an existing public dataset (e.g. SolDef_AI) used under its published license and credited by name. No private, proprietary, or customer PCB data is used.
- **Bias and limitations:** The model's notion of "normal solder" is entirely defined by the synthetic generator's parameters. If real joints vary in ways not modeled, false positives are expected — stated directly rather than implied away.
- **One-class training caveat:** Because training is purely unsupervised on defect-free data, genuinely novel anomaly types not resembling any tested defect class may not reliably trigger detection, and normal-but-unusual joints may occasionally false-flag. This is inherent to one-class/anomaly-detection training, not a fixable bug.
- **Safety / human oversight:** This system is framed as a **decision-support and inspection-aid tool**, not an autonomous accept/reject system. In production, a human or downstream verification step reviews flagged anomalies before any physical action is taken.
- **Copyright:** Synthetic data is originated by procedural code, not scraped or copied imagery. Any real dataset used is public and explicitly attributed.
- **Failure transparency:** The system's current false-positive/false-negative behavior (Sections 4 and 6) is disclosed rather than concealed.

