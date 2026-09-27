# SolSight — 90-Second Judging Demo Script

## Overview
This script is timed for a strict **90-second hackathon pitch**. It demonstrates the core unsupervised convolutional autoencoder (CAE) inspection engine, highlights its resolution-agnostic architectural proof point, and transparently analyzes failure modes.

---

## Pitch Timeline (0:00 – 1:30)

### 1. The Problem (0:00 – 0:15)
> *"Solder joint defects cause catastrophic field failures in mission-critical electronics, yet training supervised models is bottlenecked by the extreme rarity of labeled defect samples. SolSight detects micro-defects with zero labeled training examples, learning purely from defect-free golden reference fillets."*

---

### 2. Live Inspection & Structural Dissimilarity Heatmap (0:15 – 0:35)
*(Display Exhibit 2: Void Defect at 128px)*
> *"Here is an unseen defective solder joint with a blowhole void. Instead of standard MSE pixel difference—which produces blurry artifacts—SolSight computes a per-pixel Structural Dissimilarity (DSSIM) heatmap based on local luminance, contrast, and curvature.
> Notice how the model reconstructs a healthy golden joint, producing a sharp, localized thermal anomaly directly over the defect. Using Top-5% pooling, the anomaly score easily clears our threshold to trigger a reject."*

---

### 3. Resolution-Agnostic Proof & Untrained Scale Generalization (0:35 – 0:55)
*(Switch to Exhibit 5A/5B: 32px Untrained Generalization vs. Exhibit 6: 16px Micro-Joint)*
> *"Our primary technical contribution is true resolution agnosticism. Traditional autoencoders require fixed-size dense layers. SolSight is fully convolutional with dynamic runtime SSIM window sizing.
> This single set of weights was trained on discrete native tiers at 16, 64, and 128 pixels. Right now, we are evaluating it on this 32-pixel patch—a resolution the network **never saw during training**.
> Without a single retuned weight or bespoke threshold, it reconstructs healthy joints and flags voids cleanly, proving genuine zero-shot scale generalization."*

---

### 4. Honest Failure Analysis & Limitations (0:55 – 1:15)
*(Display Exhibit 7: Near-Miss Micro-Void)*
> *"In industrial inspection, honesty matters. Here is a near-miss: an ultra-subtle void occupying less than 1.5% of the patch area. Because sliding-window SSIM averages across local neighborhoods, micro-defects at the noise floor can escape detection under conservative thresholds.
> We address this using Top-5% percentile pooling rather than patch-wide averaging, and we publish our ROC trade-offs openly."*

---

### 5. Production Context & Next Steps (1:15 – 1:30)
> *"Today, SolSight proves the unsupervised neural core. In production, this engine integrates into automated CAD/Gerber crop pipelines and multi-angle lighting rigs to provide complete PCBA quality assurance. Full technical documentation and responsible AI governance notes are available in the repository."*

---

## Preparation Checklist for Presenters

- [ ] Open `outputs/demo_visuals/` in fullscreen image viewer or run `notebooks/demo.ipynb`.
- [ ] Pre-load all 7 exhibit figures (do not generate images live during judging window).
- [ ] Verify Exhibit 2 (128px void), Exhibit 5B (32px untrained generalization), and Exhibit 7 (near-miss) are queued in order.
- [ ] Have a screen-recording backup video ready on desktop in case of hardware or projector issues.
- [ ] Rehearse with a stopwatch twice before judging begins.
