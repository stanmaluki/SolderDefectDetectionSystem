# SolSight Presentation Generation Prompt

## Master Prompt for AI-Generated Presentations

Use this comprehensive prompt with ChatGPT, Claude, or other AI tools to generate, refine, or expand your SolSight presentation slides.

---

## 🎯 PRIMARY PROMPT

```
You are an expert technical presentation designer creating a PITCH DECK style 
presentation for a Solder Defect Detection System (SolSight) project using LaTeX 
Powerdot package.

PROJECT BACKGROUND:
- SolSight: Unsupervised convolutional autoencoder for automatic optical inspection 
  (AOI) of solder-joint defects on PCBAs
- Key innovation: Unsupervised learning on defect-free joints → anomaly detection 
  as reconstruction failures
- Resolution-agnostic: Handles 16px to 160px natively (fully convolutional, no dense layers)
- Scoring: Localized DSSIM (Top-5% spatial structural dissimilarity)
- Training data: 5,700 procedurally synthesized patches (100% synthetic)
- Real-world validated on SolDef-AI benchmark (250 physical PCBAs)

PRESENTATION REQUIREMENTS:

1. STYLE & AESTHETICS:
   - Pitch deck format (concise, visual, data-driven)
   - Modern color scheme: Primary deep blue (#1F4E79), Accent red (#C00000)
   - Minimal text, maximum impact (bullet points, not paragraphs)
   - High visual hierarchy with emphasis on metrics and data
   - Attractive typography with proper spacing
   - Consistent branding throughout

2. SLIDE STRUCTURE (Total: 17 slides, ~9-10 minute presentation):
   a) Title Slide (Branding)
   b) Problem Statement (Why this matters)
   c) Solution Overview (What we do differently)
   d) Technical Architecture (CAE design + DSSIM)
   e) Multi-Scale Training (Resolution tiers: 16px, 32px, 64px, 128px)
   f) Synthetic Benchmark Results (AUROC scores, recall metrics)
   g) Real-World Validation (SolDef-AI results, 0.879 AUROC)
   h) Failure Boundaries (Known limitations, safe operating ranges)
   i) Deployment Workflow (3-step integration: calibration → inference → monitoring)
   j) Competitive Advantages (vs traditional AOI, supervised DL)
   k) Technical Stack (Python, PyTorch, OpenCV, SSIM)
   l) Quick Start Guide (Installation, verification, inference)
   m) Key Metrics Summary (Training time, inference speed, model size)
   n) Use Cases (Manufacturing, QA, R&D)
   o) Future Roadmap (Multi-modal fusion, edge deployment, explainability)
   p) Contact & Resources (GitHub, documentation, license)
   q) Closing Slide (Call to action)

3. CONTENT GUIDELINES:

   a) PROBLEM STATEMENT SLIDE:
      - Manual inspection is time-consuming and error-prone
      - Defects degrade PCBA reliability
      - Traditional AOI needs extensive labeled datasets
      - Defect variations across hardware are hard to generalize
      - Hook: "What if we could detect anomalies without labeling defects?"

   b) SOLUTION OVERVIEW SLIDE:
      - One-line hook: "Train on Normal Joints → Detect Anomalies as Reconstruction Failures"
      - Comparison table: Unsupervised vs Traditional vs Supervised approaches
      - Key advantages: No labeling, resolution-flexible, fast inference, defect localization

   c) ARCHITECTURE SLIDE:
      - TikZ diagram: Input → Encoder (3 convs, s=2,2,1) → Bottleneck → 
        Decoder (3 deconvs, s=1,2,2) → Reconstruction → DSSIM Map
      - Highlight: No dense layers (fully convolutional)
      - Adaptive SSIM window scaling for different patch sizes

   d) MULTI-SCALE TRAINING SLIDE:
      - Table with 4 tiers: 16px (trained), 32px (zero-shot), 64px (trained), 128px (trained)
      - Key insight: Single model handles 16-160px natively
      - Advantage: No rescaling, adaptive per-resolution thresholding

   e) BENCHMARK RESULTS SLIDE:
      - Synthetic tier results table:
        | Tier | Type | AUROC | Cold Joint Recall | Bridge Recall | Normal FPR |
        | 16px | Trained | 0.940 | 96% | 84% | 10% |
        | 32px (ZERO-SHOT) | Generalize | 0.862 | 100% | 44% | 15% |
        | 64px | Trained | 0.963 | 100% | 90% | 0% |
        | 128px | Trained | 0.850 | 100% | 88% | 0% |
      - Highlight: AUROC 0.963 at 64px, strong generalization to untrained 32px

   f) REAL-WORLD VALIDATION SLIDE:
      - Zero-shot transfer from synthetic training to physical PCBAs
      - Key metrics: 0.879 AUROC, 0.0% FPR, 100% defect recall (global threshold)
      - Data source: 250 patches from SolDef-AI industrial benchmark
      - Context: Misaligned 72%, Excessive 46%, Insufficient 42%, Spike 34% recall

   g) FAILURE BOUNDARIES SLIDE:
      - Table with dimensions, safe ranges, breaking points, root causes:
        | Spatial Resolution | 16-160px | ≤14px | SSIM window > patch |
        | Defect Footprint | ≥1.5% area | <0.5% | Global threshold high |
        | Sensor Noise | σ≤0.01 | σ≥0.02 | Contrast penalty in SSIM |
      - Reassurance: Per-tier adaptive thresholding solves boundary cases

   h) DEPLOYMENT WORKFLOW SLIDE:
      - 3-phase visual flow:
        1. Calibration: Collect 100-200 site samples, fine-tune threshold
        2. Inference: Stream patches, generate anomaly scores in <50ms, visualize heatmaps
        3. Monitoring: Track FP/FN rates, batch alerts
      - Key message: Production-ready on day one

   i) COMPARATIVE ADVANTAGES SLIDE:
      - Comparison matrix (SolSight vs Traditional AOI vs Supervised DL):
        | Feature | SolSight | Traditional AOI | Supervised DL |
        | Labeling | ✗ No | ✓ Yes | ✓ Yes |
        | Resolution Flexible | ✓ Yes | ~ Limited | ✗ No |
        | Domain Transfer | ✓ 0.879 AUROC | N/A | ✗ Failed |
        | Real-Time Speed | ✓ <50ms | ~ Varies | ✓ Yes |
        | Defect Localization | ✓ Heatmap | ~ Limited | ✓ Yes |
      - Bottom line: No labeled data. Deploy on day one.

   j) TECHNICAL STACK SLIDE:
      - Language: Python 3.10+
      - DL Framework: PyTorch 2.1+
      - Vision: OpenCV, scikit-image (SSIM, morphology)
      - Metrics: ROC-AUC, AUROC, precision/recall
      - Deployment: ONNX export for edge (optional)
      - Closing: "Fully open-source • MIT License • Production-validated"

   k) QUICK START SLIDE:
      - Step 1: Clone + Install (git clone + pip install)
      - Step 2: Verify (python scripts/shape_check.py)
      - Step 3: Evaluate (python scripts/evaluate.py)
      - Code blocks formatted for easy copy-paste

   l) KEY METRICS SUMMARY SLIDE:
      - Large table: Training Data (5,700 synthetic) | Benchmark AUROC (0.963 @ 64px) | 
        Real-World AUROC (0.879)
      - Sub-metrics: Training time (30 epochs), Inference (<50ms/patch), 
        Model size (2.4 MB), Memory (280 MB GPU or CPU-compatible)

   m) USE CASES SLIDE:
      - 3 primary sectors with bullets:
        1. High-Volume Manufacturing: Real-time PCB screening, zero downtime
        2. Quality Assurance: Batch anomaly tracking, operator alerts
        3. R&D: Benchmark new processes, rapid failure analysis (no labeling)

   n) FUTURE ROADMAP SLIDE:
      - Multi-Modal Fusion: Visible + thermal imaging
      - Online Adaptation: Continuous threshold refinement with human feedback
      - Edge Deployment: ONNX + TVM quantization for ARM/FPGA
      - Explainability: Attention maps, gradient-based importance ranking

   o) CONTACT & RESOURCES SLIDE:
      - GitHub link, documentation pointer, MIT license badge
      - Minimal but complete

   p) CLOSING SLIDE:
      - Large text: "Questions?"
      - Tagline: "Automatic Optical Inspection Made Simple & Unsupervised"
      - Smaller: "AI-Powered Quality Control"

4. VISUAL DESIGN PRINCIPLES:
   - Use color strategically: Primary (deep blue) for structure, Accent (red) for emphasis
   - Whitespace generously (avoid clutter)
   - Tables for data, bullet points for concepts
   - TikZ diagrams for technical flows (architecture, deployment)
   - Icons/symbols for visual anchors (checkmarks, numbers, arrows)
   - Consistent font sizing: Headers (Large/Huge), Body (normalsize/small)
   - Maintain 16:9 aspect ratio throughout

5. TONE & VOICE:
   - Professional but approachable (pitch deck, not academic paper)
   - Data-driven (numbers, metrics, benchmarks everywhere)
   - Forward-looking (future potential emphasized)
   - Confident (proven results highlighted)
   - Jargon minimized, impact maximized

6. TECHNICAL ACCURACY:
   - All metrics from actual benchmark results
   - Architecture description accurate to CAE design
   - Real-world results from SolDef-AI benchmark documented
   - Failure boundaries reflect empirical analysis
   - Deployment workflow realistic and actionable

OUTPUT FORMAT:
- LaTeX Powerdot document
- Clean, commented code
- Ready to compile to PDF
- All 17 slides complete
- Attractive, pitch-deck quality

DELIVERABLES:
1. presentation.tex (main LaTeX file, ready to compile)
2. PRESENTATION_GUIDE.md (compilation, customization, tips)
3. PRESENTATION_PROMPT.md (this prompt for future refinements)
```

---

## 🔄 REFINEMENT PROMPTS

### For Slide Improvements:

```
Enhance slide X (SLIDE_TITLE) with:
1. More visual impact using TikZ diagrams
2. Better data visualization for metrics
3. Clearer hierarchy of information
4. Stronger color contrast and emphasis

Keep the same factual content but improve design and readability.
```

### For Adding New Content:

```
Add a new slide after slide X with the following:
- Title: [TITLE]
- Key points: [BULLET_POINTS]
- Visual elements: [DESCRIPTION]
- Target audience: [AUDIENCE]

Use the same style as existing slides (pitch deck format, color scheme, spacing).
```

### For Converting to Different Formats:

```
Convert the LaTeX Powerdot presentation to:
- Format: [PowerPoint/Beamer/Reveal.js/etc.]
- Preserve: Colors, layout, content, visual hierarchy
- Output: [PowerPoint file / HTML / PDF]
- Maintain: 16:9 aspect ratio, speaker notes if any
```

---

## 📊 DATA REFERENCE TABLE

Use this for consistency checks:

| Metric | Value | Context |
|--------|-------|---------|
| Training Data | 5,700 patches | 100% procedurally synthetic |
| 16px AUROC | 0.940 | Trained tier |
| 32px AUROC | 0.862 | Zero-shot (untrained) |
| 64px AUROC | **0.963** | Best trained tier |
| 128px AUROC | 0.850 | Trained tier |
| Real-World AUROC | 0.879 | SolDef-AI (250 physical PCBAs) |
| Real-World FPR | 0.0% | At calibrated threshold |
| Real-World Recall | 100.0% | All defect types (synthetic threshold) |
| Inference Speed | <50ms | Per patch (CPU-capable) |
| Model Size | 2.4 MB | Deployable on edge |
| GPU Memory | 280 MB | Or CPU-compatible |
| Training Time | 30 epochs | At batch size 32 |
| Cold Joint Recall | 96-100% | All tiers |
| Bridge Detection | 84-90% | (varies by tier) |
| Min Safe Resolution | 16px | Lower bound empirical |
| Max Safe Resolution | 160px | Upper bound empirical |
| Min Defect Footprint | ≥1.5% | Of patch area |
| Sensor Noise Limit | σ ≤ 0.01 | SNR ≥ 40 dB |
| Calibration Samples | 100-200 | Site-specific good samples |

---

## 🎨 COLOR PALETTE

```latex
% Primary
Primary Blue: RGB(31, 78, 121) = #1F4E79

% Accent
Accent Red: RGB(192, 0, 0) = #C00000

% Supporting
Light Gray: RGB(242, 242, 242) = #F2F2F2
Dark Gray: RGB(51, 51, 51) = #333333

% Alternatives (if needed)
Highlight Green: RGB(0, 176, 80) = #00B050
Warning Orange: RGB(255, 192, 0) = #FFC000
```

---

## 🚀 USAGE INSTRUCTIONS

1. **Initial Generation**: Copy PRIMARY PROMPT into your AI tool (ChatGPT, Claude, etc.)
2. **Refinement**: Use REFINEMENT PROMPTS for specific slide improvements
3. **Customization**: Modify color scheme, text, or data using DATA REFERENCE TABLE
4. **Version Control**: Keep PRESENTATION_PROMPT.md in repo for future iterations
5. **Compilation**: Use `pdflatex presentation.tex` to generate PDF
6. **Feedback Loop**: Regenerate slides if audience changes or new data becomes available

---

## 💡 TIPS FOR AI TOOLS

- **Be Specific**: Mention slide numbers, exact metrics, and design preferences
- **Reference Examples**: Show the AI tool existing slide LaTeX code if available
- **Iterate Quickly**: Ask for small tweaks before major rewrites
- **Check Math**: Always verify AUROC, recall, and other numerical claims
- **Consistency**: Remind AI to maintain style guidelines across all slides

---

## ✅ QUALITY CHECKLIST

Before finalizing, ensure:

- [ ] All metrics match README.md and source docs
- [ ] Color scheme consistently applied
- [ ] LaTeX compiles without errors
- [ ] Slide order matches presentation flow
- [ ] Visual hierarchy is clear (headers > body > footnotes)
- [ ] Tables are readable (not too many columns/rows)
- [ ] Code blocks are syntactically highlighted
- [ ] No typos or grammatical errors
- [ ] Branding (SolSight) appears on every slide
- [ ] Aspect ratio is 16:9 throughout

---

## 📞 SUPPORT

- **LaTeX Issues**: Check PRESENTATION_GUIDE.md troubleshooting section
- **Content Updates**: Use DATA REFERENCE TABLE to keep metrics current
- **Design Feedback**: Use REFINEMENT PROMPTS for targeted improvements
- **New Features**: Extend this prompt with additional sections as needed

---

**Last Updated**: September 27, 2026  
**Version**: 1.0  
**License**: MIT (same as project)
