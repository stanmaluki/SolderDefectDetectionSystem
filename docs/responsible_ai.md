# SolSight — Responsible AI & Data Governance Statement

## 1. Data Provenance and Licensing

SolSight maintains complete transparency and rigorous data governance across both synthetic training and real-world evaluation:
- **Procedural Synthetic Training Distribution:** The convolutional autoencoder is trained exclusively on procedurally synthesized defect-free solder joints rendered from first-principles 3D optics (Blinn-Phong specular illumination, Lambertian diffuse models, and spherical geometry), eliminating copyright infringement, proprietary NDA exposure, or unauthorized scraping.
- **Privacy & Proprietary Protection:** No customer Gerber files, proprietary CAD designs, or commercial PCBA schematics are contained in the training corpus.
- **Real-World PCBA Benchmark Validation (SolDef_AI):** Physical evaluation is conducted on genuine solder joint imagery curated from the **SolDef_AI** industrial benchmark (`mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection`). Using the author LabelMe polygon annotations, 250 individual solder joint patches (50 normal reference joints, 200 physical defects across excessive solder, insufficient solder, solder spikes, and misalignments) were extracted. Complete image-level attribution and bounding box coordinates are preserved in [`data/real/provenance.json`](../data/real/provenance.json).
- **Zero-Shot Domain Transfer:** The model achieved **0.879 AUROC** on physical PCBA imagery without any real-world retraining or fine-tuning, demonstrating genuine generalization from synthetic physical optics to real factory inspection cameras.

---

## 2. Model Bias and Operating Limitations

Because SolSight is trained in an unsupervised manner exclusively on defect-free reference samples, its internal definition of a "valid solder joint" is bounded by its training distribution:
1. **Synthetic-to-Real Domain Gap:** Procedural data accurately models specular curvature, pad geometry, flux residue, and structural anomalies (voids, bridges, cold joints, insufficient solder). However, live industrial lines introduce unique hardware artifacts (ring-light diffraction, dust, PCB warpage, conveyor vibration). Production readiness is strictly gated on collecting physical calibration images on the target SMT line.
2. **Solder Alloy Variation:** Industrial soldering processes utilize distinct alloys (SAC305 lead-free vs. Sn63Pb37 tin-lead vs. low-temperature bismuth alloys). Differences in alloy reflectivity, wetting angle, or grain structure require site-specific baseline calibration.
3. **Surface Contaminants & Flux Residue:** While benign flux halos are modeled during training to reduce false rejects, thick burnt rosin or heavy wash residue can cause elevated dissimilarity scores.
4. **Resolution Boundaries:** As empirically documented in `docs/empirical_breaking_points.md`, patches below $16\times16$ px suffer mathematical representation collapse due to SSIM window padding and stride quantization. SMT optical systems must enforce an inspection patch floor of $\ge 16\text{px}$.

These factors are documented openly as operating parameters that require site calibration rather than concealed as edge cases.

---

## 3. The One-Class Anomaly Detection Paradigm

SolSight operates under the one-class classification paradigm. This yields key advantages and intrinsic trade-offs:
- **Advantage:** SolSight does not rely on extensive libraries of labeled defects. It can detect unexpected defects (bridging, blowholes, misalignments) that human annotators did not anticipate during training.
- **Trade-Off:** An anomaly detection model cannot provide a multi-class semantic diagnosis with 100% confidence out of the box (e.g., distinguishing between an acceptable meniscus variation and a minor underfill). Genuinely novel joint geometries that conform mathematically to smooth spherical curvature might not trigger threshold alerts.

---

## 4. Human-in-the-Loop Decision Support Framework

In alignment with responsible industrial AI principles, SolSight is explicitly architected as a **Decision-Support and Inspection-Aid System**, not an autonomous unmonitored gatekeeper:
- **Triage & Curation:** SolSight flags suspicious solder joints and generates heatmaps with highlighted regions of interest to assist human Quality Assurance (QA) inspectors.
- **Review Prioritization:** QA operators review heatmaps for borderline joints ($T \approx \text{Score}$), drastically reducing inspector eye fatigue while keeping critical line decisions under human accountability.
- **Auditability:** Every inspection outputs a deterministic scalar score, spatial heatmap, and configuration record, ensuring traceability in regulated automotive, medical, and aerospace electronics manufacturing.

---

## 5. Failure Transparency & Metric Integrity

SolSight prioritizes honest, disaggregated reporting:
- False-positive rates (FPR) and per-class recall rates (TPR) are published independently across all resolution tiers.
- Demonstration scripts explicitly include near-miss and failure exhibits (e.g., subtle sub-2% voids) with physical rationales explaining why the failure occurs and how threshold calibration balances defect escape rates against false reject costs.
