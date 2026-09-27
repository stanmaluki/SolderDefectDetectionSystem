# SolSight — Responsible AI & Data Governance Statement

## 1. Data Provenance and Licensing

All training datasets used in SolSight are procedurally generated via mathematical and parametric modeling of solder fillet geometry and PCBA optics.
- **Privacy & Proprietary Protection:** No private customer Gerber files, CAD layouts, or proprietary industrial board designs are contained in the training corpus.
- **Intellectual Property:** Procedural generation algorithms synthesize imagery directly from first-principles 3D optics (Blinn-Phong illumination, Lambertian diffuse models, and spherical geometry), avoiding copyright infringement or unauthorized web scraping.
- **Real-World Validation Data:** Public evaluation benchmarks (such as the *SolDef_AI* solder joint defect dataset on Kaggle) are utilized exclusively under their declared academic and open research licenses with transparent citation and attribution.

---

## 2. Model Bias and Operating Limitations

Because SolSight is trained in an unsupervised manner exclusively on defect-free reference samples, its internal definition of a "valid solder joint" is bounded by its training distribution:
1. **Solder Alloy Variation:** Industrial soldering processes utilize distinct alloys (e.g. SAC305 lead-free vs. Sn63Pb37 tin-lead vs. bismuth-based low-temp alloys). Differences in alloy reflectivity, wetting angle, or grain structure not modeled during synthesis can produce elevated baseline dissimilarity.
2. **Surface Contaminants & Flux Residue:** Harmless no-clean flux amber residue around a pad can register as structural dissimilarity if unmodeled, potentially causing false-positive flags.
3. **Micro-Scale Resolution Bounds:** At extreme micro-scales (15–16px), the $11\times11$ Gaussian SSIM window covers over two-thirds of the patch, reducing localized spatial precision.

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
