# SolSight — Responsible AI & Data Governance Statement

## 1. Data Provenance and Licensing

All training, validation, calibration, and test datasets in this release are **100% procedurally synthesized** via physical and mathematical modeling of 3D solder fillet geometry, multi-color solder masks, and PCBA optics.
- **Synthetic-Procedural Pipeline:** Algorithms synthesize imagery directly from first-principles 3D optics (Blinn-Phong specular illumination, Lambertian diffuse models, and spherical/paraboloid geometry), eliminating copyright infringement, proprietary NDA exposure, or unauthorized scraping.
- **Privacy & Proprietary Protection:** No customer Gerber files, proprietary CAD designs, or commercial PCBA schematics are contained in the training corpus.
- **Real-World PCBA Benchmark Status (Future Work):** Real-world PCBA dataset ingestion (such as *SolDef_AI* on Kaggle) is supported via `scripts/fetch_real_data.py` only when authenticated with valid Kaggle API credentials. When unauthenticated, the script halts with a clear error. **No procedural renders are ever mislabeled or substituted for real data.** All benchmark metrics reported in this submission are derived from held-out procedural synthetic evaluation. Physical PCBA validation on a live production line remains an explicit, transparent future milestone.

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
