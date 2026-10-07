# AstroFoundation — Team Coordination Document (v3.0)
### Final Role Assignment · No Docker · Parallel Execution Plan

> **Purpose:** Complete project context + role assignments for all three members.
> Paste this into any AI assistant to get accurate help for your specific role.
> Last updated: July 2026

---

## Part 1 — Project Identity

| Field | Value |
|---|---|
| **Project Name** | AstroFoundation |
| **Type** | Final Year Undergraduate Capstone (CS — AI & ML) |
| **University** | VIT Vellore, Batch 2023–2027 |
| **Domain** | Space & Astronomy + AI / ML |
| **Goal** | Multi-modal AI system for astronomical event classification and scientific report generation |
| **Team** | Krissh · Yash · Rishi |
| **Duration** | 6 months · 26 weeks · 10 phases |
| **Setup** | conda + requirements.txt (no Docker) |

---

## Part 2 — Problem Statement

Astronomical surveys like NASA's Kepler and TESS missions generate massive volumes
of data across multiple modalities — photometric light curves, FITS telescope images,
and spectroscopic readings. Current classification pipelines use only one modality
at a time, missing the complementary information locked in the others.

**AstroFoundation** fuses all three data types to:
1. **Classify** astronomical events across 7 categories
2. **Flag anomalies** — events that don't match any known class
3. **Cross-reference** detections against SIMBAD, NASA ADS, and NED databases
4. **Generate** publication-style scientific reports using LLMs

---

## Part 3 — The 7 Event Classes

| Class | What it is | Key light curve signature |
|---|---|---|
| `exoplanet_transit` | Planet crossing in front of star | Tiny periodic symmetric dip |
| `eclipsing_binary` | Two stars orbiting each other | Two alternating unequal dips |
| `variable_star` | Star pulsating intrinsically | Smooth sinusoidal oscillation |
| `supernova` | Star exploding | Sudden spike → slow exponential decay |
| `cataclysmic_variable` | White dwarf stealing matter | Repeated outbursts, returns to baseline |
| `gamma_ray_burst` | Most energetic explosion known | Extremely brief intense spike |
| `anomaly` | Doesn't fit any known class | Model says "I don't know" → OOD flag |

---

## Part 4 — Full System Architecture

```
NASA Telescope Data (Kepler / TESS)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              DATA INGESTION & PREPROCESSING          │  ← YASH
│  lightkurve → light curves                          │
│  astropy    → FITS images                           │
│  specutils  → spectral data                         │
│  sigma clipping · normalization · percentile clip   │
└────────────┬──────────────┬──────────────┬──────────┘
             │              │              │
             ▼              ▼              ▼
     ┌──────────────┐ ┌──────────┐ ┌────────────────┐
     │ 1D TRANSFORM │ │ CNN/ViT  │ │  SPECTRAL MLP  │  ← KRISSH
     │ ER ENCODER   │ │ ENCODER  │ │  ENCODER       │
     │ [256-dim]    │ │ [256-dim]│ │  [256-dim]     │
     └──────┬───────┘ └────┬─────┘ └───────┬────────┘
            │              │               │
            ▼              ▼               ▼
     ┌──────────────────────────────────────────────┐
     │         PROJECTION HEADS (256→256)            │  ← KRISSH
     │   LC proj · Image proj · Spectral proj        │
     │   Maps all 3 into shared embedding space      │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │      MULTI-MODAL CROSS-ATTENTION FUSION       │  ← KRISSH
     │   Pairwise cross-attention (LC↔Img, LC↔Spec, │
     │   Img↔Spec) + Gating network (σ weights)     │
     │   Modality dropout during training            │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │              TASK HEADS                       │  ← KRISSH
     │   Classification head (6 classes)             │
     │   Anomaly detection head (OOD energy scoring) │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │         LANGGRAPH ReAct AGENT                 │  ← KRISSH
     │   Reason → Tool call → Reason → Tool call    │
     │   Tools: SIMBAD (Yash) · NED (Yash) ·        │
     │          NASA ADS (Rishi)                     │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │         LLM REPORT GENERATION                 │  ← RISHI
     │   GPT-4o / Gemini · Pydantic validation       │
     │   → AstronomicalReport                        │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │              FastAPI SERVICE                  │  ← RISHI
     │   POST /analyze · GET /status · GET /health  │
     └──────────────────────────────────────────────┘
```

---

## Part 5 — Final Role Assignments

---

### 🔴 KRISSH — Model Architecture & Agent

> Krissh builds the brain of the system.
> Every neural network, every attention mechanism, the fusion module,
> the task heads, and the autonomous reasoning agent.

**Owns:**
- All three encoder architectures (Transformer, CNN/ViT, Spectral MLP)
- Projection heads (inter-model pipeline)
- Multi-modal cross-attention fusion + gating network
- Task heads (classification + anomaly detection)
- Full model assembly (`full_model.py`)
- Inter-model tensor validation
- LangGraph ReAct agent

**Does NOT own:**
- Data loading and preprocessing (Yash)
- Training loops and W&B logging (Rishi)
- LLM report generation (Rishi)
- Evaluation metrics and explainability (Rishi)
- FastAPI server (Rishi)

---

### 🟢 YASH — Data Pipelines & Dataset Classes

> Yash builds everything that happens before data enters Krissh's models.
> Unchanged from original plan.

**Owns:**
- lightkurve ingestion (light curves)
- astropy ingestion (FITS images)
- specutils ingestion (spectral data)
- All preprocessing (sigma clip, normalization, percentile clip)
- All three PyTorch Dataset classes (LightCurve, FITS, Spectral)
- DataLoader factory
- Train/val/test split logic
- MC Dropout (inference uncertainty)
- Temperature scaling (calibration)
- Cosine warmup LR scheduler
- Active learning loop (modAL)
- SIMBAD tool wrapper
- NED tool wrapper
- Unit tests for datasets

---

### 🔵 RISHI — Integration, Training & Reporting

> Rishi wires everything together, runs training, and produces all outputs.

**Owns:**
- Training loops for all encoders and fusion model
- W&B experiment logging
- Ablation study runner (11 configurations × 3 seeds)
- Evaluation metrics (Macro F1, per-class F1, AUROC, ECE)
- Grad-CAM explainability
- SHAP explainability
- LLM report generation (GPT-4o / Gemini)
- Pydantic schemas (all of them)
- NASA ADS tool wrapper
- LangGraph tool integration (wires Yash's + his tools into Krissh's agent)
- FastAPI server (main.py, routes.py, background_tasks.py)
- CI/CD (GitHub Actions)
- README and HuggingFace upload
- Final benchmarks vs published baselines

---

## Part 6 — File Ownership

| File | Owner |
|---|---|
| `environment.yml` | Rishi |
| `requirements.txt` | Rishi |
| `.gitignore` | Rishi |
| `.github/workflows/ci.yml` | Rishi |
| `README.md` | Rishi |
| `scripts/download_data.py` | Rishi |
| `scripts/train_encoders.py` | Rishi |
| `scripts/train_fusion.py` | Rishi |
| `scripts/run_ablations.py` | Rishi |
| `scripts/extract_embeddings.py` | Rishi |
| `src/data/ingestion.py` | Yash |
| `src/data/preprocessing.py` | Yash |
| `src/data/datasets.py` | Yash (all 3 classes) |
| `src/data/dataloaders.py` | Yash |
| `src/models/transformer_encoder.py` | **Krissh** |
| `src/models/image_encoder.py` | **Krissh** |
| `src/models/spectral_encoder.py` | **Krissh** |
| `src/models/projection_heads.py` | **Krissh** |
| `src/models/fusion.py` | **Krissh** |
| `src/models/task_heads.py` | **Krissh** |
| `src/models/full_model.py` | **Krissh** |
| `src/models/embedding_validator.py` | **Krissh** |
| `src/agent/agent.py` | **Krissh** |
| `src/agent/tools.py` | Yash (SIMBAD+NED) + Rishi (ADS) |
| `src/agent/schemas.py` | Rishi |
| `src/training/trainer.py` | Rishi |
| `src/training/losses.py` | Rishi |
| `src/training/schedulers.py` | Yash |
| `src/training/active_learning.py` | Yash |
| `src/training/uncertainty.py` | Yash |
| `src/training/calibration.py` | Yash |
| `src/evaluation/metrics.py` | Rishi |
| `src/evaluation/explainability.py` | Rishi |
| `src/evaluation/benchmarks.py` | Rishi |
| `src/reporting/llm_reporter.py` | Rishi |
| `src/reporting/prompts.py` | Rishi |
| `src/reporting/report_schema.py` | Rishi |
| `src/api/main.py` | Rishi |
| `src/api/routes.py` | Rishi |
| `src/api/background_tasks.py` | Rishi |
| `tests/test_datasets.py` | Yash |
| `tests/test_models.py` | **Krissh** |
| `tests/test_agent.py` | **Krissh** |
| `tests/test_api.py` | Rishi |
| `configs/model_config.yaml` | **Krissh** |
| `configs/training_config.yaml` | Rishi |
| `configs/agent_config.yaml` | **Krissh** |
| `configs/wandb_config.yaml` | Rishi |
| `notebooks/01_eda_light_curves.ipynb` | Yash |
| `notebooks/02_eda_fits_images.ipynb` | **Krissh** ✅ IN PROGRESS |
| `notebooks/03_eda_spectra.ipynb` | **Krissh** |
| `notebooks/04_encoder_experiments.ipynb` | Rishi |
| `notebooks/05_fusion_ablations.ipynb` | Rishi |
| `notebooks/06_final_results.ipynb` | Rishi |

---

## Part 7 — Phase-by-Phase Parallel Plan

> Within each phase all three work simultaneously on different files.
> Between phases everyone must finish before moving forward.

---

### Phase 0 — Repo Bootstrap
**Week 1 · Rishi alone · Yash and Krissh wait then clone**

| Member | Task |
|---|---|
| 🔵 Rishi | Create GitHub repo, folder structure, environment.yml, requirements.txt, .gitignore, branch protection → PUSH |
| 🟢 Yash | Wait → clone |
| 🔴 Krissh | Wait → clone |

```
GATE 0: Rishi pushes → everyone clones → Phase 1 begins
```

---

### Phase 1 — Astronomy Data Exploration
**Weeks 1–2 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Set up GitHub Actions CI skeleton | `.github/workflows/ci.yml` |
| 🔵 Rishi | Write download_data.py | `scripts/download_data.py` |
| 🔵 Rishi | Set up W&B project | W&B dashboard |
| 🟢 Yash | Explore lightkurve, download Kepler/TESS curves | local |
| 🟢 Yash | Research sigma-clipping strategies | notes |
| 🟢 Yash | Build EDA notebook 01 (light curves) | `notebooks/01_eda_light_curves.ipynb` |
| 🔴 Krissh | Install astropy, lightkurve, matplotlib | local ✅ |
| 🔴 Krissh | Download Kepler-10 TPF, inspect HDU structure | `notebooks/02_eda_fits_images.ipynb` ✅ |
| 🔴 Krissh | Extract header, aperture mask, flux frame | `notebooks/02_eda_fits_images.ipynb` ✅ |
| 🔴 Krissh | Visualize aperture + flux, 3-channel conversion | `notebooks/02_eda_fits_images.ipynb` ✅ |
| 🔴 Krissh | Percentile clipping | `notebooks/02_eda_fits_images.ipynb` 🔄 |
| 🔴 Krissh | Complete notebook 02 | `notebooks/02_eda_fits_images.ipynb` |
| 🔴 Krissh | Survey specutils, build notebook 03 | `notebooks/03_eda_spectra.ipynb` |

```
GATE 1: All 3 EDA notebooks complete → team agrees on 3-modality approach
```

---

### Phase 2 — PyTorch Dataset Pipeline
**Weeks 3–4 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Write train/val/test split logic | `src/data/dataloaders.py` |
| 🔵 Rishi | Build DataLoader factory | `src/data/dataloaders.py` |
| 🔵 Rishi | Build baseline MLP, write training script | `scripts/train_encoders.py` |
| 🟢 Yash | Implement LightCurveDataset | `src/data/datasets.py` |
| 🟢 Yash | Implement FITSDataset | `src/data/datasets.py` |
| 🟢 Yash | Implement SpectralDataset | `src/data/datasets.py` |
| 🟢 Yash | Write ingestion.py + preprocessing.py | `src/data/` |
| 🔴 Krissh | Define model config YAML | `configs/model_config.yaml` |
| 🔴 Krissh | Write embedding_validator.py | `src/models/embedding_validator.py` |
| 🔴 Krissh | Write stub for transformer_encoder.py | `src/models/transformer_encoder.py` |
| 🔴 Krissh | Write test_models.py (dummy tensor tests) | `tests/test_models.py` |

```
GATE 2: All 3 Dataset classes pass tests → DataLoader factory working
        → baseline MLP trains → encoder stubs accept correct input shapes
```

---

### Phase 3 — 1D Transformer Encoder
**Weeks 5–7 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Write trainer.py training loop + W&B logging | `src/training/trainer.py` |
| 🔵 Rishi | Write losses.py (focal loss) | `src/training/losses.py` |
| 🔵 Rishi | Run 3-seed LC encoder training, log to W&B | W&B dashboard |
| 🟢 Yash | Implement cosine warmup LR scheduler | `src/training/schedulers.py` |
| 🟢 Yash | Implement sigma clipping + normalization | `src/data/preprocessing.py` |
| 🟢 Yash | Write test_datasets.py | `tests/test_datasets.py` |
| 🔴 Krissh | Implement positional encoding | `src/models/transformer_encoder.py` |
| 🔴 Krissh | Implement multi-head self-attention | `src/models/transformer_encoder.py` |
| 🔴 Krissh | Implement Transformer blocks (MHA + FFN + LayerNorm) | `src/models/transformer_encoder.py` |
| 🔴 Krissh | Implement CLS token + 256-dim output head | `src/models/transformer_encoder.py` |
| 🔴 Krissh | Validate output shape: (batch, 256) | `tests/test_models.py` |

```
GATE 3: transformer_encoder.py outputs (batch, 256) confirmed
        → 3-seed training runs complete → W&B logs look healthy
```

---

### Phase 4 — CNN / ViT Image Encoder
**Weeks 8–10 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Write eval metrics: Macro F1, per-class F1 | `src/evaluation/metrics.py` |
| 🔵 Rishi | Produce MLP vs LC Transformer benchmark table | `notebooks/04_encoder_experiments.ipynb` |
| 🔵 Rishi | Run 3-seed image encoder training | W&B dashboard |
| 🟢 Yash | Implement percentile clip + resize 224×224 | `src/data/preprocessing.py` |
| 🟢 Yash | Finalize FITSDataset with all preprocessing | `src/data/datasets.py` |
| 🟢 Yash | Write unit tests for FITS preprocessing | `tests/test_datasets.py` |
| 🔴 Krissh | Implement image_encoder.py (ResNet-50 path) | `src/models/image_encoder.py` |
| 🔴 Krissh | Implement ViT path via timm | `src/models/image_encoder.py` |
| 🔴 Krissh | Write FITS → 3-channel conversion utility | `src/models/image_encoder.py` |
| 🔴 Krissh | Validate output shape: (batch, 256) | `tests/test_models.py` |
| 🔴 Krissh | Compare ResNet-50 vs ViT in notebook | `notebooks/02_eda_fits_images.ipynb` |

```
GATE 4: image_encoder.py outputs (batch, 256) → ResNet vs ViT comparison done
```

---

### Phase 5 — Spectral MLP Encoder
**Weeks 11–12 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Implement Grad-CAM (captum) | `src/evaluation/explainability.py` |
| 🔵 Rishi | Run 3-seed spectral encoder training | W&B dashboard |
| 🔵 Rishi | Produce 3-encoder side-by-side benchmark | `notebooks/04_encoder_experiments.ipynb` |
| 🟢 Yash | Implement spectral feature engineering via specutils | `src/data/preprocessing.py` |
| 🟢 Yash | Download SDSS/LAMOST spectra, configure DVC | `data/raw/spectra/` |
| 🟢 Yash | Finalize SpectralDataset | `src/data/datasets.py` |
| 🔴 Krissh | Implement spectral_encoder.py (residual MLP) | `src/models/spectral_encoder.py` |
| 🔴 Krissh | Implement BatchNorm + GELU + skip connections | `src/models/spectral_encoder.py` |
| 🔴 Krissh | Integrate PCA(128) dimensionality reduction | `src/models/spectral_encoder.py` |
| 🔴 Krissh | Validate output shape: (batch, 256) | `tests/test_models.py` |

```
GATE 5: All 3 encoders output (batch, 256) confirmed
        → Each trained independently → benchmark table complete
```

---

### Phase 6 — Multi-Modal Fusion Architecture
**Weeks 13–16 · All three parallel · LONGEST PHASE**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Write train_fusion.py (dual LR) | `scripts/train_fusion.py` |
| 🔵 Rishi | Run 3-seed fusion training | W&B dashboard |
| 🔵 Rishi | Implement SHAP explainability | `src/evaluation/explainability.py` |
| 🔵 Rishi | Build ablation scaffold + run 11 configs | `src/evaluation/benchmarks.py` |
| 🔵 Rishi | Build fusion ablations notebook | `notebooks/05_fusion_ablations.ipynb` |
| 🟢 Yash | Implement MC Dropout (50 forward passes) | `src/training/uncertainty.py` |
| 🟢 Yash | Implement temperature scaling calibration | `src/training/calibration.py` |
| 🟢 Yash | Write active_learning.py (modAL) | `src/training/active_learning.py` |
| 🔴 Krissh | Implement projection_heads.py (256→256 × 3) | `src/models/projection_heads.py` |
| 🔴 Krissh | Implement pairwise cross-attention (LC↔Img, LC↔Spec, Img↔Spec) | `src/models/fusion.py` |
| 🔴 Krissh | Implement gating network (σ-weighted sum) | `src/models/fusion.py` |
| 🔴 Krissh | Implement modality dropout | `src/models/fusion.py` |
| 🔴 Krissh | Assemble full_model.py end-to-end | `src/models/full_model.py` |
| 🔴 Krissh | Validate full forward pass: input → (batch, 256) fused | `tests/test_models.py` |

```
GATE 6: full_model.py forward pass works on dummy tensors
        → Fused model beats all single encoders by >5% macro F1
        → Ablation table (11 configs) complete
```

---

### Phase 7 — Task Heads, Anomaly Detection & Uncertainty
**Weeks 17–18 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Implement AUROC metric | `src/evaluation/metrics.py` |
| 🔵 Rishi | Implement ECE calibration metric | `src/evaluation/metrics.py` |
| 🔵 Rishi | Generate calibration curves + UMAP plots | `outputs/figures/` |
| 🔵 Rishi | Write OOD energy loss | `src/training/losses.py` |
| 🟢 Yash | Run active learning loop (20 × 10 rounds) | `src/training/active_learning.py` |
| 🟢 Yash | Log delta F1 after each retrain | W&B dashboard |
| 🔴 Krissh | Implement classification head (6 classes + focal loss input) | `src/models/task_heads.py` |
| 🔴 Krissh | Implement anomaly detection head (energy-based OOD) | `src/models/task_heads.py` |
| 🔴 Krissh | Define TaskHeadOutput dataclass | `src/models/task_heads.py` |
| 🔴 Krissh | Wire task heads into full_model.py | `src/models/full_model.py` |
| 🔴 Krissh | Validate: full model outputs TaskHeadOutput correctly | `tests/test_models.py` |

```
GATE 7: Full model input → TaskHeadOutput pipeline confirmed
        → OOD AUROC > 0.85 → ECE < 0.05 after temperature scaling
```

---

### Phase 8 — LangGraph Agent & FastAPI
**Weeks 19–21 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Build FastAPI main.py + routes.py + background_tasks.py | `src/api/` |
| 🔵 Rishi | Define all Pydantic schemas | `src/agent/schemas.py` |
| 🔵 Rishi | Implement NASA ADS tool wrapper | `src/agent/tools.py` |
| 🔵 Rishi | Write test_api.py (5 endpoint tests) | `tests/test_api.py` |
| 🟢 Yash | Implement SIMBAD tool wrapper (retry logic) | `src/agent/tools.py` |
| 🟢 Yash | Implement NED tool wrapper (retry logic) | `src/agent/tools.py` |
| 🟢 Yash | Write test_agent.py | `tests/test_agent.py` |
| 🔴 Krissh | Define agent state (typed LangGraph state) | `src/agent/agent.py` |
| 🔴 Krissh | Implement reason_node (LLM decides next tool) | `src/agent/agent.py` |
| 🔴 Krissh | Implement simbad_node, ads_node, ned_node | `src/agent/agent.py` |
| 🔴 Krissh | Implement conditional edges (confidence thresholds) | `src/agent/agent.py` |
| 🔴 Krissh | Compile and test LangGraph agent locally | `src/agent/agent.py` |
| 🔴 Krissh | Write agent_config.yaml | `configs/agent_config.yaml` |

```
GATE 8: FastAPI live → agent successfully queries all 3 databases
        → All 5 endpoints responding < 10 seconds
```

---

### Phase 9 — LLM Reports & Active Learning
**Weeks 22–23 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Implement llm_reporter.py (GPT-4o / Gemini) | `src/reporting/llm_reporter.py` |
| 🔵 Rishi | Write system + user prompts (Jinja2) | `src/reporting/prompts.py` |
| 🔵 Rishi | Define AstronomicalReport Pydantic model | `src/reporting/report_schema.py` |
| 🔵 Rishi | Wire LLM into FastAPI /analyze endpoint | `src/api/routes.py` |
| 🟢 Yash | Finalize active learning loop | `src/training/active_learning.py` |
| 🟢 Yash | Run 10 active learning rounds, log delta F1 | W&B dashboard |
| 🔴 Krissh | Wire agent output into Rishi's LLM reporter | `src/agent/agent.py` |
| 🔴 Krissh | Test full pipeline: model → agent → report | end-to-end test |
| 🔴 Krissh | Edge case handling in agent (low confidence, missing data) | `src/agent/agent.py` |

```
GATE 9: Star ID in → AstronomicalReport out (Pydantic-validated)
        → Active learning demonstrably improving F1
```

---

### Phase 10 — Benchmarks, Polish & Release
**Weeks 24–26 · All three parallel**

| Member | Task | File |
|---|---|---|
| 🔵 Rishi | Write final README (results + architecture + quick-start) | `README.md` |
| 🔵 Rishi | Set up GitHub Actions CI (pytest on push) | `.github/workflows/ci.yml` |
| 🔵 Rishi | Upload model weights to HuggingFace Hub | HuggingFace |
| 🔵 Rishi | Run final benchmarks vs published baselines | `src/evaluation/benchmarks.py` |
| 🔵 Rishi | Build final results notebook | `notebooks/06_final_results.ipynb` |
| 🟢 Yash | Extract and store all embeddings | `scripts/extract_embeddings.py` |
| 🟢 Yash | Write HuggingFace model card | HuggingFace |
| 🟢 Yash | Final data pipeline polish + DVC audit | `src/data/` |
| 🔴 Krissh | Final model architecture review and cleanup | `src/models/` |
| 🔴 Krissh | Ensure all model tests pass on CI | `tests/test_models.py` |
| 🔴 Krissh | Document all model configs clearly | `configs/model_config.yaml` |
| 🔴 Krissh | Write 2-minute interview pitch | notes |

```
GATE 10 (RELEASE): All tests pass on CI → HuggingFace upload done
                   → Public README polished → v1.0.0 tagged
```

---

## Part 8 — Key Architecture Numbers (Krissh Must Memorize)

| Parameter | Value | Why |
|---|---|---|
| Embedding dim (all encoders) | **256** | Fusion expects exactly this — never change |
| LC sequence length | 1024 timesteps | Input to Transformer |
| Transformer d_model | 128 | Internal dimension |
| Transformer n_heads | 8 | Attention heads |
| Transformer n_layers | 6 | Encoder blocks |
| Transformer d_ff | 512 | Feed-forward hidden dim |
| FITS image input | 224 × 224 × 3 | ResNet-50/ViT standard |
| Spectral features after PCA | 128 | Input to Spectral MLP |
| Spectral hidden dims | [256, 256, 256] | 3 residual MLP blocks |
| Cross-attention heads | 8 | In fusion module |
| Cross-attention layers | 2 | Per modality pair |
| Modality dropout prob | 0.15 | Drop one modality during training |
| MC Dropout passes | 50 | Uncertainty estimation at inference |
| Training seeds | 3 (42, 123, 456) | Statistical significance |

---

## Part 9 — Krissh's Interview Story (Revised)

> "My role in AstroFoundation was building the model architecture end to end.
> I implemented all three encoders from scratch in PyTorch —
> a 1D Transformer with multi-head self-attention for light curves,
> a ResNet-50 and ViT comparison for FITS telescope images,
> and a residual MLP for spectral data — each outputting a 256-dim embedding.
> I then built the multi-modal fusion module using pairwise cross-attention
> between all three modalities with a learned gating network,
> which gave us over 5% improvement in macro F1 over the best single-modality baseline.
> On top of the model I built a LangGraph ReAct agent that autonomously
> cross-references every prediction against SIMBAD, NASA ADS, and NED databases
> before passing enriched findings to an LLM report generator.
> The whole model pipeline is validated at every interface —
> every encoder's output is shape-checked before entering fusion,
> and the agent's output is schema-validated before entering the LLM layer."

---

## Part 10 — Why No Docker

Docker was removed from this project. Here's why it was originally included
and why it's fine without it:

| Docker would have given | Alternative we use instead |
|---|---|
| One-command deployment | `conda env create -f environment.yml` |
| Reproducible environment | `environment.yml` + `requirements.txt` |
| CUDA version pinning | Documented in README (CUDA 11.8 / 12.1) |
| Easy sharing | GitHub clone + conda setup |

For a student capstone project that is not running in production,
Docker adds 2–3 weeks of engineering effort for no real benefit.
Anyone cloning the repo can be up and running in under 10 minutes
with conda and requirements.txt.

---

## Part 11 — Git Workflow

```
main (protected — never push directly)
  └── dev
        ├── krissh/phase-N-description    ← Krissh's branches
        ├── yash/phase-N-description      ← Yash's branches
        └── rishi/phase-N-description     ← Rishi's branches
```

**Rules:**
1. Never push directly to `main` or `dev`
2. Each member works on their own feature branch
3. At each phase gate — all three open PRs into `dev`
4. After all PRs merged and gate confirmed — merge `dev` → `main`
5. Branch naming: `<name>/phase-<N>-<short-description>`

---

## Part 12 — Setup (No Docker)

```bash
# Clone
git clone https://github.com/RishiChalana/AstroFoundation
cd AstroFoundation

# Create environment
conda env create -f environment.yml
conda activate astrofoundation

# Pull large data files (DVC)
dvc pull

# Download data
python scripts/download_data.py --kic_ids path/to/kic_ids.txt

# Train encoders (Rishi runs these)
python scripts/train_encoders.py --encoder lc_transformer --config configs/model_config.yaml
python scripts/train_encoders.py --encoder image_encoder
python scripts/train_encoders.py --encoder spectral_mlp

# Train fusion model (Rishi runs this)
python scripts/train_fusion.py --config configs/model_config.yaml

# Run ablation study (Rishi runs this)
python scripts/run_ablations.py --output outputs/ablation_results.csv

# Start API server (Rishi runs this)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
pytest tests/ -v

# Analyze a star
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"star_id": "KIC_9002278", "kic_id": 9002278, "include_agent": true, "include_report": true}'
```

---

## Part 13 — Instructions for AI Assistants

When a team member asks for help, focus only on their layer:

- **Krissh asks for help** → focus on `src/models/`, `src/agent/agent.py`,
  `configs/model_config.yaml`, `tests/test_models.py`.
  Krissh does NOT own training loops, metrics, LLM generation, or FastAPI.

- **Yash asks for help** → focus on `src/data/`, lightkurve, astropy, specutils,
  preprocessing, PyTorch Dataset classes, active learning, schedulers.

- **Rishi asks for help** → focus on `src/training/`, `src/evaluation/`,
  `src/reporting/`, `src/api/`, `src/agent/schemas.py`, W&B logging,
  FastAPI, Pydantic schemas, LLM prompting.

### Learning agreement for Krissh
- Library function names → tell directly
- Logic / algorithm / architecture decisions → hint first, then pseudocode, then code only as last resort
- Never rewrite working code — surgical fixes only
- All embeddings must be exactly 256-dim

---

*Document version: 3.0 — July 2026*
*Krissh: models + agent | Yash: data | Rishi: training + eval + LLM + API*
