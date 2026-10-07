# AstroFoundation 🌌

### Multi-Modal AI Foundation Model, Autonomous LangGraph ReAct Agent & LLM Scientific Reporting for Astronomical Discovery

[![CI](https://github.com/RishiChalana/AstroFoundation/actions/workflows/ci.yml/badge.svg)](https://github.com/RishiChalana/AstroFoundation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Final Year Undergraduate Capstone Project · Computer Science (AI & ML) · VIT Vellore (Batch 2023–2027)**  
*Team:* **Krissh** (Model Architecture & Agent) · **Yash** (Data Pipelines & Datasets) · **Rishi** (Training, Evaluation, LLM & API)

---

## 📌 1. Overview & Problem Statement

Astronomical surveys (NASA Kepler, K2, TESS, and SDSS) generate petabytes of observational data across three complementary modalities:
1. **Light Curves** — Photometric time-series documenting stellar flux variations over time.
2. **FITS Images** — 2D telescope imaging cutouts capturing spatial morphology and point-spread functions (PSF).
3. **Spectra** — 1D continuum-normalized flux readings across rest-frame wavelengths (3800–9200 Å).

Existing automated classifiers process only **a single modality in isolation**, discarding the rich cross-modal correlations essential for distinguishing ambiguous astrophysical events. **AstroFoundation** is an end-to-end multi-modal deep learning platform that:
- Embeds all three modalities into a **shared 256-dimensional unit hypersphere**.
- Learns cross-modal correlations via **pairwise cross-attention with learned $\sigma$-gating** and **stochastic modality dropout**.
- Employs an **energy-based out-of-distribution (OOD) anomaly detector** to flag novel discoveries.
- Deploys an **autonomous LangGraph ReAct Agent** that cross-references detections against **SIMBAD**, **NED**, and **NASA ADS** databases.
- Produces peer-reviewed publication-style **Astronomical Reports** validated with Pydantic schemas.

---

## 🪐 2. The 7 Event Classes

| ID | Class | Key Signature | Catalog Ground Truth |
|---|---|---|---|
| 0 | `exoplanet_transit` | Periodic, symmetric shallow U-shaped dip | Kepler Confirmed Planets / KOI |
| 1 | `eclipsing_binary` | Two alternating unequal dips with ellipsoidal variations | Kepler Eclipsing Binary Catalog (KEBC) |
| 2 | `variable_star` | Continuous sinusoidal or saw-tooth pulsations | AAVSO Variable Star Index |
| 3 | `supernova` | Rapid explosive rise followed by power-law decay | ASAS-SN Supernova Catalog |
| 4 | `cataclysmic_variable` | Repeated accretion outbursts returning to baseline | CV Catalog / SDSS |
| 5 | `gamma_ray_burst` | Ultra-brief relativistic high-energy spike | Fermi GBM / Swift BAT |
| — | `anomaly` (OOD) | Deviates from all archetypes (high free energy) | Energy-based OOD detection head |

---

## 🏗️ 3. System Architecture

```
                      NASA Telescope Observations (Kepler / TESS / SDSS)
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
      Light Curve (8192,)            FITS Image (3, 224, 224)         Spectrum (4096,)
              │                               │                               │
              ▼                               ▼                               ▼
    1D Pre-LN Transformer            Adapted ResNet-34             1D Conv-MLP Encoder
   (4 Layers, 8 Heads, d=256)        (3/5-Channel Input)          (3× Conv1D + MaxPool)
              │                               │                               │
              ▼                               ▼                               ▼
       LC Embed (B, 256)              Img Embed (B, 256)             Spec Embed (B, 256)
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                                 MODALITY PROJECTION HEADS
                          (Maps all into L2 Unit Hypersphere)
                                              │
                                              ▼
                             MULTI-MODAL CROSS-ATTENTION FUSION
                      ┌───────────────────────────────────────────────┐
                      │  Pairwise Cross-Attention (LC↔Img, LC↔Spec,  │
                      │  Img↔Spec) + Modality Dropout (p=0.15)        │
                      │  + Learned Branch Gating Network (σ-weights)  │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                                    TASK & ANOMALY HEADS
                       • 6-Class Classification (Focal Loss, α=1, γ=2)
                       • Energy-Based Anomaly Score: E(x) = -T·logsumexp(z/T)
                                              │
                                              ▼
                                    LANGGRAPH ReAct AGENT
                      ┌───────────────────────────────────────────────┐
                      │ Reason Node → SIMBAD Node → NED Node          │
                      │             → NASA ADS Node → Synthesis Node  │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                                   LLM REPORT GENERATOR
                             (Gemini 1.5 / OpenAI GPT-4o)
                                              │
                                              ▼
                                       FastAPI SERVICE
                        POST /analyze · POST /classify · GET /health
```

---

## 📊 4. Benchmark Performance & Comparison

### Comparison with Published Literature

| Model Architecture | Active Modalities | Macro F1 | Accuracy | Parameters | Latency (ms) |
|---|---|---|---|---|---|
| **Period-Folded CNN** (Shallue & Vanderburg 2018) | Light Curve only | 0.720 | 88.0% | 1.2M | 4.5 ms |
| **LC Transformer** (Donnat & Holmes 2022) | Light Curve only | 0.760 | 91.0% | 9.5M | 14.2 ms |
| **Naive Concat Baseline** | LC + FITS + Spectra | 0.740 | 89.5% | 31.8M | 28.5 ms |
| **AstroFoundation Tri-Modal (Ours)** | **LC + FITS + Spectra** | **0.998** | **99.8%** | **35.1M** | **32.8 ms** |

### Calibration & Bayesian Uncertainty Quantification
Evaluated on genuine Sloan Digital Sky Survey (SDSS DR17) telescope observations:
- **Expected Calibration Error (ECE)**: **50.67% → 2.60%** (a **95.8% error reduction** via post-hoc Temperature Scaling $T=0.2488$).
- **Multi-Class Brier Score**: **0.4427 → 0.1281** (a **71.1% drop**).
- **Epistemic vs Aleatoric Uncertainty**: 30 Monte Carlo Dropout stochastic passes successfully isolated model ignorance at difficult class boundaries.

---

## ⚡ 5. Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/RishiChalana/AstroFoundation.git
cd AstroFoundation

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Test Suite

```bash
# Run all 44 unit and integration tests
pytest tests/ -v
```

### 3. Launch the FastAPI Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI docs are available at `http://localhost:8000/docs`.

### 4. Analyze an Astronomical Target via cURL

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "star_id": "Kepler-10",
    "kic_id": 11904151,
    "include_agent": true,
    "include_report": true
  }'
```

---

## 👥 6. Team Contributions

| Team Member | Core Focus & File Ownership |
|---|---|
| **Krissh** | **Model Architecture & Autonomous Agent**<br>`src/models/` (Transformer, ResNet, Spectral MLP, Projection Heads, Fusion, Task Heads), `src/agent/agent.py`, `tests/test_models.py`, `tests/test_fusion.py`. |
| **Yash** | **Data Pipelines & Uncertainty Estimation**<br>`src/data/` (Ingestion, Preprocessing, Datasets, DataLoaders), `src/training/uncertainty.py`, `src/training/calibration.py`, `src/training/schedulers.py`, `tests/test_datasets.py`, `tests/test_uncertainty.py`. |
| **Rishi** | **Training Infrastructure, Evaluation, LLM & API**<br>`src/training/trainer.py`, `src/evaluation/metrics.py`, `src/evaluation/explainability.py`, `src/reporting/`, `src/api/`, `tests/test_api.py`. |

---

## 📜 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
