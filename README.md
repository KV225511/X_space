# AstroFoundation 🌌

### Multi-Modal AI Foundation Model, Autonomous LangGraph ReAct Agent, Explainable AI (XAI) & LLM Scientific Reporting for Astronomical Discovery

[![CI](https://github.com/RishiChalana/AstroFoundation/actions/workflows/ci.yml/badge.svg)](https://github.com/RishiChalana/AstroFoundation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19.2%2B-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Final Year Undergraduate Capstone Project · Computer Science (AI & ML) · VIT Vellore (Batch 2023–2027)**  
*Team:* **Krissh** (Model Architecture, Autonomous Agent & Frontend/UI) · **Yash** (Data Pipelines, Ingestion & Uncertainty Quantification) · **Rishi** (Training Infrastructure, Evaluation, LLM Reporting & API)

---

## 📑 Table of Contents

1. [Overview & Problem Statement](#-1-overview--problem-statement)
2. [What's New in AstroFoundation v4.0](#-2-whats-new-in-astrofoundation-v40)
3. [The 7 Astrophysical Event Classes](#-3-the-7-astrophysical-event-classes)
4. [System Architecture](#-4-system-architecture)
5. [Explainable AI (XAI) & Interpretability](#-5-explainable-ai-xai--interpretability)
6. [Interactive Web Dashboard (React 19 + Vite)](#-6-interactive-web-dashboard-react-19--vite)
7. [Benchmark Performance & Calibration](#-7-benchmark-performance--calibration)
8. [API Reference & Microservice Suite](#-8-api-reference--microservice-suite)
9. [Repository Structure](#-9-repository-structure)
10. [Quickstart Guide](#-10-quickstart-guide)
11. [Sample Datasets & Downloader](#-11-sample-datasets--downloader)
12. [Team Contributions](#-12-team-contributions)
13. [License & Acknowledgments](#-13-license--acknowledgments)

---

## 📌 1. Overview & Problem Statement

Modern astronomical sky surveys (**NASA Kepler**, **K2**, **TESS**, and **SDSS DR17**) capture petabytes of observational data across three complementary telescope modalities:
1. **Photometric Light Curves** — Time-series measurements documenting stellar flux variations over time ($8192$ timesteps).
2. **2D FITS Images** — Multi-band telescope cutouts ($3 \times 224 \times 224$) capturing point-spread functions (PSF), centroid shifts, and surrounding field morphology.
3. **1D Rest-Frame Spectra** — Continuum-normalized flux readings across rest-frame wavelengths ($3800$–$9200\text{ \AA}$, $4096$ bins).

Existing astronomical classification pipelines typically evaluate **only a single modality in isolation**, missing the rich cross-modal signatures required to differentiate ambiguous astrophysical phenomena (e.g., distinguishing a true exoplanetary transit from a blended background eclipsing binary).

**AstroFoundation** is an end-to-end multi-modal deep learning platform that:
- Embeds all three modalities into a **shared 256-dimensional unit hypersphere** ($\mathcal{S}^{255}$).
- Fuses multi-modal features via **pairwise cross-attention with learned $\sigma$-gating** and **stochastic modality dropout**.
- Employs an **energy-based out-of-distribution (OOD) anomaly detector** to flag uncatalogued astronomical discoveries.
- Deploys an **autonomous LangGraph ReAct Agent** that cross-references candidate detections in real time against **SIMBAD**, **NED**, and **NASA ADS** databases.
- Integrates an **Explainable AI (XAI)** suite (1D/2D Grad-CAM, SmoothGrad noise saliency, KernelSHAP game-theoretic attribution, and canonical atomic line identification).
- Generates peer-reviewed publication-style **Astronomical Reports** powered by **DeepSeek-V3 / DeepSeek-R1** with automated **Astronomer's Telegram (ATel)** drafts.
- Serves an interactive **React 19 + Vite Web Application** backed by a production-grade **FastAPI microservice**.

---

## ✨ 2. What's New in AstroFoundation v4.0

- 🖥️ **Full-Stack React 19 Web Dashboard (`frontend/`)**: Modern dark-mode interface featuring catalog presets, interactive HTML5 Canvas light curve waveform visualizer, and drag-and-drop FITS image inspection.
- 🔬 **Comprehensive Explainability (XAI) Suite (`src/evaluation/explainability.py`)**: 1D Grad-CAM for light curves, 2D Spatial Grad-CAM for FITS cutouts, SmoothGrad noise suppression, KernelSHAP coalition credit, and automated atomic line identification.
- 📁 **Direct FITS Filepath & Header Inspection (`/api/fits/inspect`)**: Seamless parsing of local `.fits` files with real-time extraction of WCS coordinates, exposure times, telescope filters, pixel statistics, and percentile-clipped colormap previews.
- 🔀 **Adaptive Modality Gating & Optional FITS Computation**: On-the-fly toggling of 2D FITS image processing with adaptive cross-attention reweighting, enabling flexible single-, bi-, or tri-modal evaluation.
- 🤖 **DeepSeek Scientific Reporting & ATel Generator (`src/reporting/`)**: Default reporting engine powered by **DeepSeek-V3 / DeepSeek-R1**, producing physical parameter estimates, false-positive contamination forensics, observational urgency triage, and formatted Astronomer's Telegram (ATel) drafts.
- 🛡️ **Rate Moderation & Telemetry (`src/api/rate_limiter.py`)**: Turn-based quota moderation (`RATE_LIMIT_PER_TURN`), in-memory report caching, and live quota savings metrics.

---

## 🪐 3. The 7 Astrophysical Event Classes

| ID | Class | Key Astronomical Signature | Physical Origin | Catalog Ground Truth |
|:---:|---|---|---|---|
| **0** | `exoplanet_transit` | Periodic, shallow, symmetric U-shaped dip ($\sim 0.01\% - 1\%$ depth) | Exoplanet transiting stellar disk | Kepler Confirmed / KOI |
| **1** | `eclipsing_binary` | Two alternating unequal dips with ellipsoidal variations | Binary star mutual occultation | Kepler Eclipsing Binary Catalog (KEBC) |
| **2** | `variable_star` | Multi-periodic sinusoidal or saw-tooth pulsations | Stellar envelope pulsation (e.g. $\delta$ Scuti, Cepheid) | AAVSO Variable Star Index |
| **3** | `supernova` | Rapid explosive rise followed by power-law radioactive decay | Core collapse or thermonuclear stellar explosion | ASAS-SN Supernova Catalog |
| **4** | `cataclysmic_variable` | Repeated stochastic accretion outbursts returning to baseline | Mass transfer onto white dwarf in close binary | SDSS CV Catalog / AAVSO |
| **5** | `gamma_ray_burst` | Ultra-brief relativistic high-energy spike and optical afterglow | Relativistic jet from compact object merger / collapsar | Fermi GBM / Swift BAT / GCN |
| **—** | `anomaly` (OOD) | Deviates significantly from all archetypes (high free energy $E(x)$) | Uncatalogued or novel astrophysical discovery | Energy-Based OOD Detector |

---

## 🏗️ 4. System Architecture

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
                          (Maps embeddings into L2 Unit Hypersphere)
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
                        • Energy-Based Anomaly Score: E(x) = -T · logsumexp(z / T)
                                               │
                                               ▼
                                     LANGGRAPH ReAct AGENT
                       ┌───────────────────────────────────────────────┐
                       │ Reason Node → SIMBAD Node → NED Node          │
                       │             → NASA ADS Node → Synthesis Node  │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                    EXPLAINABLE AI & XAI ENGINE
                       ┌───────────────────────────────────────────────┐
                       │ • 1D Grad-CAM (Transit Ingress / Pulsations)  │
                       │ • 2D Spatial Grad-CAM (FITS PSF / Centroids)  │
                       │ • SmoothGrad & KernelSHAP Coalitions          │
                       │ • Modality Gating Contribution Breakdown (%)  │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                                    LLM SCIENTIFIC REPORTING
                        • DeepSeek-V3 / DeepSeek-R1 (Gemini / OpenAI fallback)
                        • Derived Physical Parameters ($R_p/R_\star$, $T_{eq}$, $q$)
                        • Contamination Forensics & Observational Urgency Triage
                        • Automated Astronomer's Telegram (ATel) Drafts
                                               │
                                               ▼
                                    FASTAPI & REACT 19 SPA
                        • REST Microservice (`/api/predict`, `/analyze`)
                        • Interactive Web Visualizer (`http://localhost:5173`)
```

---

## 🔍 5. Explainable AI (XAI) & Interpretability

AstroFoundation provides physically grounded interpretability across all modalities:

1. **1D Grad-CAM (`GradCAM1D`)**:
   - Hooks into the 1D Convolutional/Transformer layers to isolate high-gradient temporal or spectral regions.
   - Highlights critical astrophysical phenomena: transit ingress/egress dips, primary vs. secondary eclipse depths, and pulsation harmonics.

2. **2D Spatial Grad-CAM (`GradCAM2D`)**:
   - Evaluates ResNet-34 activations on 2D astronomical FITS cutouts.
   - Verifies whether model attention centers on the target stellar Point Spread Function (PSF) or spurious off-target sources, detecting **blended background eclipsing binary (BBEB)** contamination.

3. **Captum Integrated Gradients & SHAP (`ShapleyExplainer`)**:
   - Computes axiomatic path-integral feature attributions relative to neutral baselines.
   - Accurately ranks the importance of photometric cadence, photometric flux variances, and continuum slopes.

4. **Modality Gating Contribution Breakdown (`MultiModalGatingExplainer`)**:
   - Analyzes learned $\sigma$-gating activation weights across cross-attention branches.
   - Reports dynamic percentage contributions (e.g., $58.5\%$ Light Curve, $28.0\%$ FITS Image, $13.5\%$ Spectra) during inference.

5. **Canonical Rest-Frame Spectral Line Identifier (`LineIdentifier`)**:
   - Matches salient peaks in the spectral gradient maps against canonical atomic transitions:
     $$\text{Balmer Series: } \mathrm{H}_\alpha\,(6562.8\text{ \AA}),\ \mathrm{H}_\beta\,(4861.3\text{ \AA}),\ \mathrm{H}_\gamma\,(4340.5\text{ \AA})$$
     $$\text{Fraunhofer \& Forbidden Lines: } \mathrm{Ca\ II\ H\&K}\,(3933.7/3968.5\text{ \AA}),\ \mathrm{Na\ I\ D}\,(5892.0\text{ \AA}),\ [\mathrm{O\ III}]\,(5006.8\text{ \AA})$$

---

## 🖥️ 6. Interactive Web Dashboard (React 19 + Vite)

The web client (`frontend/`) is a reactive single-page application built with React 19, Vite, and custom dark-theme glassmorphism:

- **Catalog Presets**: Instant selection of benchmark targets:
  - `Kepler-10` (Exoplanet transit host star)
  - `KIC-9002278` (Algol detached eclipsing binary)
  - `KIC-9700322` ($\delta$ Scuti pulsating variable star)
  - `ASASSN-15lh` (Superluminous supernova event)
  - `GW170817-SGRB` (Multi-messenger kilonova & short GRB)
  - `ANOMALY-OOD-99` (Aperiodic stochastic OOD candidate)
- **Real-Time Light Curve Waveform Canvas**: Dynamic HTML5 canvas drawing normalized stellar flux over time with gradient shading and custom crosshairs.
- **Direct Local FITS Path Input**: Load `.fits` files directly from disk; calls `/api/fits/inspect` to instantly preview FITS header metadata (RA/Dec, telescope filter, exposure time) and compute pixel min/max/mean stats.
- **Drag & Drop Upload**: Upload custom FITS cutouts (`.fits`, `.fit`, `.fts`) or optical imaging (`.png`, `.jpg`) with automatic cutout generation and colormap rendering.
- **Adaptive Modality Switch**: Toggle 2D FITS image calculation on or off to inspect model performance differences under single-, bi-, or tri-modal conditions.
- **System Status Indicator**: Displays real-time API connectivity and online system status.
- **DeepSeek Scientific Report View**: Formatted peer-reviewed report viewer complete with executive summary, physical parameter derivations, contamination forensics, urgency badges, and an **Astronomer's Telegram (ATel) 1-Click Copy** action.

---

## 📊 7. Benchmark Performance & Calibration

### Comparison with Published Literature

| Model Architecture | Active Modalities | Macro F1 | Accuracy | Parameters | Latency (ms) |
|---|---|---|---|---|---|
| **Period-Folded CNN** (Shallue & Vanderburg 2018) | Light Curve only | 0.720 | 88.0% | 1.2M | 4.5 ms |
| **LC Transformer** (Donnat & Holmes 2022) | Light Curve only | 0.760 | 91.0% | 9.5M | 14.2 ms |
| **Naive Concat Baseline** | LC + FITS + Spectra | 0.740 | 89.5% | 31.8M | 28.5 ms |
| **AstroFoundation Tri-Modal (Ours)** | **LC + FITS + Spectra** | **0.998** | **99.8%** | **35.1M** | **32.8 ms** |

### Calibration & Bayesian Uncertainty Quantification
Evaluated on genuine Sloan Digital Sky Survey (SDSS DR17) telescope observations:
- **Expected Calibration Error (ECE)**: **$50.67\% \to 2.60\%$** (a **$95.8\%$ error reduction** via post-hoc Temperature Scaling $T=0.2488$).
- **Multi-Class Brier Score**: **$0.4427 \to 0.1281$** (a **$71.1\%$ reduction**).
- **Epistemic vs Aleatoric Uncertainty**: 30 Monte Carlo Dropout stochastic passes successfully isolate model ignorance from intrinsic photometric noise at difficult decision boundaries.

---

## 🔌 8. API Reference & Microservice Suite

The FastAPI service exposes comprehensive RESTful endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, CUDA status, and active compute device. |
| `GET` | `/status` | Server uptime and background worker queue metrics. |
| `GET` | `/api/targets` | List of predefined catalog benchmarks and synthetic curves. |
| `GET` | `/api/moderation-stats` | DeepSeek cache hit counter, live API calls, and token savings. |
| `POST` | `/api/fits/inspect` | Inspect FITS header metadata, pixel statistics, and render image preview. |
| `POST` | `/api/predict` | Web endpoint for model inference, agent verification, XAI attributions, and DeepSeek reports. |
| `POST` | `/classify` | Fast neural classification forward pass returning probabilities and OOD energy score. |
| `POST` | `/cross-match` | Direct multi-catalog querying across SIMBAD, NED, and NASA ADS. |
| `POST` | `/analyze` | End-to-end pipeline execution (synchronous or background asynchronous execution). |
| `GET` | `/tasks/{task_id}` | Retrieve status and output for asynchronous background analysis tasks. |

### Turn-Based Rate Limiting
To ensure fair resource allocation and avoid quota exhaustion:
- Configured via `RATE_LIMIT_PER_TURN` (default: `50` requests) over a sliding window (`RATE_LIMIT_WINDOW_SECONDS = 60.0s`).
- Pass the `X-Turn-ID` header in client requests to isolate rate limit quotas across sessions.

---

## 📂 9. Repository Structure

```
AstroFoundation/
├── config/
│   ├── agent_config.yaml         # Agent thresholds, catalog timeouts, and LLM config
│   ├── feedback_config.yaml      # Active learning and human-in-the-loop settings
│   └── model_config.yaml         # Model layer dimensions, heads, and parameters
├── data/
│   ├── sample_images/            # Ready-to-use sample .fits and .png images
│   ├── processed/                # Preprocessed multi-modal tensors
│   ├── raw/                      # Downloaded telescope survey files
│   └── splits/                   # Train / Validation / Test index splits
├── frontend/                     # React 19 + Vite Interactive Web SPA
│   ├── src/
│   │   ├── App.jsx               # Main React dashboard & state management
│   │   ├── App.css               # Modern glassmorphic theme & layout styling
│   │   └── main.jsx              # React DOM root entrypoint
│   ├── dist/                     # Compiled static production bundle
│   ├── index.html                # Application HTML shell
│   └── vite.config.js            # Vite build and dev server config
├── notebooks/                    # Jupyter demonstration and ablation notebooks
├── scripts/
│   ├── download_sample_images.py # Downloader & cutout generator for sample images
│   ├── download_real_sdss_spectra.py # SDSS DR17 spectra downloader
│   ├── train_encoders.py         # Unimodal encoder pretraining
│   ├── train_fusion.py           # Cross-attention fusion training
│   └── run_phase7_benchmarks.py  # Model benchmark and evaluation runner
├── src/
│   ├── agent/
│   │   ├── agent.py              # LangGraph ReAct Agent workflow
│   │   ├── schemas.py            # Pydantic schemas for agent state & tools
│   │   └── tools.py              # SIMBAD, NED, and NASA ADS API client tools
│   ├── api/
│   │   ├── main.py               # FastAPI application setup & static mounting
│   │   ├── routes.py             # API route endpoints & pipeline controllers
│   │   ├── rate_limiter.py       # Turn-based sliding window rate limiter
│   │   └── background_tasks.py   # Asynchronous job queue manager
│   ├── data/
│   │   ├── ingestion.py          # FITS, light curve, and spectra loading
│   │   ├── preprocessing.py      # Sigma-clipping, arcsinh scaling, normalizations
│   │   ├── dataset.py            # PyTorch Dataset definitions
│   │   └── dataloaders.py        # Multi-modal batch collators and loaders
│   ├── evaluation/
│   │   ├── explainability.py     # 1D/2D Grad-CAM, SmoothGrad, SHAP, and LineIdentifier
│   │   ├── metrics.py            # Macro F1, ECE, Brier score calculations
│   │   └── benchmarks.py         # Comparative benchmark suites
│   ├── models/
│   │   ├── transformer_encoder.py# 1D Pre-LN Light Curve Transformer
│   │   ├── image_encoder.py      # Adapted ResNet-34 FITS Image Encoder
│   │   ├── spectral_encoder.py   # 1D Conv-MLP Optical Spectra Encoder
│   │   ├── projection_heads.py   # Unit hypersphere projection heads
│   │   ├── fusion.py             # Cross-attention fusion & learned gating
│   │   ├── task_heads.py         # Focal loss classifier & energy OOD head
│   │   └── full_model.py         # Unified AstroFoundationModel wrapper
│   ├── reporting/
│   │   ├── llm_reporter.py       # DeepSeek / Gemini / OpenAI report generator
│   │   ├── report_schema.py      # Pydantic schema with ATel and physical parameters
│   │   └── prompts.py            # Astrophysicist system prompts & templates
│   └── training/
│       ├── trainer.py            # Multi-modal training loop
│       ├── calibration.py        # Post-hoc Temperature Scaling
│       ├── uncertainty.py        # Monte Carlo Dropout UQ engine
│       └── active_learning.py    # Active learning acquisition functions
├── tests/                        # Full test suite (44 unit & integration tests)
├── .env.example                  # Environment configuration template
├── pyproject.toml                # Project build metadata
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

---

## ⚡ 10. Quickstart Guide

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ and `npm` (for the frontend web application)
- **Git**

### 2. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/RishiChalana/AstroFoundation.git
cd AstroFoundation

# Create and activate Python virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy example environment configuration
cp .env.example .env
```

Edit `.env` to supply your API credentials:
```env
# LLM Provider: "deepseek" (recommended), "gemini", or "openai"
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# NASA ADS API Token (optional, for literature cross-matching)
ADS_DEV_KEY=your_nasa_ads_token_here

# Turn-Based Rate Limit
RATE_LIMIT_PER_TURN=50
```

### 4. Run the Test Suite

```bash
# Run all unit and integration tests
pytest tests/ -v
```

### 5. Launch the FastAPI Backend Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Alternative ReDoc UI**: `http://localhost:8000/redoc`

### 6. Launch the React Web Dashboard

To run the frontend in development mode with Hot Module Replacement (HMR):
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

> **Note:** The FastAPI server automatically serves the precompiled production bundle from `frontend/dist` directly at `http://localhost:8000/` if accessed through the backend port! To rebuild the bundle at any time:
> ```bash
> cd frontend && npm run build
> ```

### 7. Analyze a Target via cURL

```bash
# Analyze Kepler-10 using the full multi-modal pipeline:
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "star_id": "Kepler-10",
    "kic_id": 11904151,
    "include_fits": true,
    "include_agent": true,
    "include_report": true
  }'
```

---

## 🌌 11. Sample Datasets & Downloader

A set of curated sample observations representing all 6 archetypal event classes is included in [`data/sample_images/`](data/sample_images/):
- `kepler_10_transit.fits` & `.png` (Exoplanet transit)
- `beta_lyrae_binary.fits` & `.png` (Eclipsing binary)
- `delta_cephei_pulsator.fits` & `.png` (Pulsating variable star)
- `sn2011fe_supernova.fits` & `.png` (Supernova explosion)
- `ss_cygni_cataclysmic.fits` & `.png` (Cataclysmic variable)
- `ngc4993_grb_kilonova.fits` & `.png` (Short GRB / kilonova)

To regenerate or fetch additional sample cutouts from NASA SkyView and Lightkurve:
```bash
python scripts/download_sample_images.py
```

---

## 👥 12. Team Contributions

| Team Member | Core Focus & Subsystem Ownership |
|---|---|
| **Krissh** | **Model Architecture, Autonomous Agent & Frontend UI**<br>• Core foundation models (`src/models/`): 1D Transformer, Adapted ResNet-34, 1D Conv-MLP, Projection Heads, Cross-Attention Fusion & Task Heads.<br>• LangGraph ReAct Agent (`src/agent/agent.py`, `src/agent/schemas.py`).<br>• React 19 + Vite Interactive Web SPA (`frontend/src/App.jsx`, `frontend/src/App.css`).<br>• Test coverage: `tests/test_models.py`, `tests/test_fusion.py`, `tests/test_agent.py`. |
| **Yash** | **Data Pipelines, Ingestion & Uncertainty Estimation**<br>• Data ingestion and preprocessing (`src/data/ingestion.py`, `src/data/preprocessing.py`, `src/data/dataset.py`, `src/data/dataloaders.py`).<br>• Bayesian Monte Carlo Dropout uncertainty estimation (`src/training/uncertainty.py`).<br>• Post-hoc Temperature Scaling calibration (`src/training/calibration.py`).<br>• Active learning acquisition strategies (`src/training/active_learning.py`).<br>• Test coverage: `tests/test_datasets.py`, `tests/test_uncertainty.py`. |
| **Rishi** | **Training Infrastructure, Evaluation, LLM Reporting & API**<br>• Distributed training loops & schedulers (`src/training/trainer.py`, `src/training/losses.py`, `src/training/schedulers.py`).<br>• Performance metrics & explainability (`src/evaluation/metrics.py`, `src/evaluation/explainability.py`).<br>• DeepSeek / Gemini scientific report generator (`src/reporting/llm_reporter.py`, `src/reporting/report_schema.py`).<br>• FastAPI production service & rate limiter (`src/api/main.py`, `src/api/routes.py`, `src/api/rate_limiter.py`).<br>• Test coverage: `tests/test_api.py`, `tests/test_explainability.py`. |

---

## 📜 13. License & Acknowledgments

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Acknowledgments & Data Sources
- **NASA Kepler, K2 & TESS Missions**: NASA Exoplanet Archive and Mikulski Archive for Space Telescopes (MAST).
- **Sloan Digital Sky Survey (SDSS DR17)**: Optical stellar spectroscopy and imaging cutouts.
- **CDS Strasbourg & NASA/IPAC**: Astronomical cross-matching powered by SIMBAD, NED, and NASA ADS.
- **DeepSeek AI**: Reasoning and scientific reporting synthesis via DeepSeek-V3 and DeepSeek-R1.
