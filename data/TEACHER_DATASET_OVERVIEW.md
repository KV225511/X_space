# 🌌 AstroFoundation: Multi-Modal Astronomical Dataset Overview

> **Prepared for Academic Project Evaluation & Teacher Demonstration**  
> **Project:** AstroFoundation (`X_space`) · Multi-Modal Foundation Model for Deep Space Discovery  
> **Storage Footprint:** **~578.46 MB (0.565 GB)** — *strictly below the 5 GB quota constraint (available disk space: ~142 GB)*.

---

## 📊 1. Dataset Scale & Storage Verification

| Modality / Component | Format | Count | Disk Size | Location |
|---|:---:|:---:|:---:|---|
| **Raw Kepler Light Curves** | `.fits` | 11 targets | ~33.5 MB | `data/raw/light_curves/mastDownload/` |
| **Raw Target Pixel Files (TPFs)** | `.fits.gz` | 12 targets | ~383.6 MB | `data/raw/fits/mastDownload/` |
| **Raw SDSS DR17 Calibrated Spectra** | `.fits` | 593 spectra | ~120.2 MB | `data/raw/spectra/` |
| **NASA SkyView 2D Cutouts & Previews**| `.fits` + `.png` | 12 files | ~1.5 MB | `data/sample_images/` |
| **Master Processed Light Curves** | `.npz` | 2,000 curves (8192-dim) | ~60.7 MB | `data/processed/light_curves_2000.npz` |
| **Master Processed Spectra** | `.npz` | 593 + 2,000 (4096-dim) | ~32.9 MB | `data/processed/spectra_*.npz` |
| **Catalogs & Splits** | `.csv` | 5 files | ~0.2 MB | `data/processed/` & `data/splits/` |
| **TOTAL DATASET SIZE** | — | **637 files** | **~578.46 MB (~0.565 GB)** | `data/` |

---

## 🪐 2. Event Classes & Multi-Modal Mapping

AstroFoundation unifies **3 complementary observational modalities** across **6 core astrophysical event classes**:

```
                       NASA Telescope Observations (Kepler / TESS / SDSS)
                                                │
                ┌───────────────────────────────┼───────────────────────────────┐
                ▼                               ▼                               ▼
        Light Curve (8192,)            FITS Image (3, 224, 224)         Spectrum (4096,)
      [Time-Series Photometry]             [2D Spatial PSF]             [1D Optical Flux]
```

### Classification Breakdown

| Class ID | Event Class Name | Astronomical Physical Origin | Distinct Observational Signature | Exemplar Targets Downloaded |
|:---:|---|---|---|---|
| **0** | `exoplanet_transit` | Exoplanet transiting stellar disk | Periodic, shallow, flat-bottomed U-shape dip ($\approx 0.01\% - 1\%$) | **Kepler-10**, **Kepler-90**, **Kepler-22** |
| **1** | `eclipsing_binary` | Mutual occultation of orbiting star pair | Two alternating unequal dips (primary & secondary eclipse) | **KIC 5597970**, **KIC 9002278**, **KIC 3241344** |
| **2** | `variable_star` | Pulsating stellar envelope ($\kappa$-mechanism) | Periodic sinusoidal or saw-tooth brightness variations | **RR Lyrae** (KIC 7199397), **KIC 7548479** (Delta Scuti) |
| **3** | `supernova` | Core-collapse or thermonuclear runaway | Sudden rapid rise (shock breakout) followed by power-law decay | **KSN 2011b** (KIC 8210018), **SN 2011fe** |
| **4** | `cataclysmic_variable` | Mass transfer onto white dwarf accretion disk | Repeated stochastic accretion flares returning to baseline | **V344 Lyr** (KIC 7659570), **V434 Lyr** (KIC 11412044) |
| **5** | `gamma_ray_burst` / QSO | Relativistic jet from compact merger / blazar | Ultra-brief gamma spike + power-law continuum synchrotron flux | **NGC 4993** / High-z QSO |

---

## 📁 3. Which File Belongs to Which Category

### A. Raw Kepler Time-Series Light Curves (`data/raw/light_curves/`)
* **Class 0 (`exoplanet_transit`)**:
  - `kplr011904151-*.fits`: **Kepler-10** (KIC 11904151) — Prototypical rocky transiting exoplanet.
  - `kplr011442793-*.fits`: **Kepler-90** (KIC 11442793) — 8-planet multi-transiting system.
  - `kplr010593626-*.fits`: **Kepler-22** (KIC 10593626) — Habitable-zone transiting exoplanet.
* **Class 1 (`eclipsing_binary`)**:
  - `kplr005597970-*.fits`: **KIC 5597970** — Close contact eclipsing binary.
  - `kplr009002278-*.fits`: **KIC 9002278** — Detached eclipsing binary.
  - `kplr003241344-*.fits`: **KIC 3241344** — Algol-type eclipsing binary.
* **Class 2 (`variable_star`)**:
  - `kplr007199397-*.fits`: **RR Lyrae** (KIC 7199397) — Pulsating horizontal branch giant star.
  - `kplr007548479-*.fits`: **KIC 7548479** — $\delta$ Scuti pulsating variable star.
* **Class 3 (`supernova`)**:
  - `kplr008210018-*.fits`: **KSN 2011b** (KIC 8210018) — Kepler Supernova shock breakout transient.
* **Class 4 (`cataclysmic_variable`)**:
  - `kplr007659570-*.fits`: **V344 Lyr** (KIC 7659570) — Dwarf nova with recurrent accretion disk outbursts.
  - `kplr011412044-*.fits`: **V434 Lyr** (KIC 11412044) — Rapidly flaring cataclysmic variable.

### B. Raw Kepler Target Pixel Files (TPFs) (`data/raw/fits/`)
* **Class 0**: `kplr011904151-*` (Kepler-10), `kplr011442793-*` (Kepler-90), `kplr010593626-*` (Kepler-22).
* **Class 1**: `kplr005597970-*`, `kplr009002278-*`, `kplr003241344-*`.
* **Class 2**: `kplr007199397-*` (RR Lyrae), `kplr007548479-*` (Delta Scuti).
* **Class 3**: `kplr008210018-*` (KSN 2011b Supernova).
* **Class 4**: `kplr007659570-*` (V344 Lyr), `kplr011412044-*` (V434 Lyr).

### C. 2D Visual FITS Cutouts & Web PNGs (`data/sample_images/`)
Directly retrieved from NASA SkyView (DSS2) and ready for React 19 web visualizer:
* `kepler_10_transit.fits` & `.png`: **Class 0 (`exoplanet_transit`)**
* `beta_lyrae_binary.fits` & `.png`: **Class 1 (`eclipsing_binary`)**
* `delta_cephei_pulsator.fits` & `.png`: **Class 2 (`variable_star`)**
* `sn2011fe_supernova.fits` & `.png`: **Class 3 (`supernova`)**
* `ss_cygni_cataclysmic.fits` & `.png`: **Class 4 (`cataclysmic_variable`)**
* `ngc4993_grb_kilonova.fits` & `.png`: **Class 5 (`gamma_ray_burst`)**

### D. Real SDSS DR17 Calibrated Spectra (`data/raw/spectra/`)
593 real optical spectra ($3800\text{ \AA}$ to $9200\text{ \AA}$) from Sloan Digital Sky Survey:
* `spec_0_*.fits` (99 files): **Class 0 (`exoplanet_transit`)** — Solar-type G-dwarf host stars.
* `spec_1_*.fits` (99 files): **Class 1 (`eclipsing_binary`)** — White Dwarf / compact binary spectra.
* `spec_2_*.fits` (98 files): **Class 2 (`variable_star`)** — Cool pulsating M-giant stars (prominent TiO bands).
* `spec_3_*.fits` (100 files): **Class 3 (`supernova`)** — Starburst / star-forming host galaxies ($\text{H}\alpha$, $[\text{O III}]$ emission).
* `spec_4_*.fits` (97 files): **Class 4 (`cataclysmic_variable`)** — Cataclysmic variables with broad Balmer accretion lines.
* `spec_5_*.fits` (100 files): **Class 5 (`gamma_ray_burst` / QSO)** — High-redshift quasars / relativistic engine.

### E. Master Processed Arrays (`data/processed/`)
* `light_curves_2000.npz`: Array containing `flux` (2000, 8192), `masks` (2000, 8192), `labels` (2000,), `kic_ids` (2000,).
* `spectra_sdss_real.npz`: Array containing `spectra` (593, 4096), `labels` (593,), `wavelength` (4096,), `spec_obj_ids` (593,).
* `astro_manifest_2000.csv`: Metadata manifest mapping each target's KIC ID, label (0–5), class name, and catalog source.
* `sdss_spectra_manifest_600.csv`: SDSS metadata (Plate, MJD, FiberID, Redshift $z$, SpecObjID, SubClass).

### F. Dataset Splits (`data/splits/`)
* `train.csv`: 70% of dataset (1,400 samples)
* `val.csv`: 15% of dataset (300 samples)
* `test.csv`: 15% of dataset (300 samples)

---

## 💻 4. How to Inspect & Demonstrate to Your Teacher

```python
from astropy.io import fits
import numpy as np

# 1. Inspect raw Kepler Light Curve (e.g. Kepler-10 Exoplanet Transit)
with fits.open("data/raw/light_curves/mastDownload/Kepler/kplr011904151_sc_Q001313330333033302/kplr011904151-2009291181958_slc.fits") as hdul:
    lc_table = hdul[1].data
    print("Kepler-10 Photometry Data Points:", len(lc_table))
    print("Columns available:", lc_table.columns.names[:5])

# 2. Inspect raw Kepler Target Pixel File (2D CCD pixel array over time)
with fits.open("data/raw/fits/mastDownload/Kepler/kplr011904151_lc_Q111111110111011101/kplr011904151-2009166043257_lpd-targ.fits.gz") as hdul:
    tpf_flux = hdul[1].data["FLUX"]
    print("TPF Image Shape (Time x Height x Width):", tpf_flux.shape)

# 3. Inspect Preprocessed Multi-Modal Arrays
lc_data = np.load("data/processed/light_curves_2000.npz")
print("Processed Light Curves Tensor:", lc_data["flux"].shape)    # (2000, 8192)

spec_data = np.load("data/processed/spectra_sdss_real.npz")
print("Processed Spectra Tensor:", spec_data["spectra"].shape)     # (593, 4096)
```
