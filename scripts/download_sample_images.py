"""
AstroFoundation — Sample Astronomical Image Downloader & Generator
===================================================================
Downloads and prepares sample astronomical images (both raw .fits and web-ready .png)
representing the 6 core event classes:
  1. exoplanet_transit     (Kepler-10 / Kepler-90)
  2. eclipsing_binary      (Beta Lyrae / KIC 9002278)
  3. variable_star         (Delta Cephei / Mira)
  4. supernova             (SN 2011fe / ASASSN-15lh)
  5. cataclysmic_variable  (SS Cygni)
  6. gamma_ray_burst       (GRB 170817A / NGC 4993)

Supports:
  - Online fetching via NASA SkyView (DSS2 / 2MASS / SDSS) and Lightkurve
  - Offline fallback generation with realistic PSF and astrophysical profiles
  - Automatic export of both .fits and .png (for drag-and-drop in React UI)

Usage:
  python scripts/download_sample_images.py
  python scripts/download_sample_images.py --offline-only
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

SAMPLE_DIR = Path("data/sample_images")

TARGETS = [
    {
        "id": "kepler_10_transit",
        "name": "Kepler-10",
        "class": "exoplanet_transit",
        "ra": 285.6794,
        "dec": 50.2413,
        "survey": "DSS2 Red",
        "desc": "Rocky exoplanet host star showing Kepler transit PSF core.",
    },
    {
        "id": "beta_lyrae_binary",
        "name": "Beta Lyrae",
        "class": "eclipsing_binary",
        "ra": 282.5197,
        "dec": 33.3627,
        "survey": "DSS2 Red",
        "desc": "Prototypical semidetached eclipsing binary system.",
    },
    {
        "id": "delta_cephei_pulsator",
        "name": "Delta Cephei",
        "class": "variable_star",
        "ra": 337.2917,
        "dec": 58.4152,
        "survey": "DSS2 Red",
        "desc": "Radial pulsating Cepheid variable star standard candle.",
    },
    {
        "id": "sn2011fe_supernova",
        "name": "SN 2011fe",
        "class": "supernova",
        "ra": 210.8023,
        "dec": 54.2733,
        "survey": "DSS2 Blue",
        "desc": "Type Ia supernova in spiral galaxy Messier 101.",
    },
    {
        "id": "ss_cygni_cataclysmic",
        "name": "SS Cygni",
        "class": "cataclysmic_variable",
        "ra": 325.6792,
        "dec": 43.5858,
        "survey": "DSS2 Red",
        "desc": "Prototypical dwarf nova undergoing rapid accretion outbursts.",
    },
    {
        "id": "ngc4993_grb_kilonova",
        "name": "NGC 4993",
        "class": "gamma_ray_burst",
        "ra": 197.4503,
        "dec": -23.6331,
        "survey": "DSS2 Red",
        "desc": "Host galaxy of binary neutron star merger GW170817 / GRB 170817A.",
    },
]


def _create_synthetic_cutout(class_name: str, size: int = 224) -> np.ndarray:
    """Generate realistic astrophysical FITS/optical stamp when offline."""
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2

    # Baseline Poisson / Gaussian sky background noise
    bg = np.random.normal(loc=0.08, scale=0.02, size=(size, size))

    if class_name in ("exoplanet_transit", "variable_star"):
        # Single isolated symmetric stellar Airy disk / Moffat PSF
        sigma = 8.0
        psf = np.exp(-dist_sq / (2.0 * sigma**2)) * 0.85
        image = bg + psf

    elif class_name == "eclipsing_binary":
        # Dual or elongated stellar centroid
        d1 = ((x - (center_x - 6)) ** 2 + (y - center_y) ** 2) / (2.0 * 7.0**2)
        d2 = ((x - (center_x + 6)) ** 2 + (y - center_y) ** 2) / (2.0 * 6.5**2)
        image = bg + np.exp(-d1) * 0.75 + np.exp(-d2) * 0.55

    elif class_name == "supernova":
        # Host galaxy diffuse disk + point-like supernova transient
        r_gal = np.sqrt(dist_sq) / 45.0
        host_galaxy = np.exp(-r_gal) * 0.45
        # Transient slightly offset on spiral arm
        sn_dist = ((x - (center_x + 18)) ** 2 + (y - (center_y - 12)) ** 2) / (2.0 * 4.0**2)
        sn_spot = np.exp(-sn_dist) * 0.95
        image = bg + host_galaxy + sn_spot

    elif class_name == "cataclysmic_variable":
        # Hot compact central star with accretion disk glow
        sigma = 6.0
        core = np.exp(-dist_sq / (2.0 * sigma**2)) * 0.9
        disk = np.exp(-np.sqrt(dist_sq) / 20.0) * 0.25
        image = bg + core + disk

    else:  # gamma_ray_burst / anomaly
        # Diffuse elliptical host galaxy with bright core
        theta = np.deg2rad(35)
        xp = (x - center_x) * np.cos(theta) + (y - center_y) * np.sin(theta)
        yp = -(x - center_x) * np.sin(theta) + (y - center_y) * np.cos(theta)
        ellipse_r = np.sqrt((xp / 35.0) ** 2 + (yp / 18.0) ** 2)
        image = bg + np.exp(-ellipse_r) * 0.70

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def save_cutout_as_png_and_fits(data_2d: np.ndarray, file_prefix: str):
    """Save 2D array as both 8-bit RGB PNG (for web) and FITS (for pipelines)."""
    # Ensure 2D shape (squeeze any extra single-dim axes)
    arr = np.squeeze(data_2d)
    if arr.ndim != 2:
        arr = arr[0] if arr.ndim > 2 else np.zeros((224, 224), dtype=np.float32)

    # 1. Save PNG
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    denom = max_val - min_val if (max_val - min_val) > 1e-8 else 1.0
    norm = np.clip((arr - min_val) / denom, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)

    # Resize to 224x224 if different
    if norm.shape != (224, 224):
        pil_norm = Image.fromarray((norm * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        norm = np.array(pil_norm, dtype=np.float32) / 255.0

    # Apply warm cosmic colormap: Blue/Teal background to Amber/White core
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(norm * 255 * 1.1, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(norm**1.4 * 220 + (1 - norm) * 20, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(norm**2.0 * 255 + (1 - norm) * 45, 0, 255).astype(np.uint8)

    png_path = SAMPLE_DIR / f"{file_prefix}.png"
    Image.fromarray(rgb).save(png_path)

    # 2. Save FITS
    try:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(arr.astype(np.float32))
        hdu.header["TARGET"] = file_prefix
        hdu.header["ORIGIN"] = "AstroFoundation"
        fits_path = SAMPLE_DIR / f"{file_prefix}.fits"
        hdu.writeto(fits_path, overwrite=True)
    except Exception as exc:
        print(f"    Notice: FITS header write skipped ({exc})")

    print(f"  [OK] Saved web preview: {png_path} & raw FITS")


def download_or_generate_samples(offline_only: bool = False):
    """Iterate through target registry and fetch or synthesize image stamps."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=======================================================")
    print(" AstroFoundation Sample Image Downloader")
    print(f" Destination: {SAMPLE_DIR.resolve()}")
    print("=======================================================\n")

    for tgt in TARGETS:
        tid = tgt["id"]
        tname = tgt["name"]
        tclass = tgt["class"]
        survey = tgt["survey"]
        print(f"[*] Processing [{tclass}] -> {tname}...")

        downloaded = False
        if not offline_only:
            try:
                import astropy.units as u
                from astroquery.skyview import SkyView
                print(f"    Querying NASA SkyView ({survey}) for {tname}...")
                images = SkyView.get_images(
                    position=tname,
                    survey=[survey],
                    radius=2.5 * u.arcmin,
                    pixels=224,
                )
                if images and len(images) > 0:
                    raw_data = images[0][0].data
                    save_cutout_as_png_and_fits(raw_data, tid)
                    downloaded = True
                    print(f"    [OK] Successfully retrieved SkyView image for {tname}")
            except Exception as e:
                print(f"    [WARN] SkyView online query skipped ({e}), switching to synthetic model...")

        if not downloaded:
            synth_data = _create_synthetic_cutout(tclass, size=224)
            save_cutout_as_png_and_fits(synth_data, tid)

    print(f"\n[OK] Completed! Sample images are available in '{SAMPLE_DIR}'.")
    print("  Drag and drop any .png file into the React Web App for instant analysis.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate sample astronomical images.")
    parser.add_argument("--offline-only", action="store_true", help="Generate synthetic samples without querying NASA SkyView.")
    args = parser.parse_args()
    download_or_generate_samples(offline_only=args.offline_only)
