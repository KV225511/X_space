import os
import sys
from pathlib import Path
import lightkurve as lk

# Ensure UTF-8 output in Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE_TARGETS = [
    {"class": "exoplanet_transit", "target": "Kepler-10", "quarter": 3},
    {"class": "eclipsing_binary", "target": "KIC 5597970", "quarter": 3},
    {"class": "variable_star", "target": "KIC 7199397", "quarter": 3},
    {"class": "supernova", "target": "KIC 8210018", "quarter": 11},
    {"class": "cataclysmic_variable", "target": "V344 Lyr", "quarter": 3},
]

def setup_directories():
    """Ensure the raw data, processed, and split directories exist."""
    dirs = [
        "data/raw/light_curves", 
        "data/raw/fits", 
        "data/raw/spectra",
        "data/sample_images",
        "data/processed",
        "data/splits"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[OK] Data directories initialized.")

def download_sample_data():
    """Download representative Kepler light curves and FITS target pixel files."""
    for item in SAMPLE_TARGETS:
        cname = item["class"]
        tgt = item["target"]
        q = item["quarter"]
        print(f"\n--- Fetching {cname} sample: {tgt} (Quarter {q}) ---")
        try:
            lc_search = lk.search_lightcurve(tgt, author='Kepler', quarter=q)
            if len(lc_search) > 0:
                lc_search[0].download(download_dir='data/raw/light_curves')
                print(f"  [OK] Light Curve for {tgt} downloaded successfully.")
            else:
                print(f"  [!] No light curve found for {tgt}")
        except Exception as e:
            print(f"  [WARN] Light curve download failed for {tgt}: {e}")

        try:
            tpf_search = lk.search_targetpixelfile(tgt, author='Kepler', quarter=q)
            if len(tpf_search) > 0:
                tpf_search[0].download(download_dir='data/raw/fits')
                print(f"  [OK] FITS Target Pixel File for {tgt} downloaded successfully.")
            else:
                print(f"  [!] No TPF found for {tgt}")
        except Exception as e:
            print(f"  [WARN] TPF download failed for {tgt}: {e}")

if __name__ == "__main__":
    print("=== AstroFoundation Sample Dataset Downloader ===")
    setup_directories()
    download_sample_data()
    print("\n=== Data download complete. All sample categories ready! ===")