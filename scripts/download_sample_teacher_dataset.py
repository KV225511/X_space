"""
AstroFoundation — Curated Sample Dataset Downloader for Academic Evaluation
=============================================================================
Downloads representative Kepler Light Curves and 2D Target Pixel Files (TPFs)
across the core astronomical event classes, keeping total size well under 5 GB.
"""

import sys
import os
from pathlib import Path
import lightkurve as lk

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE_TARGETS = [
    # 1. Exoplanet Transits (Class 0)
    {"id": "kepler_10", "kic": 11904151, "name": "Kepler-10", "class": "exoplanet_transit", "quarter": 3},
    {"id": "kepler_90", "kic": 11442793, "name": "Kepler-90", "class": "exoplanet_transit", "quarter": 4},
    {"id": "kepler_22", "kic": 10593626, "name": "Kepler-22", "class": "exoplanet_transit", "quarter": 3},

    # 2. Eclipsing Binaries (Class 1)
    {"id": "kic_5597970", "kic": 5597970, "name": "KIC 5597970", "class": "eclipsing_binary", "quarter": 3},
    {"id": "kic_9002278", "kic": 9002278, "name": "KIC 9002278", "class": "eclipsing_binary", "quarter": 3},
    {"id": "kic_3241344", "kic": 3241344, "name": "KIC 3241344", "class": "eclipsing_binary", "quarter": 3},

    # 3. Variable Stars (Class 2)
    {"id": "rr_lyr", "kic": 7199397, "name": "RR Lyrae", "class": "variable_star", "quarter": 3},
    {"id": "delta_scuti", "kic": 7548479, "name": "KIC 7548479", "class": "variable_star", "quarter": 3},

    # 4. Supernova Transient (Class 3)
    {"id": "ksn_2011b", "kic": 8210018, "name": "KSN 2011b (Supernova)", "class": "supernova", "quarter": 11},

    # 5. Cataclysmic Variables (Class 4)
    {"id": "v344_lyr", "kic": 7659570, "name": "V344 Lyr", "class": "cataclysmic_variable", "quarter": 3},
    {"id": "v434_lyr", "kic": 11412044, "name": "V434 Lyr", "class": "cataclysmic_variable", "quarter": 3},
]

def main():
    lc_dir = Path("data/raw/light_curves")
    fits_dir = Path("data/raw/fits")
    lc_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  🌌 AstroFoundation Curated Teacher Sample Downloader")
    print(f"  Target Classes: exoplanet_transit, eclipsing_binary, variable_star, supernova, cataclysmic_variable")
    print(f"  Total Targets to Fetch: {len(SAMPLE_TARGETS)}")
    print("=================================================================\n")

    success_count = 0
    for idx, item in enumerate(SAMPLE_TARGETS, 1):
        cname = item["class"]
        name = item["name"]
        kic = item["kic"]
        quarter = item["quarter"]

        print(f"[{idx}/{len(SAMPLE_TARGETS)}] Downloading [{cname}] -> {name} (KIC {kic}, Q{quarter})...")
        
        # 1. Download Light Curve
        try:
            lc_search = lk.search_lightcurve(f"KIC {kic}", author="Kepler", quarter=quarter)
            if len(lc_search) > 0:
                lc_search[0].download(download_dir=str(lc_dir))
                print(f"   ✓ [Light Curve] Downloaded")
            else:
                print(f"   ⚠ [Light Curve] No data found on MAST for Q{quarter}")
        except Exception as e:
            print(f"   ✗ [Light Curve] Failed: {e}")

        # 2. Download Target Pixel File
        try:
            tpf_search = lk.search_targetpixelfile(f"KIC {kic}", author="Kepler", quarter=quarter)
            if len(tpf_search) > 0:
                tpf_search[0].download(download_dir=str(fits_dir))
                print(f"   ✓ [Target Pixel File FITS] Downloaded")
                success_count += 1
            else:
                print(f"   ⚠ [Target Pixel File] No data found on MAST for Q{quarter}")
        except Exception as e:
            print(f"   ✗ [Target Pixel File] Failed: {e}")

    print(f"\n=================================================================")
    print(f"  ✓ Download Complete! {success_count}/{len(SAMPLE_TARGETS)} targets processed.")
    print("=================================================================")

if __name__ == "__main__":
    main()
