"""
AstroFoundation — Build NPZ from Downloaded Real SDSS FITS Files
================================================================
Scans data/raw/spectra/ for downloaded FITS spectra, processes them
via process_spectrum_v4, and saves data/processed/spectra_sdss_real.npz.
Can be executed at any time to build a valid dataset from whatever
files have completed downloading.
"""

import io
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import process_spectrum_v4

RAW_DIR      = Path("data/raw/spectra")
MANIFEST     = Path("data/processed/sdss_spectra_manifest_600.csv")
OUTPUT_NPZ   = Path("data/processed/spectra_sdss_real.npz")
GRID_LEN     = 4096


def main():
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        return

    df_manifest = pd.read_csv(MANIFEST)
    manifest_map = {int(row['specObjID']): row for _, row in df_manifest.iterrows()}

    fits_files = sorted(list(RAW_DIR.glob("*.fits")))
    print(f"Found {len(fits_files)} FITS files in {RAW_DIR}")

    if len(fits_files) == 0:
        print("No FITS files found to process.")
        return

    valid_spectra = []
    valid_labels  = []
    valid_ids     = []
    class_names   = [
        "exoplanet_transit",
        "eclipsing_binary",
        "variable_star",
        "supernova",
        "cataclysmic_variable",
        "gamma_ray_burst"
    ]

    for p in tqdm(fits_files, desc="Processing FITS"):
        try:
            # Filename format: spec_{cid}_{specObjID}.fits
            spec_id = int(p.stem.split('_')[-1])
            if spec_id not in manifest_map:
                continue

            row = manifest_map[spec_id]
            z = float(row['z'])
            cid = int(row['class_id'])

            with fits.open(p) as hdul:
                coadd = hdul[1].data
                flux = coadd['flux']
                loglam = coadd['loglam']
                wavelength = 10.0 ** loglam

            norm_flux = process_spectrum_v4(observed_wavelength=wavelength, flux=flux, redshift_z=z, target_len=GRID_LEN)
            norm_flux = np.nan_to_num(norm_flux, nan=1.0, posinf=5.0, neginf=0.0)
            norm_flux = np.clip(norm_flux, 0.01, 10.0)

            valid_spectra.append(norm_flux)
            valid_labels.append(cid)
            valid_ids.append(spec_id)
        except Exception as e:
            continue

    if len(valid_spectra) == 0:
        print("No valid spectra processed.")
        return

    spectra_array = np.array(valid_spectra, dtype=np.float32)
    labels_array  = np.array(valid_labels,  dtype=np.int64)
    ids_array     = np.array(valid_ids,     dtype=np.uint64)
    grid          = np.linspace(3800.0, 9200.0, GRID_LEN, dtype=np.float32)

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        spectra=spectra_array,
        labels=labels_array,
        spec_obj_ids=ids_array,
        wavelength=grid,
        class_names=np.array(class_names)
    )

    size_mb = os.path.getsize(OUTPUT_NPZ) / (1024 * 1024)
    print(f"\nSuccessfully created {OUTPUT_NPZ} ({size_mb:.2f} MB)")
    print(f"  Total samples: {len(labels_array)}")
    print(f"  Class breakdown:")
    for cid in range(6):
        cnt = (labels_array == cid).sum()
        print(f"    Class {cid} ({class_names[cid]}): {cnt}")


if __name__ == "__main__":
    main()
