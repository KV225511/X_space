"""
AstroFoundation — Real 2D FITS Image Dataset Builder
=====================================================
Extracts, stretches, and standardizes real telescope spatial images from:
  1. Real Kepler Target Pixel Files (TPFs) in data/raw/fits/mastDownload/Kepler/
  2. Real NASA SkyView / DSS2 FITS cutouts in data/sample_images/

Outputs:
  data/processed/images_real.npz:
    - 'images': shape (N, 3, 224, 224) float32 (arcsinh-stretched, standardized)
    - 'labels': shape (N,) int64 class indices (0 to 5)
    - 'target_names': list of target identifiers
"""

import sys
import glob
from pathlib import Path
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import process_fits_image_v4

# Mapping from Kepler KIC ID to class label
KIC_TO_CLASS = {
    # Class 0: exoplanet_transit
    "11904151": 0, "011904151": 0, # Kepler-10
    "11442793": 0, "011442793": 0, # Kepler-90
    "10593626": 0, "010593626": 0, # Kepler-22
    # Class 1: eclipsing_binary
    "3241344":  1, "003241344": 1,
    "5597970":  1, "005597970": 1,
    "9002278":  1, "009002278": 1,
    # Class 2: variable_star
    "7199397":  2, "007199397": 2, # RR Lyrae
    "7548479":  2, "007548479": 2, # Delta Scuti
    # Class 3: supernova
    "8210018":  3, "008210018": 3, # KSN 2011b
    # Class 4: cataclysmic_variable
    "7659570":  3, "007659570": 4, # V344 Lyr
    "11412044": 4, "011412044": 4, # V434 Lyr
}

# SkyView 2D FITS cutouts
SKYVIEW_CUTOUTS = [
    ("data/sample_images/kepler_10_transit.fits",   0, "Kepler-10"),
    ("data/sample_images/beta_lyrae_binary.fits",   1, "Beta Lyrae"),
    ("data/sample_images/delta_cephei_pulsator.fits",2, "Delta Cephei"),
    ("data/sample_images/sn2011fe_supernova.fits",  3, "SN 2011fe"),
    ("data/sample_images/ss_cygni_cataclysmic.fits",4, "SS Cygni"),
    ("data/sample_images/ngc4993_grb_kilonova.fits", 5, "NGC 4993"),
]


def build_real_images_dataset(output_path: str = "data/processed/images_real.npz", samples_per_target: int = 175):
    print("=" * 65)
    print("  AstroFoundation — Processing Real Telescope FITS Images (Expanded 2000+)")
    print("=" * 65)

    all_images = []
    all_labels = []
    all_names  = []

    # 1. Process SkyView 2D Cutouts (high-resolution spatial deep-sky cutouts)
    print("\n[1/2] Processing NASA SkyView 2D FITS Cutouts...")
    for fpath, label, name in SKYVIEW_CUTOUTS:
        p = Path(fpath)
        if not p.exists():
            print(f"  Warning: {fpath} not found, skipping.")
            continue
        with fits.open(p) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    data = np.array(hdu.data, dtype=np.float32)
                    break
            if data is not None:
                tensor_3ch = process_fits_image_v4(data, target_shape=(224, 224))
                # Add full D4 group dihedral symmetries (rotations and flips)
                for rot in range(4):
                    for flip in [False, True]:
                        aug_img = np.rot90(tensor_3ch, k=rot, axes=(1, 2)).copy()
                        if flip:
                            aug_img = np.flip(aug_img, axis=2).copy()
                        all_images.append(aug_img)
                        all_labels.append(label)
                        all_names.append(f"{name}_rot{rot*90}_flip{flip}")
                print(f"  Processed {name} (Class {label}): generated 8 dihedral representations.")

    # 2. Process Kepler Target Pixel Files (TPFs)
    print("\n[2/2] Processing Real Kepler Target Pixel Files (TPFs)...")
    tpf_pattern = "data/raw/fits/mastDownload/Kepler/*/*.fits*"
    tpf_files = glob.glob(tpf_pattern)
    print(f"  Found {len(tpf_files)} Kepler TPF archives.")

    for fpath in tpf_files:
        kic_id = None
        for k in KIC_TO_CLASS:
            if k in fpath:
                kic_id = k
                break
        if kic_id is None:
            continue

        label = KIC_TO_CLASS[kic_id]

        with fits.open(fpath) as hdul:
            if len(hdul) > 1 and hasattr(hdul[1], "data") and "FLUX" in hdul[1].data.names:
                flux_cube = hdul[1].data["FLUX"]  # (N_cadences, H, W)
                valid_mask = ~np.isnan(flux_cube).all(axis=(1, 2))
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    continue

                # Sample cadences evenly across the observation window
                step = max(1, len(valid_indices) // samples_per_target)
                selected_cadences = valid_indices[::step][:samples_per_target]

                for cad_idx in selected_cadences:
                    frame_2d = np.nan_to_num(flux_cube[cad_idx], nan=0.0).astype(np.float32)
                    if frame_2d.max() > frame_2d.min():
                        tensor_3ch = process_fits_image_v4(frame_2d, target_shape=(224, 224))
                        all_images.append(tensor_3ch)
                        all_labels.append(label)
                        all_names.append(f"KIC_{kic_id}_cad{cad_idx}")

                print(f"  Processed KIC {kic_id} (Class {label}): extracted {len(selected_cadences)} real cadences.")

    images_arr = np.array(all_images, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=images_arr,
        labels=labels_arr,
        target_names=np.array(all_names, dtype=str)
    )

    print("\n" + "=" * 65)
    print(f"  Dataset successfully compiled: {out.resolve()}")
    print(f"  Total Real FITS Images: {len(images_arr)} images")
    print(f"  Shape:                 {images_arr.shape}")
    print(f"  Class Distribution:    {np.bincount(labels_arr, minlength=6).tolist()}")
    print("=" * 65)


if __name__ == "__main__":
    build_real_images_dataset()
