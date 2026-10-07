"""
AstroFoundation — NASA / SDSS DR17 Real Spectra Downloader & Processor
======================================================================
Downloads 600 genuine telescope spectra from the Sloan Digital Sky Survey
(Data Release 17) across the 6 AstroFoundation event classes (100 per class),
preprocesses them with rest-frame Doppler de-shifting and continuum normalization,
and packages them into data/processed/spectra_sdss_600.npz.

Classes:
  0 = Solar-type dwarf host star (G/K dwarf)        [exoplanet_transit proxy]
  1 = White Dwarf / Compact binary star (WD)        [eclipsing_binary proxy]
  2 = Variable cool pulsating giant (M-star)        [variable_star proxy]
  3 = Starburst / Star-forming host galaxy          [supernova proxy]
  4 = Cataclysmic variable / Accretion disk (CV)     [cataclysmic_variable proxy]
  5 = High-redshift power-law quasar / engine (QSO)  [gamma_ray_burst proxy]
"""

import os
import io
import time
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astroquery.sdss import SDSS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Local preprocessing
from src.data.preprocessing import process_spectrum_v4

RAW_SPECTRA_DIR = Path("data/raw/spectra")
PROCESSED_DIR   = Path("data/processed")
MANIFEST_OUT    = PROCESSED_DIR / "sdss_spectra_manifest_600.csv"
NPZ_OUT         = PROCESSED_DIR / "spectra_sdss_600.npz"

TARGETS_PER_CLASS = 100
MAX_DOWNLOAD_WORKERS = 4
REQUEST_TIMEOUT = 15

CLASS_DEFINITIONS = [
    {
        "class_id": 0,
        "class_name": "exoplanet_transit",
        "description": "Solar / G-dwarf stellar host",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'STAR' AND subClass LIKE 'G%' AND zwarning = 0"""
    },
    {
        "class_id": 1,
        "class_name": "eclipsing_binary",
        "description": "White dwarf / compact binary",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'STAR' AND subClass LIKE 'WD%' AND zwarning = 0"""
    },
    {
        "class_id": 2,
        "class_name": "variable_star",
        "description": "Cool pulsating giant (M-star with TiO)",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'STAR' AND subClass LIKE 'M%' AND zwarning = 0"""
    },
    {
        "class_id": 3,
        "class_name": "supernova",
        "description": "Starburst / Star-forming host galaxy",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'GALAXY' AND subClass LIKE '%STARFORMING%' AND zwarning = 0"""
    },
    {
        "class_id": 4,
        "class_name": "cataclysmic_variable",
        "description": "Cataclysmic variable / Accretion disk",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'STAR' AND subClass LIKE '%CV%' AND zwarning = 0"""
    },
    {
        "class_id": 5,
        "class_name": "gamma_ray_burst",
        "description": "High-z QSO synchrotron / power-law engine",
        "sql": """SELECT TOP 100 plate, mjd, fiberID, class, subClass, z, specObjID, run2d
                  FROM SpecObj 
                  WHERE class = 'QSO' AND z > 0.5 AND zwarning = 0"""
    }
]


def fetch_sdss_manifest() -> pd.DataFrame:
    """Queries SDSS DR17 CasJobs for 100 targets per class."""
    print(f"\n[1/4] Querying SDSS DR17 for {TARGETS_PER_CLASS} targets per class...")
    records = []
    
    for cdef in CLASS_DEFINITIONS:
        cid = cdef['class_id']
        cname = cdef['class_name']
        print(f"  Querying Class {cid} ({cname}: {cdef['description']})...")
        res = SDSS.query_sql(cdef['sql'])
        if res is None or len(res) == 0:
            raise RuntimeError(f"SDSS query returned 0 rows for class {cid} ({cname})")
        
        for row in res:
            records.append({
                'class_id': cid,
                'class_name': cname,
                'plate': int(row['plate']),
                'mjd': int(row['mjd']),
                'fiberID': int(row['fiberID']),
                'subClass': str(row['subClass']),
                'z': float(row['z']),
                'specObjID': int(row['specObjID']),
                'run2d': str(row['run2d'])
            })
        print(f"    -> Retrieved {len(res)} targets.")

    df = pd.DataFrame(records)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f"  Saved master SDSS manifest to {MANIFEST_OUT} ({len(df)} total targets)")
    return df


def download_single_spectrum(row: pd.Series) -> tuple[int, bytes | None]:
    """Downloads one spec-lite FITS file from dr17.sdss.org."""
    plate = int(row['plate'])
    mjd = int(row['mjd'])
    fiber = int(row['fiberID'])
    spec_id = int(row['specObjID'])
    cid = int(row['class_id'])
    
    local_path = RAW_SPECTRA_DIR / f"spec_{cid}_{spec_id}.fits"
    if local_path.exists() and local_path.stat().st_size > 10000:
        with open(local_path, "rb") as f:
            return spec_id, f.read()

    url = f"https://dr17.sdss.org/optical/spectrum/view/data/format=fits?plateid={plate}&mjd={mjd}&fiberid={fiber}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AstroFoundation/4.0'})
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = resp.read()
                if len(data) > 10000:
                    with open(local_path, "wb") as f:
                        f.write(data)
                    return spec_id, data
        except Exception:
            time.sleep(1.0 * (attempt + 1))
            
    return spec_id, None


def download_all_spectra(df: pd.DataFrame) -> dict[int, bytes]:
    """Downloads all 600 spectra concurrently using a controlled thread pool."""
    print(f"\n[2/4] Downloading {len(df)} real FITS spectra from dr17.sdss.org...")
    RAW_SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = {}
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_single_spectrum, row): row['specObjID'] for _, row in df.iterrows()}
        
        pbar = tqdm(total=len(df), desc="Downloading FITS")
        for fut in as_completed(futures):
            spec_id, data = fut.result()
            if data is not None:
                downloaded[spec_id] = data
            pbar.update(1)
        pbar.close()

    print(f"  Successfully downloaded/verified {len(downloaded)}/{len(df)} spectra.")
    return downloaded


def process_and_package(df: pd.DataFrame, downloaded: dict[int, bytes]):
    """Processes downloaded FITS files through process_spectrum_v4 and saves NPZ."""
    print(f"\n[3/4] Processing spectra (rest-frame de-shifting, 4096-grid interp, continuum norm)...")
    
    valid_rows = df[df['specObjID'].isin(downloaded.keys())].reset_index(drop=True)
    n_samples = len(valid_rows)
    
    spectra_array = np.zeros((n_samples, 4096), dtype=np.float32)
    labels_array  = np.zeros(n_samples, dtype=np.int64)
    ids_array     = np.zeros(n_samples, dtype=np.uint64)
    
    grid = np.linspace(3800.0, 9200.0, 4096, dtype=np.float32)
    
    for i, row in tqdm(valid_rows.iterrows(), total=n_samples, desc="Processing"):
        spec_id = row['specObjID']
        data = downloaded[spec_id]
        
        with fits.open(io.BytesIO(data)) as hdul:
            coadd = hdul[1].data
            flux = coadd['flux']
            loglam = coadd['loglam']
            wavelength = 10.0 ** loglam
            
        z = float(row['z'])
        norm_flux = process_spectrum_v4(observed_wavelength=wavelength, flux=flux, redshift_z=z, target_len=4096)
        
        # Clean extreme sky line residuals / cosmic spikes
        norm_flux = np.nan_to_num(norm_flux, nan=1.0, posinf=5.0, neginf=0.0)
        norm_flux = np.clip(norm_flux, 0.01, 10.0)
        
        spectra_array[i] = norm_flux
        labels_array[i]  = int(row['class_id'])
        ids_array[i]     = int(spec_id)
        
    print(f"\n[4/4] Packaging real SDSS dataset to {NPZ_OUT}...")
    np.savez_compressed(
        NPZ_OUT,
        spectra=spectra_array,
        labels=labels_array,
        spec_obj_ids=ids_array,
        wavelength=grid,
        class_names=np.array([c['class_name'] for c in CLASS_DEFINITIONS])
    )
    
    size_mb = os.path.getsize(NPZ_OUT) / (1024 * 1024)
    print(f"  Successfully built {NPZ_OUT} ({size_mb:.2f} MB)")
    print(f"  Spectra tensor shape: {spectra_array.shape}")
    print(f"  Labels tensor shape:  {labels_array.shape}")
    print(f"  Class distribution:\n{pd.Series(labels_array).value_counts().sort_index()}")
    print("\nDataset ready for training!")


def main():
    print("=" * 65)
    print("  AstroFoundation — NASA / SDSS Real Spectroscopic Pipeline")
    print("=" * 65)
    
    manifest_df = fetch_sdss_manifest()
    downloaded_data = download_all_spectra(manifest_df)
    process_and_package(manifest_df, downloaded_data)


if __name__ == "__main__":
    main()
