import os
from pathlib import Path
import lightkurve as lk

def setup_directories():
    """Ensure the DVC-tracked raw data directories exist."""
    dirs = [
        "data/raw/light_curves", 
        "data/raw/fits", 
        "data/raw/spectra",
        "data/processed",
        "data/splits"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Data directories initialized.")

def download_sample_data():
    """Download a sample Kepler-10 light curve and FITS image."""
    print("\nSearching for Kepler-10 Light Curve...")
    lc_search = lk.search_lightcurve('Kepler-10', author='Kepler', quarter=3)
    if len(lc_search) > 0:
        lc_search.download(download_dir='data/raw/light_curves')
        print("✓ Light Curve downloaded successfully.")
    
    print("\nSearching for Kepler-10 FITS Target Pixel File...")
    tpf_search = lk.search_targetpixelfile('Kepler-10', author='Kepler', quarter=1)
    if len(tpf_search) > 0:
        tpf_search.download(download_dir='data/raw/fits')
        print("✓ FITS image downloaded successfully.")

if __name__ == "__main__":
    print("--- Starting X_space Data Download ---")
    setup_directories()
    download_sample_data()
    print("\n--- Data download complete. Ready for Training! ---")