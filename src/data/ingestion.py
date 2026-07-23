import lightkurve as lk
from astropy.io import fits
from functools import lru_cache
from .preprocessing import (
    process_light_curve_v4, 
    process_fits_image_v4, 
    process_spectrum_v4
)

# =====================================================================
# 1. LIGHT CURVE INGESTION & CACHING
# =====================================================================
@lru_cache(maxsize=500)
def load_light_curve(kic_id):
    """
    Downloads raw Kepler data, executes the v4.0 preprocessing pipeline, 
    and caches the resulting (8192,) flux array and validity mask in RAM.
    """
    search_result = lk.search_lightcurve(kic_id, author='Kepler', quarter=3)
    if len(search_result) == 0:
        raise ValueError(f"No data found for KIC {kic_id}")
        
    lc = search_result.download()
    
    # Note: lc.fill_gaps() is intentionally omitted to preserve raw telemetry gaps
    raw_time = lc.time.value
    raw_flux = lc.flux.value
    
    final_flux, final_mask = process_light_curve_v4(raw_time, raw_flux, target_len=8192)
    return final_flux, final_mask


# =====================================================================
# 2. FITS IMAGE INGESTION & CACHING
# =====================================================================
@lru_cache(maxsize=500)
def load_fits_image(file_path):
    """
    Loads a local FITS image file, fixes endianness, applies arcsinh 
    stretching, and returns a (3, 224, 224) numpy array.
    """
    with fits.open(file_path) as hdul:
        # Extract primary image array from HDU 0 or HDU 1
        raw_image = hdul[0].data if hdul[0].data is not None else hdul[1].data
        raw_image = np.array(raw_image, dtype=np.float32)

    # Fix big-endian system compatibility issues if present
    if raw_image.dtype.byteorder == '>':
        raw_image = raw_image.byteswap().newbyteorder()

    processed_image = process_fits_image_v4(raw_image, target_shape=(224, 224))
    return processed_image


# =====================================================================
# 3. SPECTRAL INGESTION & CACHING
# =====================================================================
@lru_cache(maxsize=500)
def load_spectrum(file_path, redshift_z=0.0):
    """
    Loads a 1D astronomical spectrum FITS file, Doppler de-shifts to 
    rest frame, and returns a normalized (4096,) array.
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        wavelength = data['wavelength'] if 'wavelength' in data.names else data['loglam']
        flux = data['flux']

    processed_spectrum = process_spectrum_v4(
        observed_wavelength=wavelength, 
        flux=flux, 
        redshift_z=redshift_z, 
        target_len=4096
    )
    return processed_spectrum