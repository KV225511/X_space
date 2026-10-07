import lightkurve as lk
import numpy as np
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
# 2. FITS IMAGE INGESTION, PARSING & CACHING
# =====================================================================
import io
import os
import base64
from typing import Union, Dict, Any, Optional
from PIL import Image


def parse_and_inspect_fits(file_source: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Parses any astronomical FITS file (from local file path, raw bytes, or base64 data URL),
    extracts 2D image data (including multi-extension FITS and Kepler/TESS Target Pixel Files),
    generates a web-ready base64 PNG preview, extracts astronomical header metadata,
    and returns a preprocessed (3, 224, 224) float tensor array.
    """
    hdul = None
    try:
        if isinstance(file_source, str):
            if file_source.startswith("data:"):
                # Base64 data URL
                b64_str = file_source.split(",", 1)[1] if "," in file_source else file_source
                hdul = fits.open(io.BytesIO(base64.b64decode(b64_str)))
            elif os.path.exists(file_source):
                # Local filesystem path
                hdul = fits.open(file_source)
            else:
                # Attempt base64 decode if plain base64 string
                try:
                    raw_b = base64.b64decode(file_source)
                    hdul = fits.open(io.BytesIO(raw_b))
                except Exception:
                    raise FileNotFoundError(f"FITS file path does not exist on disk: {file_source}")
        elif isinstance(file_source, bytes):
            hdul = fits.open(io.BytesIO(file_source))
        elif isinstance(file_source, io.BytesIO):
            hdul = fits.open(file_source)
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        # 1. Inspect HDUs to find 2D image or TPF table
        raw_2d = None
        target_hdr = hdul[0].header

        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                # Check for direct 2D image array
                if hasattr(hdu.data, "ndim") and hdu.data.ndim == 2:
                    raw_2d = np.array(hdu.data, dtype=np.float32)
                    target_hdr = hdu.header
                    break
                # Check for 3D/4D datacube (take median/first channel)
                elif hasattr(hdu.data, "ndim") and hdu.data.ndim >= 3:
                    arr = np.array(hdu.data, dtype=np.float32)
                    raw_2d = np.nanmedian(arr, axis=0) if arr.ndim == 3 else np.nanmedian(arr[0], axis=0)
                    target_hdr = hdu.header
                    break
                # Check for Kepler / TESS Target Pixel File binary table
                elif hasattr(hdu.data, "names"):
                    if "FLUX" in hdu.data.names:
                        flux_cube = hdu.data["FLUX"]
                        raw_2d = np.nanmedian(flux_cube, axis=0).astype(np.float32)
                        target_hdr = hdu.header
                        break
                    elif "RAW_CNTS" in hdu.data.names:
                        cnts_cube = hdu.data["RAW_CNTS"]
                        raw_2d = np.nanmedian(cnts_cube, axis=0).astype(np.float32)
                        target_hdr = hdu.header
                        break

        if raw_2d is None:
            raise ValueError("No 2D image matrix or Target Pixel File FLUX table found in FITS extensions.")

        # Fix endianness
        if raw_2d.dtype.byteorder == ">":
            raw_2d = raw_2d.byteswap().newbyteorder()

        # Handle NaNs and Infs
        raw_2d = np.nan_to_num(raw_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Extract Astronomical Metadata from Primary & Image Headers
        primary_hdr = hdul[0].header
        def get_hdr_val(*keys, default="N/A"):
            for k in keys:
                if k in target_hdr:
                    return target_hdr[k]
                if k in primary_hdr:
                    return primary_hdr[k]
            return default

        target_name = str(get_hdr_val("TARGET", "OBJECT", "KEPLERID", "TICID", "KEPID", default="Custom Target"))
        ra_val = get_hdr_val("RA_OBJ", "RA", "CRVAL1", default=None)
        dec_val = get_hdr_val("DEC_OBJ", "DEC", "CRVAL2", default=None)
        telescope = str(get_hdr_val("TELESCOP", "INSTRUME", "OBSERVER", default="Optical Observatory"))
        filter_val = str(get_hdr_val("FILTER", "BAND", "SURVEY", default="Broadband"))
        exptime = get_hdr_val("EXPTIME", "TELAPSE", default=None)

        # 3. Generate Preprocessed (3, 224, 224) Tensor
        tensor_3ch = process_fits_image_v4(raw_2d, target_shape=(224, 224))

        # 4. Generate Web-Ready PNG Preview (Base64 Data URL)
        min_v = np.percentile(raw_2d, 1) if raw_2d.size > 0 else 0.0
        max_v = np.percentile(raw_2d, 99.5) if raw_2d.size > 0 else 1.0
        denom = max_v - min_v if (max_v - min_v) > 1e-8 else 1.0
        norm = np.clip((raw_2d - min_v) / denom, 0.0, 1.0)
        
        pil_preview = Image.fromarray((norm * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        norm_arr = np.array(pil_preview, dtype=np.float32) / 255.0
        
        rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(norm_arr * 255 * 1.15, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(norm_arr**1.4 * 220 + (1 - norm_arr) * 15, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(norm_arr**2.0 * 255 + (1 - norm_arr) * 40, 0, 255).astype(np.uint8)
        
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        preview_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # 5. Summary Statistics
        stats = {
            "min": float(np.nanmin(raw_2d)),
            "max": float(np.nanmax(raw_2d)),
            "median": float(np.nanmedian(raw_2d)),
            "std": float(np.nanstd(raw_2d)),
            "original_shape": list(raw_2d.shape),
            "processed_shape": [3, 224, 224],
        }

        metadata = {
            "target_name": target_name,
            "ra": float(ra_val) if isinstance(ra_val, (int, float)) else None,
            "dec": float(dec_val) if isinstance(dec_val, (int, float)) else None,
            "telescope": telescope,
            "filter": filter_val,
            "exptime": float(exptime) if isinstance(exptime, (int, float)) else None,
            "hdu_count": len(hdul),
        }

        return {
            "tensor": tensor_3ch,
            "metadata": metadata,
            "preview_data_url": preview_data_url,
            "pixel_stats": stats,
            "is_valid_fits": True,
        }
    finally:
        if hdul is not None:
            hdul.close()


@lru_cache(maxsize=500)
def load_fits_image(file_path):
    """
    Loads a local FITS image file, fixes endianness, applies arcsinh 
    stretching, and returns a (3, 224, 224) numpy array.
    """
    res = parse_and_inspect_fits(file_path)
    return res["tensor"]


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