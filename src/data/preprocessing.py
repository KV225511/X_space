import numpy as np

# =====================================================================
# 1. LIGHT CURVE PREPROCESSING (N = 8192 + DYNAMIC GAP MASKING)
# =====================================================================
def process_light_curve_v4(time, flux, target_len=8192):
    """
    Resamples light curves to 8192 points and generates a dynamic 
    attention mask that flags synthetic interpolation across telemetry gaps.
    """
    # Step 1: Strip invalid NaN observations
    valid_mask = ~np.isnan(flux) & ~np.isnan(time)
    clean_time = time[valid_mask]
    clean_flux = flux[valid_mask]

    if len(clean_time) == 0:
        return np.zeros(target_len, dtype=np.float32), np.zeros(target_len, dtype=np.float32)

    # Step 2: Identify original telemetry gaps (> 5x median time delta)
    time_diffs = np.diff(clean_time)
    median_delta = np.median(time_diffs)
    gap_threshold = 5.0 * median_delta

    # Step 3: Uniform 8192 Grid Generation & Interpolation
    new_time_grid = np.linspace(clean_time[0], clean_time[-1], target_len)
    resampled_flux = np.interp(new_time_grid, clean_time, clean_flux)

    # Step 4: Dynamic Telemetry Gap Masking
    # 1.0 = valid observation, 0.0 = gap interpolation (to be ignored by FlashAttention)
    mask = np.ones(target_len, dtype=np.float32)
    gap_indices = np.where(time_diffs > gap_threshold)[0]

    for idx in gap_indices:
        gap_start = clean_time[idx]
        gap_end = clean_time[idx + 1]
        # Set mask to 0.0 for any grid point falling inside a physical gap
        in_gap = (new_time_grid > gap_start) & (new_time_grid < gap_end)
        mask[in_gap] = 0.0

    # Step 5: Zero-Centered Baseline Normalization
    valid_flux = resampled_flux[mask == 1.0]
    median_val = np.median(valid_flux) if len(valid_flux) > 0 else np.median(resampled_flux)
    
    if median_val != 0:
        resampled_flux = (resampled_flux / median_val) - 1.0

    return resampled_flux.astype(np.float32), mask.astype(np.float32)


# =====================================================================
# 2. FITS IMAGE ARCSINH STRETCHING & MAD SCALING
# =====================================================================
def process_fits_image_v4(image_data, target_shape=(224, 224), epsilon=1e-8):
    """
    Normalizes 2D FITS images using Median Absolute Deviation (MAD) 
    and applies an arcsinh non-linear stretch to handle high dynamic range.
    Outputs shape: (3, H, W) for Vision Backbones.
    """
    # Step 1: Sky Background Statistics ignoring NaNs
    valid_pixels = image_data[~np.isnan(image_data)]
    if len(valid_pixels) == 0:
        return np.zeros((3, *target_shape), dtype=np.float32)
        
    sky_median = np.median(valid_pixels)
    mad = np.median(np.abs(valid_pixels - sky_median))
    sigma_sky = 1.4826 * mad  # Scaled background noise estimate

    # Step 2: Zero-Centered Normalization
    normalized = (image_data - sky_median) / (sigma_sky + epsilon)

    # Step 3: Arcsinh Stretch (Linear near sky level, logarithmic for bright cores)
    stretched = np.arcsinh(normalized)
    stretched = np.nan_to_num(stretched, nan=0.0)

    # Step 4: Simple Resize to Target Shape (224x224) via Crop/Pad or Interpolation
    h, w = stretched.shape
    if (h, w) != target_shape:
        # Simple nearest-neighbor cropping/padding for demonstration
        resampled = np.zeros(target_shape, dtype=np.float32)
        min_h, min_w = min(h, target_shape[0]), min(w, target_shape[1])
        resampled[:min_h, :min_w] = stretched[:min_h, :min_w]
        stretched = resampled

    # Step 5: Convert 2D image into 3-channel (3, H, W) for vision backbones
    stacked_3channel = np.stack([stretched] * 3, axis=0)

    return stacked_3channel.astype(np.float32)


# =====================================================================
# 3. SPECTRAL REST-FRAME ALIGNMENT & CONTINUUM NORMALIZATION
# =====================================================================
def process_spectrum_v4(observed_wavelength, flux, redshift_z=0.0, target_len=4096, epsilon=1e-8):
    """
    De-shifts spectra to the rest frame using redshift z, interpolates to a 
    standardized 4096 Å grid, and normalizes the continuum via rolling median.
    """
    # Step 1: Doppler De-shifting to universal rest frame
    rest_wavelength = observed_wavelength / (1.0 + redshift_z)

    # Step 2: Uniform Wavelength Binning (3800 Å to 9200 Å)
    standard_grid = np.linspace(3800.0, 9200.0, target_len)
    resampled_flux = np.interp(standard_grid, rest_wavelength, flux)

    # Step 3: Rolling Median Continuum Estimation
    window_size = 101
    pad_width = window_size // 2
    padded = np.pad(resampled_flux, pad_width, mode='edge')
    
    continuum = np.array([
        np.median(padded[i : i + window_size]) 
        for i in range(target_len)
    ])

    # Step 4: Continuum Normalization (baseline sits at 1.0)
    normalized_flux = resampled_flux / (continuum + epsilon)

    return normalized_flux.astype(np.float32)