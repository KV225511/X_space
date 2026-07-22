import numpy as np

def clean_and_center_flux(flux):
    """
    Zero-centers the baseline and clips (instead of deleting) extreme outliers 
    to preserve the length and time alignment of the array.
    """
    # 1. Ignore NaNs to find the safe median
    median_val = np.nanmedian(flux)
    
    # 2. Zero-Centered Normalization
    # The normal baseline becomes 0.0. Transit dips become negative numbers.
    relative_flux = (flux / median_val) - 1.0
    
    # 3. Soft Outlier Clipping
    # Calculates standard deviation and caps massive spikes at 5 sigma.
    std_val = np.nanstd(relative_flux)
    clip_limit = 5 * std_val
    relative_flux = np.clip(relative_flux, -clip_limit, clip_limit)
    
    return relative_flux

def resample_and_mask(time, flux, target_len=1024):
    """
    Compresses the entire light curve timeline into exactly 1024 points 
    and generates the validity mask for the Transformer attention mechanism.
    """
    # 1. Strip out the NaN rows before resampling so they don't break the interpolation
    valid_indices = ~np.isnan(flux)
    clean_time = time[valid_indices]
    clean_flux = flux[valid_indices]
    
    # 2. Create a perfectly evenly spaced time grid of 1024 points
    # spanning from the very first observation to the very last
    new_time_grid = np.linspace(clean_time[0], clean_time[-1], target_len)
    
    # 3. Interpolate (Resample) the flux onto this new 1024-length grid
    resampled_flux = np.interp(new_time_grid, clean_time, clean_flux)
    
    # 4. Create the Validity Mask
    # Since we successfully resampled the whole curve, all 1024 points are valid data (1s)
    mask = np.ones(target_len, dtype=np.float32)
    
    return resampled_flux, mask