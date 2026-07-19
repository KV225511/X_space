import numpy as np

def sigma_clip(flux, sigma=5):
    """
    Removes NaN values and clips extreme statistical outliers (cosmic rays).
    """
    # 1. Drop NaNs first so the math doesn't break
    flux = flux[~np.isnan(flux)]
    
    # 2. Calculate the baseline and standard deviation
    mean = np.mean(flux)
    std = np.std(flux)
    
    # 3. Create a filter to only keep data within our 'sigma' boundary
    safe_data_mask = np.abs(flux - mean) <= (sigma * std)
    
    return flux[safe_data_mask]

def normalize_flux(flux):
    """
    Forces the baseline of the star's brightness to center exactly at 1.0.
    """
    median_val = np.median(flux)
    return flux / median_val

def pad_or_truncate(flux, target_len=1024):
    """
    Forces the array to be exactly target_len for the PyTorch model.
    """
    current_len = len(flux)
    
    if current_len > target_len:
        return flux[:target_len]
        
    elif current_len < target_len:
        pad_amount = target_len - current_len
        return np.pad(flux, (0, pad_amount), mode='constant', constant_values=0.0)
        
    return flux