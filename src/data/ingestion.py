import lightkurve as lk
from functools import lru_cache
# Import the updated math functions
from .preprocessing import clean_and_center_flux, resample_and_mask

@lru_cache(maxsize=500)
def load_light_curve(kic_id):
    """
    Downloads raw data, processes it, and caches it in RAM to feed the GPU.
    Returns the flux array and the validity mask.
    """
    # 1. Fetch data from NASA
    search_result = lk.search_lightcurve(kic_id, author='Kepler', quarter=3)
    if len(search_result) == 0:
        raise ValueError(f"No data found for {kic_id}")
        
    lc = search_result.download()
    
    # Fill physical time gaps while it is still a LightCurve object
    lc = lc.fill_gaps()
    
    # 2. Extract raw time AND flux arrays (Time is required for resampling now)
    raw_time = lc.time.value
    raw_flux = lc.flux.value
    
    # 3. Route through the new preprocessing pipeline
    centered_flux = clean_and_center_flux(raw_flux)
    final_flux, final_mask = resample_and_mask(raw_time, centered_flux, target_len=1024)
    
    return final_flux, final_mask