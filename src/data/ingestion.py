import lightkurve as lk
# Import the custom math functions you just built
from .preprocessing import sigma_clip, normalize_flux, pad_or_truncate

def load_light_curve(kic_id):
    """
    Downloads raw data and passes it through the preprocessing pipeline.
    Returns a pristine (1024,) numpy array.
    """
    # 1. Fetch data
    search_result = lk.search_lightcurve(kic_id, author='Kepler', quarter=3)
    if len(search_result) == 0:
        raise ValueError(f"No data found for {kic_id}")
        
    lc = search_result.download()
    
    # Optional: We still use lightkurve's fill_gaps() here because it 
    # relies on time-stamps which are lost once we convert to pure Numpy.
    lc = lc.fill_gaps()
    
    # 2. Extract raw array
    raw_flux = lc.flux.value
    
    # 3. Route through your preprocessing pipeline
    clean_flux = sigma_clip(raw_flux, sigma=5)
    normalized_flux = normalize_flux(clean_flux)
    final_array = pad_or_truncate(normalized_flux, target_len=1024)
    
    return final_array