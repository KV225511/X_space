import numpy as np
from PIL import Image

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

    # Step 4: Robust Bilinear Resize to Target Shape (224, 224) preserving PSF centering
    h, w = stretched.shape
    if (h, w) != target_shape:
        min_v = float(stretched.min())
        max_v = float(stretched.max())
        range_v = max_v - min_v if (max_v - min_v) > 1e-8 else 1.0
        norm01 = np.clip((stretched - min_v) / range_v, 0.0, 1.0)
        pil_img = Image.fromarray((norm01 * 65535.0).astype(np.uint16))
        pil_resized = pil_img.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
        stretched = (np.array(pil_resized, dtype=np.float32) / 65535.0) * range_v + min_v

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


# =====================================================================
# 4. SPECTROSCOPIC DATA AUGMENTATION (PHYSICS-INFORMED)
# =====================================================================
def augment_spectrum(spectrum: np.ndarray, p: float = 0.5, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Physics-informed 1D spectroscopic data augmentation:
    1. Random Doppler velocity jitter: roll shift by up to ±4 bins (~±80 km/s)
    2. Random instrumental Gaussian noise injection
    3. Random subtle continuum tilt / extinction slope across wavelength
    4. SpecAugment cutout: mask 10-25 contiguous bins to baseline (1.0)
    """
    if rng is None:
        rng = np.random.RandomState()

    aug = spectrum.copy()

    # 1. Random Doppler shift
    if rng.rand() < p:
        shift = rng.randint(-4, 5)
        aug = np.roll(aug, shift)

    # 2. Random Gaussian noise injection
    if rng.rand() < p:
        noise = rng.normal(0.0, 0.015, size=len(aug))
        aug = aug + noise

    # 3. Continuum tilt / reddening slope
    if rng.rand() < p:
        slope = rng.uniform(-0.04, 0.04)
        tilt = np.linspace(1.0 - slope, 1.0 + slope, len(aug))
        aug = aug * tilt

    # 4. SpecAugment cutout
    if rng.rand() < p:
        cutout_width = rng.randint(10, 25)
        start_idx = rng.randint(0, len(aug) - cutout_width)
        aug[start_idx : start_idx + cutout_width] = 1.0

    return np.clip(aug, 0.01, 10.0).astype(np.float32)


# =====================================================================
# 5. LIGHT CURVE DATA AUGMENTATION (PHYSICS-INFORMED)
# =====================================================================
def augment_light_curve(
    flux: np.ndarray, 
    mask: np.ndarray, 
    p: float = 0.5, 
    rng: np.random.RandomState = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Physics-informed 1D photometric light curve data augmentation:
    1. Random circular phase shift: rolls flux along timeline (arbitrary observation epoch).
    2. Gaussian photometric noise injection: simulates Poisson photon noise / faint targets.
    3. Stellar spot / baseline drift: adds subtle low-frequency sinusoidal variation.
    4. Telemetry dropout simulation: drops contiguous blocks and sets mask to False.
    5. Cosmic ray hit simulation: injects 1-3 sharp positive outlier spikes.
    """
    if rng is None:
        rng = np.random.RandomState()

    aug_flux = flux.copy()
    aug_mask = mask.copy()
    n_pts = len(aug_flux)

    # 1. Circular phase shift
    if rng.rand() < p:
        shift = rng.randint(0, n_pts)
        aug_flux = np.roll(aug_flux, shift)
        aug_mask = np.roll(aug_mask, shift)

    # 2. Gaussian photometric shot noise
    if rng.rand() < p:
        sigma = rng.uniform(0.005, 0.025)
        noise = rng.normal(0.0, sigma, size=n_pts)
        aug_flux = aug_flux + noise

    # 3. Stellar spot low-frequency baseline modulation
    if rng.rand() < p:
        period = rng.uniform(n_pts * 0.5, n_pts * 2.0)
        amplitude = rng.uniform(0.005, 0.025)
        t = np.arange(n_pts)
        trend = amplitude * np.sin(2.0 * np.pi * t / period)
        aug_flux = aug_flux + trend

    # 4. Telemetry dropouts (contiguous gap simulation)
    if rng.rand() < p:
        gap_len = rng.randint(30, 150)
        start = rng.randint(0, n_pts - gap_len)
        aug_flux[start : start + gap_len] = 0.0
        aug_mask[start : start + gap_len] = False

    # 5. Cosmic ray positive spikes (1-3 events)
    if rng.rand() < p:
        num_spikes = rng.randint(1, 4)
        spike_indices = rng.randint(0, n_pts, size=num_spikes)
        spike_amps = rng.uniform(0.05, 0.20, size=num_spikes)
        aug_flux[spike_indices] += spike_amps

    return aug_flux.astype(np.float32), aug_mask.astype(bool)


# =====================================================================
# 6. FITS IMAGE DATA AUGMENTATION (ASTRONOMICAL SYMMETRY & SENSORS)
# =====================================================================
def augment_fits_image(
    image_3ch: np.ndarray, 
    p: float = 0.5, 
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Astronomical 2D FITS image data augmentation:
    1. D4 Group Symmetries: 90/180/270 deg rotation & horizontal/vertical flips.
    2. Sub-pixel translation jitter: ±2-4 pixel spatial pointing drift.
    3. Background sky & CCD readout noise injection.
    """
    if rng is None:
        rng = np.random.RandomState()

    aug = image_3ch.copy()  # shape (3, H, W)

    # 1. Random D4 rotational symmetries
    if rng.rand() < p:
        k = rng.randint(0, 4)
        aug = np.rot90(aug, k=k, axes=(1, 2))

    # 2. Random horizontal and vertical flips
    if rng.rand() < p:
        aug = np.flip(aug, axis=1)  # vertical flip
    if rng.rand() < p:
        aug = np.flip(aug, axis=2)  # horizontal flip

    # 3. Spatial translation jitter (pointing drift)
    if rng.rand() < p:
        dy = rng.randint(-3, 4)
        dx = rng.randint(-3, 4)
        aug = np.roll(aug, shift=(dy, dx), axis=(1, 2))

    # 4. CCD sky readout noise
    if rng.rand() < p:
        noise_sigma = rng.uniform(0.005, 0.02)
        noise = rng.normal(0.0, noise_sigma, size=aug.shape)
        aug = aug + noise

    return aug.astype(np.float32)