"""
AstroFoundation — 1D Spectra Dataset Generator (2,000 Targets)
==============================================================
Synthesizes continuum-normalized rest-frame stellar and transient spectra
on the standardized 3800–9200 Å grid (N=4096) for all 2,000 targets in the
AstroFoundation master manifest.

Physical Line Signatures:
- Class 0 (Exoplanet transit host / G-K dwarf): Balmer (H-a, H-b, H-g) + Na I D + Mg I b + Ca II H/K absorption.
- Class 1 (Eclipsing binary): Dual Doppler-split composite Balmer & Fe absorption systems.
- Class 2 (Variable star): Deep TiO molecular bandheads (5167, 6158, 7054, 7666 Å).
- Class 3 (Supernova): Broad P-Cygni Doppler-broadened (~10,000 km/s) profiles, deep Si II 6355 Å dip.
- Class 4 (Cataclysmic variable): Broad double-peaked accretion-disk emission lines (H-a, H-b, He I 5876 Å).
- Class 5 (Gamma-ray burst afterglow): Synchrotron power-law continuum + narrow intervening metal absorbers.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

MANIFEST_PATH = "data/processed/astro_manifest_2000.csv"
OUTPUT_NPZ = "data/processed/spectra_2000.npz"
GRID_LEN = 4096
LAMBDA_MIN = 3800.0
LAMBDA_MAX = 9200.0


def make_gaussian_line(grid, center, fwhm, amplitude):
    """
    Creates a Gaussian profile. Negative amplitude = absorption, Positive = emission.
    """
    sigma = fwhm / 2.35482
    return amplitude * np.exp(-0.5 * ((grid - center) / sigma) ** 2)


def make_double_peaked_emission(grid, center, separation, fwhm, peak_height):
    """
    Creates a classic double-peaked accretion disk emission profile with central dip.
    """
    blue_peak = make_gaussian_line(grid, center - separation / 2.0, fwhm, peak_height)
    red_peak = make_gaussian_line(grid, center + separation / 2.0, fwhm, peak_height)
    central_dip = make_gaussian_line(grid, center, fwhm * 0.8, -0.4 * peak_height)
    return blue_peak + red_peak + central_dip


def make_tio_bandhead(grid, head_wavelength, width, depth):
    """
    Creates an asymmetric molecular TiO band (sharp head towards blue, degrades to red).
    """
    diff = grid - head_wavelength
    profile = np.zeros_like(grid)
    mask = diff >= 0
    profile[mask] = -depth * np.exp(-diff[mask] / width)
    return profile


def generate_spectrum_for_class(class_name: str, rng: np.random.RandomState, grid: np.ndarray) -> np.ndarray:
    """
    Generates a continuum-normalized spectrum (baseline ~ 1.0) with characteristic
    physical features of the specified astrophysical event type.
    """
    # Baseline continuum normalized at 1.0
    flux = np.ones_like(grid)

    if class_name == "exoplanet_transit":
        # Solar / G-K type stellar spectrum with narrow atomic absorption lines
        # Small Doppler jitter
        v_offset = rng.uniform(-5.0, 5.0)

        # Balmer series absorption
        flux += make_gaussian_line(grid, 6562.8 + v_offset, rng.uniform(8.0, 14.0), -rng.uniform(0.35, 0.55))  # H-alpha
        flux += make_gaussian_line(grid, 4861.3 + v_offset, rng.uniform(6.0, 11.0), -rng.uniform(0.30, 0.45))  # H-beta
        flux += make_gaussian_line(grid, 4340.5 + v_offset, rng.uniform(5.0, 9.0), -rng.uniform(0.20, 0.35))   # H-gamma

        # Na I D doublet (5890, 5896 A)
        flux += make_gaussian_line(grid, 5889.9 + v_offset, 5.0, -rng.uniform(0.30, 0.45))
        flux += make_gaussian_line(grid, 5895.9 + v_offset, 5.0, -rng.uniform(0.25, 0.40))

        # Mg I b triplet (5175 A)
        flux += make_gaussian_line(grid, 5175.0 + v_offset, 8.0, -rng.uniform(0.30, 0.45))

        # Ca II H & K (3934, 3968 A)
        flux += make_gaussian_line(grid, 3933.7 + v_offset, 10.0, -rng.uniform(0.40, 0.60))
        flux += make_gaussian_line(grid, 3968.5 + v_offset, 9.0, -rng.uniform(0.35, 0.50))

        # Instrumental noise
        flux += rng.normal(0, 0.015, len(grid))

    elif class_name == "eclipsing_binary":
        # Double-lined binary: two sets of Doppler-shifted absorption lines
        v_split = rng.uniform(12.0, 30.0)  # Relative orbital velocity shift in A

        # Primary star lines
        flux += make_gaussian_line(grid, 6562.8 - v_split / 2, 9.0, -rng.uniform(0.25, 0.35))
        flux += make_gaussian_line(grid, 4861.3 - v_split / 2, 7.0, -rng.uniform(0.20, 0.30))

        # Secondary star lines
        flux += make_gaussian_line(grid, 6562.8 + v_split / 2, 9.0, -rng.uniform(0.20, 0.30))
        flux += make_gaussian_line(grid, 4861.3 + v_split / 2, 7.0, -rng.uniform(0.18, 0.28))

        # Combined metallic lines
        flux += make_gaussian_line(grid, 5175.0, 10.0, -rng.uniform(0.20, 0.30))
        flux += rng.normal(0, 0.015, len(grid))

    elif class_name == "variable_star":
        # Cool pulsating giant with prominent TiO molecular bands
        v_offset = rng.uniform(-3.0, 3.0)

        # Deep TiO molecular bands Degrading to the red
        flux += make_tio_bandhead(grid, 5167.0 + v_offset, width=120.0, depth=rng.uniform(0.25, 0.40))
        flux += make_tio_bandhead(grid, 6158.0 + v_offset, width=160.0, depth=rng.uniform(0.30, 0.50))
        flux += make_tio_bandhead(grid, 7054.0 + v_offset, width=220.0, depth=rng.uniform(0.35, 0.55))
        flux += make_tio_bandhead(grid, 7666.0 + v_offset, width=180.0, depth=rng.uniform(0.30, 0.45))

        # Weak H-alpha
        flux += make_gaussian_line(grid, 6562.8 + v_offset, 12.0, -rng.uniform(0.15, 0.30))
        flux += rng.normal(0, 0.015, len(grid))

    elif class_name == "supernova":
        # Massive relativistic expansion velocity (~10,000 km/s -> ~150-250 A broad P-Cygni profiles)
        sn_shift = rng.uniform(80.0, 140.0)  # Blueshifted absorption

        # Classic broad Si II 6355 A absorption minimum near 6150 A
        flux += make_gaussian_line(grid, 6355.0 - sn_shift, 140.0, -rng.uniform(0.45, 0.65))
        # P-Cygni emission peak near rest wavelength
        flux += make_gaussian_line(grid, 6355.0, 90.0, rng.uniform(0.20, 0.35))

        # Broad Ca II IR triplet absorption (~8500 A blueshifted)
        flux += make_gaussian_line(grid, 8542.0 - sn_shift, 180.0, -rng.uniform(0.40, 0.60))
        flux += make_gaussian_line(grid, 8542.0, 110.0, rng.uniform(0.20, 0.35))

        # Fe II broad complex (4800-5100 A)
        flux += make_gaussian_line(grid, 4900.0, 150.0, -rng.uniform(0.25, 0.40))
        flux += rng.normal(0, 0.02, len(grid))

    elif class_name == "cataclysmic_variable":
        # Accretion disk: Hot continuum with broad double-peaked emission lines
        # H-alpha double-peaked emission
        flux += make_double_peaked_emission(grid, 6562.8, separation=25.0, fwhm=35.0, peak_height=rng.uniform(1.2, 2.2))
        # H-beta double-peaked emission
        flux += make_double_peaked_emission(grid, 4861.3, separation=20.0, fwhm=28.0, peak_height=rng.uniform(0.8, 1.6))
        # He I 5876 emission
        flux += make_gaussian_line(grid, 5875.6, 25.0, rng.uniform(0.3, 0.7))
        # He II 4686 emission
        flux += make_gaussian_line(grid, 4685.7, 22.0, rng.uniform(0.25, 0.6))
        flux += rng.normal(0, 0.02, len(grid))

    elif class_name == "gamma_ray_burst":
        # Synchrotron power-law afterglow with narrow intervening absorption lines
        # Power law continuum trend across wavelength
        slope = rng.uniform(-0.15, 0.15)
        norm_grid = (grid - 6500.0) / 2700.0
        flux += slope * norm_grid

        # Intervening absorption systems (random metal absorbers along line of sight)
        n_absorbers = rng.randint(3, 7)
        for _ in range(n_absorbers):
            z_wave = rng.uniform(4000.0, 8500.0)
            flux += make_gaussian_line(grid, z_wave, rng.uniform(3.0, 6.0), -rng.uniform(0.25, 0.50))
        flux += rng.normal(0, 0.025, len(grid))

    # Guarantee continuum normalized floor >= 0.05
    flux = np.clip(flux, 0.05, 5.0)
    return flux.astype(np.float32)


def main():
    print(f"\n{'=' * 60}")
    print(f"  AstroFoundation Spectral Dataset Builder (2000 Targets)")
    print(f"{'=' * 60}")

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded master manifest: {len(df)} objects.")

    grid = np.linspace(LAMBDA_MIN, LAMBDA_MAX, GRID_LEN, dtype=np.float32)
    all_spectra = np.zeros((len(df), GRID_LEN), dtype=np.float32)
    all_labels = df['label'].values.astype(np.int64)
    kic_ids = df['kic_id'].values.astype(np.int64)

    rng = np.random.RandomState(42)

    for i, row in df.iterrows():
        spec = generate_spectrum_for_class(row['class_name'], rng, grid)
        all_spectra[i] = spec

        if (i + 1) % 400 == 0:
            print(f"  Generated {i + 1}/{len(df)} physical spectra...")

    # Save to compressed NPZ
    os.makedirs(Path(OUTPUT_NPZ).parent, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        spectra=all_spectra,
        labels=all_labels,
        kic_ids=kic_ids,
        wavelength=grid,
        class_names=np.array([
            "exoplanet_transit",
            "eclipsing_binary",
            "variable_star",
            "supernova",
            "cataclysmic_variable",
            "gamma_ray_burst"
        ])
    )

    size_mb = os.path.getsize(OUTPUT_NPZ) / (1024 * 1024)
    print(f"\nSuccessfully generated {OUTPUT_NPZ} ({size_mb:.1f} MB)")
    print(f"  Spectra tensor shape: {all_spectra.shape}")
    print(f"  Labels tensor shape:  {all_labels.shape}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
