import torch
import numpy as np
from torch.utils.data import Dataset

from .ingestion import load_light_curve, load_fits_image, load_spectrum
from .preprocessing import augment_spectrum

# =====================================================================
# 1. LIGHT CURVE DATASET
# =====================================================================
class LightCurveDataset(Dataset):
    """
    PyTorch Dataset for Kepler Light Curves.
    Outputs:
      - flux_tensor: shape (8192,) float32
      - mask_tensor: shape (8192,) bool
      - label_tensor: scalar long
    """
    def __init__(self, kic_ids, labels):
        self.kic_ids = kic_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.kic_ids)
        
    def __getitem__(self, index):
        kic_id = self.kic_ids[index]
        label = self.labels[index]
        
        flux_array, mask_array = load_light_curve(kic_id)
        
        flux_tensor = torch.tensor(flux_array, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return flux_tensor, mask_tensor, label_tensor


# =====================================================================
# 2. FITS IMAGE DATASET
# =====================================================================
class FITSDataset(Dataset):
    """
    PyTorch Dataset for 2D Telescope FITS Images.
    Outputs:
      - image_tensor: shape (3, 224, 224) float32
      - label_tensor: scalar long
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        image_array = load_fits_image(file_path)

        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


# =====================================================================
# 3. SPECTRAL DATASET
# =====================================================================
class SpectralDataset(Dataset):
    """
    PyTorch Dataset for 1D Spectroscopic Observations.
    Outputs:
      - spectral_tensor: shape (4096,) float32
      - label_tensor: scalar long
    """
    def __init__(self, file_paths, redshifts, labels):
        self.file_paths = file_paths
        self.redshifts = redshifts
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        redshift_z = self.redshifts[index]
        label = self.labels[index]

        spectral_array = load_spectrum(file_path, redshift_z=redshift_z)

        spectral_tensor = torch.tensor(spectral_array, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spectral_tensor, label_tensor


# =====================================================================
# 4. NPZ LIGHT CURVE DATASET (Pre-downloaded, offline, fast)
# =====================================================================
class NPZLightCurveDataset(Dataset):
    """
    Fast offline PyTorch Dataset that reads from the pre-built
    data/processed/light_curves_2000.npz archive.
    No network calls — all data is already on disk.
    Outputs:
      - flux_tensor: shape (8192,) float32
      - mask_tensor: shape (8192,) bool
      - label_tensor: scalar long
    """
    def __init__(self, npz_path: str, indices=None):
        """
        Args:
            npz_path: Path to light_curves_2000.npz
            indices:  Optional list/array of row indices for train/val/test split.
                      If None, loads all 2000 samples.
        """
        data = np.load(npz_path, allow_pickle=True)
        self.flux   = data['flux']    # (N, 8192) float32
        self.masks  = data['masks']   # (N, 8192) float32  → converted to bool
        self.labels = data['labels']  # (N,)      int64

        if indices is not None:
            self.flux   = self.flux[indices]
            self.masks  = self.masks[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flux_tensor  = torch.tensor(self.flux[index],            dtype=torch.float32)
        mask_tensor  = torch.tensor(self.masks[index] > 0.5,     dtype=torch.bool)
        label_tensor = torch.tensor(int(self.labels[index]),     dtype=torch.long)
        return flux_tensor, mask_tensor, label_tensor


# =====================================================================
# 5. NPZ SPECTRAL DATASET (Pre-processed 4096-bin spectra, offline)
# =====================================================================
class NPZSpectralDataset(Dataset):
    """
    Fast offline PyTorch Dataset that reads from data/processed/spectra_2000.npz or SDSS archives.
    Outputs:
      - spectral_tensor: shape (4096,) float32
      - label_tensor: scalar long
    """
    def __init__(self, npz_path: str, indices=None, augment: bool = False):
        data = np.load(npz_path, allow_pickle=True)
        self.spectra = data['spectra']  # (N, 4096) float32
        self.labels  = data['labels']   # (N,)      int64
        self.augment = augment

        if indices is not None:
            self.spectra = self.spectra[indices]
            self.labels  = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        spec = self.spectra[index]
        if self.augment:
            spec = augment_spectrum(spec)
        spec_tensor  = torch.tensor(spec,                    dtype=torch.float32)
        label_tensor = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return spec_tensor, label_tensor