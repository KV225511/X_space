import torch
from torch.utils.data import Dataset
from .ingestion import load_light_curve, load_fits_image, load_spectrum

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