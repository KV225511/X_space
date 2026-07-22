import torch
from torch.utils.data import Dataset
# Import the cached orchestrator function
from .ingestion import load_light_curve

class LightCurveDataset(Dataset):
    """
    PyTorch Dataset for Kepler Light Curves.
    Fetches data (or pulls from RAM cache) and returns (flux, mask, label).
    """
    
    def __init__(self, kic_ids, labels):
        self.kic_ids = kic_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.kic_ids)
        
    def __getitem__(self, index):
        kic_id = self.kic_ids[index]
        label = self.labels[index]
        
        # 1. Fetch processed data (instantly returned from RAM if already seen)
        flux_array, mask_array = load_light_curve(kic_id)
        
        # 2. Convert to PyTorch Tensors
        flux_tensor = torch.tensor(flux_array, dtype=torch.float32)
        
        # Masks are typically booleans or integers in PyTorch attention layers
        mask_tensor = torch.tensor(mask_array, dtype=torch.bool) 
        
        # Labels must be Long integers for classification
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 3. Return the three-part tuple required by the Master Plan
        return flux_tensor, mask_tensor, label_tensor