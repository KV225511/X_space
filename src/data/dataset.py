import torch
from torch.utils.data import Dataset
# Import the orchestrator function you just built
from .ingestion import load_light_curve

class LightCurveDataset(Dataset):
    """
    PyTorch Dataset for Kepler Light Curves.
    Dynamically fetches and preprocesses telescope data on the fly.
    """
    
    def __init__(self, kic_ids, labels):
        """
        Initializes the dataset with a list of target stars and their categories.
        """
        self.kic_ids = kic_ids
        self.labels = labels
        
    def __len__(self):
        """
        Returns the total number of stars in this dataset.
        """
        return len(self.kic_ids)
        
    def __getitem__(self, index):
        """
        Retrieves a single star's data by its index, processes it, 
        and converts it into PyTorch tensors.
        """
        # 1. Look up the specific star ID and its corresponding label
        kic_id = self.kic_ids[index]
        label = self.labels[index]
        
        # 2. Fetch and process the data using your ingestion pipeline
        # This returns a cleaned, normalized numpy array of shape (1024,)
        flux_array = load_light_curve(kic_id)
        
        # 3. Convert the numpy array to a PyTorch Tensor
        # Neural networks require 32-bit floats, so we explicitly cast it
        flux_tensor = torch.tensor(flux_array, dtype=torch.float32)
        
        # 4. Convert the label to a PyTorch Tensor
        # Classification labels must be integers (Long tensors in PyTorch)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 5. Return the finalized tuple
        return flux_tensor, label_tensor