import pytest
import torch
import numpy as np
from unittest.mock import patch

from src.data.dataset import LightCurveDataset, FITSDataset, SpectralDataset, NPZLightCurveDataset
from src.data.dataloaders import create_dataloader, get_dataloaders

# =====================================================================
# FIXTURES & MOCKS (Isolating the tests from the Internet/Disk)
# =====================================================================

@pytest.fixture
def dummy_light_curve_data():
    """Mocks ingestion.py to return a perfectly shaped Light Curve tuple."""
    with patch("src.data.dataset.load_light_curve") as mock:
        # Returns: (flux_array, mask_array)
        mock.return_value = (
            np.random.randn(8192).astype(np.float32), 
            np.ones(8192, dtype=np.float32)
        )
        yield mock

@pytest.fixture
def dummy_fits_data():
    """Mocks ingestion.py to return a perfectly shaped 3-channel FITS image."""
    with patch("src.data.dataset.load_fits_image") as mock:
        # Returns: (3, H, W) image array
        mock.return_value = np.random.randn(3, 224, 224).astype(np.float32)
        yield mock

@pytest.fixture
def dummy_spectral_data():
    """Mocks ingestion.py to return a perfectly shaped 1D Spectrum."""
    with patch("src.data.dataset.load_spectrum") as mock:
        # Returns: (4096,) spectral array
        mock.return_value = np.random.randn(4096).astype(np.float32)
        yield mock


# =====================================================================
# DATASET CONTRACT TESTS
# =====================================================================

def test_lightcurve_dataset_contract(dummy_light_curve_data):
    """Verifies the Light Curve Dataset returns the exact v4.0 tensors."""
    kic_ids = ["Kepler-10", "KIC-9700322"]
    labels = [1, 0]
    
    dataset = LightCurveDataset(kic_ids=kic_ids, labels=labels)
    assert len(dataset) == 2, "Dataset length should match input list."
    
    flux, mask, label = dataset[0]
    
    # 1. Type Checks
    assert isinstance(flux, torch.Tensor), "Flux must be a PyTorch Tensor."
    assert isinstance(mask, torch.Tensor), "Mask must be a PyTorch Tensor."
    assert isinstance(label, torch.Tensor), "Label must be a PyTorch Tensor."
    
    # 2. Shape Checks
    assert flux.shape == (8192,), f"Expected flux shape (8192,), got {flux.shape}"
    assert mask.shape == (8192,), f"Expected mask shape (8192,), got {mask.shape}"
    
    # 3. Data Type Checks (Crucial for GPU stability)
    assert flux.dtype == torch.float32, "Neural networks require float32."
    assert mask.dtype == torch.bool, "Attention masks must be boolean."
    assert label.dtype == torch.long, "Classification labels must be Long integers."


def test_fits_dataset_contract(dummy_fits_data):
    """Verifies the FITS Vision Dataset returns the exact v4.0 tensors."""
    file_paths = ["/fake/path/1.fits", "/fake/path/2.fits"]
    labels = [2, 1]
    
    dataset = FITSDataset(file_paths=file_paths, labels=labels)
    image, label = dataset[0]
    
    assert image.shape == (3, 224, 224), "Vision backbone requires (3, 224, 224)."
    assert image.dtype == torch.float32
    assert label.dtype == torch.long


def test_spectral_dataset_contract(dummy_spectral_data):
    """Verifies the Spectral Dataset returns the exact v4.0 tensors."""
    file_paths = ["/fake/spec/1.fits"]
    redshifts = [0.01]
    labels = [4]
    
    dataset = SpectralDataset(file_paths=file_paths, redshifts=redshifts, labels=labels)
    spectrum, label = dataset[0]
    
    assert spectrum.shape == (4096,), "Spectral MLP requires (4096,)."
    assert spectrum.dtype == torch.float32
    assert label.dtype == torch.long


# =====================================================================
# DATALOADER BATCHING TESTS
# =====================================================================

def test_dataloader_batch_dimensions(dummy_light_curve_data):
    """Verifies the DataLoader correctly batches individual tensors for the GPU."""
    # Create a dummy dataset of 20 stars
    kic_ids = [f"Star_{i}" for i in range(20)]
    labels = [1] * 20
    
    dataset = LightCurveDataset(kic_ids=kic_ids, labels=labels)
    
    # Test our factory with batch_size=16 (Our RTX 4060 constraint)
    dataloaders = get_dataloaders(
        train_dataset=dataset, 
        val_dataset=dataset, 
        batch_size=16, 
        num_workers=0 # Keep at 0 for fast unit testing
    )
    
    train_loader = dataloaders["train"]
    
    # Fetch the very first batch
    batch_flux, batch_mask, batch_labels = next(iter(train_loader))
    
    # Verify the DataLoader prepended the Batch dimension correctly
    assert batch_flux.shape == (16, 8192), "Batch shape mismatch for flux."
    assert batch_mask.shape == (16, 8192), "Batch shape mismatch for mask."
    assert batch_labels.shape == (16,), "Batch shape mismatch for labels."


# =====================================================================
# NPZ DATASET TESTS (Offline, uses in-memory mock NPZ)
# =====================================================================

@pytest.fixture
def mock_npz(tmp_path):
    """Creates a tiny fake .npz archive to test NPZLightCurveDataset offline."""
    n = 50
    npz_file = tmp_path / "mock_light_curves.npz"
    np.savez(
        npz_file,
        flux=np.random.randn(n, 8192).astype(np.float32),
        masks=np.ones((n, 8192), dtype=np.float32),
        labels=np.random.randint(0, 6, size=(n,), dtype=np.int64),
        kic_ids=np.arange(n, dtype=np.int64),
        class_names=np.array(["exoplanet_transit"] * n),
    )
    return str(npz_file)


def test_npz_dataset_contract(mock_npz):
    """Verifies NPZLightCurveDataset returns correct shapes and dtypes."""
    dataset = NPZLightCurveDataset(mock_npz)
    assert len(dataset) == 50

    flux, mask, label = dataset[0]
    assert flux.shape  == (8192,),  f"Expected (8192,), got {flux.shape}"
    assert mask.shape  == (8192,),  f"Expected (8192,), got {mask.shape}"
    assert flux.dtype  == torch.float32
    assert mask.dtype  == torch.bool
    assert label.dtype == torch.long


def test_npz_dataset_index_subsetting(mock_npz):
    """Verifies that the indices parameter correctly subsets the dataset."""
    train_idx = np.arange(35)  # first 35
    val_idx   = np.arange(35, 50)  # last 15

    train_ds = NPZLightCurveDataset(mock_npz, indices=train_idx)
    val_ds   = NPZLightCurveDataset(mock_npz, indices=val_idx)

    assert len(train_ds) == 35
    assert len(val_ds)   == 15


def test_npz_dataloader_batch(mock_npz):
    """Verifies batching from NPZLightCurveDataset produces correct shapes."""
    dataset = NPZLightCurveDataset(mock_npz)
    loader  = create_dataloader(dataset, batch_size=8, is_training=True, num_workers=0)

    batch_flux, batch_mask, batch_labels = next(iter(loader))
    assert batch_flux.shape   == (8, 8192)
    assert batch_mask.shape   == (8, 8192)
    assert batch_labels.shape == (8,)


@pytest.fixture
def mock_spectral_npz(tmp_path):
    """Creates a tiny fake .npz archive to test NPZSpectralDataset offline."""
    n = 40
    npz_file = tmp_path / "mock_spectra.npz"
    np.savez(
        npz_file,
        spectra=np.random.randn(n, 4096).astype(np.float32),
        labels=np.random.randint(0, 6, size=(n,), dtype=np.int64),
    )
    return str(npz_file)


def test_npz_spectral_dataset_contract(mock_spectral_npz):
    """Verifies NPZSpectralDataset returns correct shapes and dtypes."""
    from src.data.dataset import NPZSpectralDataset
    dataset = NPZSpectralDataset(mock_spectral_npz)
    assert len(dataset) == 40

    spectrum, label = dataset[0]
    assert spectrum.shape == (4096,)
    assert spectrum.dtype == torch.float32
    assert label.dtype == torch.long


def test_npz_spectral_dataset_augmentation(mock_spectral_npz):
    """Verifies NPZSpectralDataset correctly applies data augmentation while preserving shapes."""
    from src.data.dataset import NPZSpectralDataset
    dataset = NPZSpectralDataset(mock_spectral_npz, augment=True)
    spectrum, label = dataset[0]
    assert spectrum.shape == (4096,)
    assert spectrum.dtype == torch.float32
    assert label.dtype == torch.long
    assert not torch.isnan(spectrum).any()