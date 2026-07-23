import argparse
import torch
import torch.optim as optim
import yaml

# Import Yash's Data
# from src.data.dataloaders import get_dataloaders 

# Import Krissh's Models
from src.models.transformer_encoder import LightCurveEncoder
from src.models.image_encoder import ImageEncoder
from src.models.spectral_encoder import SpectralEncoder

# Import Your Engine and Math
from src.training.trainer import Trainer
from src.training.losses import FocalLoss

def main():
    # 1. Read commands from the terminal
    parser = argparse.ArgumentParser(description="Train AstroFoundation Encoders")
    parser.add_argument('--encoder', type=str, required=True, choices=['light_curve', 'image', 'spectral'], help="Which encoder to train")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training loops")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Booting up {args.encoder} training on {device} ---")

    # 2. Load Krissh's configurations
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 3. Select the correct model based on terminal input
    if args.encoder == 'light_curve':
        model = LightCurveEncoder(config)
    elif args.encoder == 'image':
        model = ImageEncoder(config)
    elif args.encoder == 'spectral':
        model = SpectralEncoder(config)

    # 4. Set up Yash's DataLoaders (Placeholder until Yash finishes Phase 2 dataloaders.py)
    # train_loader, val_loader = get_dataloaders(args.encoder)
    train_loader, val_loader = [], [] # Dummy placeholders to prevent crashing

    # 5. Set up your Math and Optimizer
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6. Ignite the Engine
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Uncomment the line below once Yash finishes dataloaders.py!
    # trainer.train(num_epochs=args.epochs)
    
    print("✓ Script successfully wired. Waiting for DataLoaders to execute training loop.")

if __name__ == "__main__":
    main()