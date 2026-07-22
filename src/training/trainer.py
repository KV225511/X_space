import torch
from tqdm import tqdm
# import wandb  # We will uncomment this when you are ready to log graphs

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch):
        self.model.train() # Put the model into training mode
        running_loss = 0.0
        
        # Wrap the dataloader in a progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 1. Wipe the memory
            self.optimizer.zero_grad()
            
            # 2. Push data through the model
            outputs = self.model(inputs)
            
            # 3. Calculate the penalty
            loss = self.criterion(outputs, labels)
            
            # 4 & 5. Figure out the fixes and apply them
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return running_loss / len(self.train_loader)

    def val_epoch(self, epoch):
        self.model.eval() # Put the model into strict testing mode (no learning allowed)
        running_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad(): # Disable calculus to save memory and speed up testing
            for batch in pbar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
        return running_loss / len(self.val_loader)

    def train(self, num_epochs):
        print(f"Starting X_space training on {self.device} for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            print(f"Epoch {epoch} Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")