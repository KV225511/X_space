import torch
from tqdm import tqdm
# import wandb  # Uncomment + `wandb login` to enable experiment tracking

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def _unpack_batch(self, batch):
        """
        Handles both batch shapes coming from our three Dataset classes:
          - 3-item: (flux, mask, label)  <- NPZLightCurveDataset / LightCurveDataset
          - 2-item: (inputs, label)       <- FITSDataset / SpectralDataset
        Returns (inputs, labels, extra_kwargs).
        extra_kwargs carries {'mask': mask} for the LightCurveEncoder
        so the attention mask is passed through correctly.
        """
        if len(batch) == 3:
            inputs, mask, labels = batch
            inputs = inputs.to(self.device)
            mask   = mask.to(self.device)
            labels = labels.to(self.device)
            return inputs, labels, {"mask": mask}
        else:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            return inputs, labels, {}

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            inputs, labels, extra_kwargs = self._unpack_batch(batch)

            # 1. Wipe accumulated gradients
            self.optimizer.zero_grad()

            # 2. Forward pass (pass mask kwarg if present)
            outputs = self.model(inputs, **extra_kwargs)

            # 3. Compute loss
            loss = self.criterion(outputs, labels)

            # 4. Backprop and update
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return running_loss / len(self.train_loader)

    def val_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]  ")

        with torch.no_grad():
            for batch in pbar:
                inputs, labels, extra_kwargs = self._unpack_batch(batch)

                outputs = self.model(inputs, **extra_kwargs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Track accuracy
                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        return running_loss / len(self.val_loader), acc

    def train(self, num_epochs):
        print(f"Starting AstroFoundation training on {self.device} for {num_epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_loss         = self.train_epoch(epoch)
            val_loss, val_acc  = self.val_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            marker = "  <- [BEST]" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Train Loss: {train_loss:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  |  "
                f"Val Acc: {val_acc:.3f}  |  "
                f"LR: {current_lr:.2e}{marker}"
            )