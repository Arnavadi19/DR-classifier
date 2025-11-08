import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .utils import AverageMeter, calculate_metrics


class Trainer:
    """
    Trainer class for binary classification
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        use_amp=True
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        
        # For mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Track best model
        self.best_f1 = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate(self, val_loader, epoch):
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        checkpoint_dir,
        early_stopping_patience=10
    ):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Dictionary of training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        patience_counter = 0
        
        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['train_auc'].append(train_metrics['auc'])
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auc'].append(val_metrics['auc'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | "
                  f"AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                
                print(f"✓ New best F1: {self.best_f1:.4f} (saved to {checkpoint_path})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                print(f"  Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
                break
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best F1 Score: {self.best_f1:.4f} (Epoch {self.best_epoch})")
        
        return history
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test set
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of test metrics and predictions
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        
        pbar = tqdm(test_loader, desc="Testing")
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        print("\nTest Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        }
