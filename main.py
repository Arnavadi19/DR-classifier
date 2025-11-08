import torch
import torch.optim as optim
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.config import Config
from src.dataset import create_dataloaders
from src.model import create_model, get_loss_function
from src.train import Trainer
from src.utils import set_seed, plot_confusion_matrix


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['train_auc'], label='Train AUC', marker='o')
    axes[1, 1].plot(history['val_auc'], label='Val AUC', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Training and Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function"""
    
    # Load configuration
    config = Config()
    
    print("\n" + "="*60)
    print("DIABETIC RETINOPATHY BINARY CLASSIFICATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model:        {config.MODEL_NAME}")
    print(f"  Batch Size:   {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Epochs:       {config.NUM_EPOCHS}")
    print(f"  Image Size:   {config.IMAGE_SIZE}")
    print(f"  Loss:         {config.LOSS_FUNCTION}")
    print(f"  Optimizer:    {config.OPTIMIZER}")
    print(f"  Scheduler:    {config.SCHEDULER}")
    
    # Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=config.TRAIN_CSV,
        val_csv=config.VAL_CSV,
        test_csv=config.TEST_CSV,
        image_dir=config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        seed=config.RANDOM_SEED
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        dropout=config.DROPOUT,
        device=device
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = get_loss_function(
        loss_name=config.LOSS_FUNCTION,
        alpha=config.FOCAL_ALPHA,
        gamma=config.FOCAL_GAMMA
    )
    
    # Create optimizer
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Create scheduler
    if config.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_MULT
        )
    elif config.SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif config.SCHEDULER == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=config.USE_AMP
    )
    
    # Train model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        checkpoint_dir=config.CHECKPOINT_DIR,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    # Save training history
    history_path = config.LOG_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")
    
    # Plot training history
    plot_training_history(
        history,
        save_path=config.LOG_DIR / 'training_curves.png'
    )
    
    # Load best model
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(config.CHECKPOINT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Best model from epoch {checkpoint['epoch']} (F1: {checkpoint['best_f1']:.4f})")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        classes=['Negative (No/Mild DR)', 'Positive (Moderate+ DR)'],
        save_path=config.LOG_DIR / 'confusion_matrix.png'
    )
    
    # Save test results
    test_results_path = config.LOG_DIR / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_results['metrics'], f, indent=2)
    print(f"\n✓ Test results saved to {test_results_path}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
