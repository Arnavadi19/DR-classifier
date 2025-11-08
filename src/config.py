from pathlib import Path


class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = Path("archive")
    IMAGE_DIR = DATA_DIR / "gaussian_filtered_images"
    TRAIN_CSV = DATA_DIR / "train_binary.csv"
    VAL_CSV = DATA_DIR / "val_binary.csv"
    TEST_CSV = DATA_DIR / "test_binary.csv"
    
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    
    # Create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # Model
    MODEL_NAME = "vit_base_patch16_224"  # Options: efficientnet_b3, vit_base_patch16_224, densenet121
    NUM_CLASSES = 2  # Binary classification
    PRETRAINED = True
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Image
    IMAGE_SIZE = 224
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Optimization
    OPTIMIZER = "sgd"  # Options: adam, adamw, sgd
    SCHEDULER = "cosine"  # Options: cosine, step, plateau
    T_0 = 10  # For CosineAnnealingWarmRestarts
    T_MULT = 2
    
    # Loss
    LOSS_FUNCTION = "weighted_ce"  # Options: ce, focal, weighted_ce
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Regularization
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.1
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Metrics
    METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
    
    # Device
    DEVICE = "cuda"  # Will be auto-detected
    
    # Mixed precision
    USE_AMP = True  # Automatic Mixed Precision
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_BEST_ONLY = True
    
    # Seed
    RANDOM_SEED = 42
    
    def __repr__(self):
        return f"Config(model={self.MODEL_NAME}, batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE})"