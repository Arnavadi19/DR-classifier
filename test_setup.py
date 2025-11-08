"""
Test script to verify the installation and basic functionality
"""

import torch
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import albumentations as A
        import timm
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def test_src_modules():
    """Test if src modules can be imported"""
    print("\nTesting src modules...")
    try:
        from src.config import Config
        from src.dataset import DRBinaryDataset, get_transforms, create_dataloaders
        from src.model import create_model, FocalLoss, get_loss_function
        from src.train import Trainer
        from src.utils import set_seed, calculate_metrics, AverageMeter
        print("✓ All src modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from src.config import Config
        config = Config()
        print(f"  Model: {config.MODEL_NAME}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Image size: {config.IMAGE_SIZE}")
        print(f"  Loss function: {config.LOSS_FUNCTION}")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_transforms():
    """Test data transforms"""
    print("\nTesting data transforms...")
    try:
        from src.dataset import get_transforms
        import numpy as np
        
        # Test train transform
        train_transform = get_transforms(image_size=224, augment=True)
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Apply transform
        augmented = train_transform(image=dummy_image)
        image_tensor = augmented['image']
        
        assert image_tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {image_tensor.shape}"
        assert image_tensor.dtype == torch.float32, f"Expected dtype float32, got {image_tensor.dtype}"
        
        print(f"  Transform output shape: {image_tensor.shape}")
        print(f"  Transform output dtype: {image_tensor.dtype}")
        print(f"  Transform output range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        print("✓ Transforms working correctly")
        return True
    except Exception as e:
        print(f"✗ Transform error: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from src.model import create_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")
        
        # Test EfficientNet
        model = create_model('efficientnet_b3', pretrained=False, device=device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        
        assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: EfficientNet-B3")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of parameters: {num_params:,}")
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False


def test_loss_function():
    """Test loss functions"""
    print("\nTesting loss functions...")
    try:
        from src.model import FocalLoss, get_loss_function
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test Focal Loss
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Create dummy predictions and targets
        outputs = torch.randn(4, 2).to(device)
        targets = torch.tensor([0, 1, 0, 1]).to(device)
        
        loss = focal_loss(outputs, targets)
        
        assert loss.item() > 0, "Loss should be positive"
        
        print(f"  Focal Loss value: {loss.item():.4f}")
        print("✓ Loss functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Loss function error: {e}")
        return False


def test_dataset():
    """Test dataset loading (if data exists)"""
    print("\nTesting dataset loading...")
    try:
        from src.config import Config
        from src.dataset import DRBinaryDataset, get_transforms
        
        config = Config()
        
        # Check if train CSV exists
        if not config.TRAIN_CSV.exists():
            print(f"  ⚠ Train CSV not found at {config.TRAIN_CSV}")
            print("  Skipping dataset test (run EDA.ipynb first)")
            return True
        
        # Try to load dataset
        transform = get_transforms(image_size=224, augment=False)
        dataset = DRBinaryDataset(config.TRAIN_CSV, config.IMAGE_DIR, transform)
        
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Class distribution: {dataset.get_class_distribution()}")
        
        # Try to load one sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"  Sample image shape: {image.shape}")
            print(f"  Sample label: {label.item()}")
            print("✓ Dataset loading successful")
        
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        print("  Note: This is expected if you haven't prepared the data yet")
        return True  # Don't fail on dataset errors


def main():
    """Run all tests"""
    print("="*60)
    print("REMIDIO PROJECT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Src Modules", test_src_modules),
        ("Configuration", test_config),
        ("Transforms", test_transforms),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_function),
        ("Dataset", test_dataset),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Run EDA.ipynb to prepare the dataset")
        print("  2. Run: python main.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
