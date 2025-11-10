import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DRBinaryDataset(Dataset):
    """
    Diabetic Retinopathy Binary Classification Dataset
    
    Args:
        csv_path: Path to CSV file with columns [id_code, diagnosis, binary_label]
        image_dir: Root directory containing class folders
        transform: Albumentations transform pipeline
        class_names: Dictionary mapping diagnosis codes to folder names
    """
    
    def __init__(self, csv_path, image_dir, transform=None, class_names=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        self.class_names = class_names or {
            0: 'No_DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferate_DR'
        }
        
        # Verify all images exist
        self._verify_images()
    
    def _verify_images(self):
        """Check if all images in CSV exist on disk"""
        missing = []
        for idx, row in self.df.iterrows():
            img_path = self._get_image_path(row)
            if not img_path.exists():
                missing.append(img_path)
        
        if missing:
            print(f" Warning: {len(missing)} images not found")
            # Remove missing entries
            valid_mask = self.df.apply(
                lambda row: self._get_image_path(row).exists(), axis=1
            )
            self.df = self.df[valid_mask].reset_index(drop=True)
    
    def _get_image_path(self, row):
        """Get full image path from row"""
        original_class = self.class_names[row['diagnosis']]
        return self.image_dir / original_class / f"{row['id_code']}.png"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self._get_image_path(row)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get label
        label = torch.tensor(row['binary_label'], dtype=torch.long)
        
        return image, label
    
    def get_class_distribution(self):
        """Return class distribution"""
        return self.df['binary_label'].value_counts().sort_index().to_dict()


def get_transforms(image_size=224, augment=True):
    """
    Get data augmentation transforms
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentation (True for train, False for val/test)
    
    Returns:
        Albumentations Compose transform
    """
    
    if augment:
        # Training transforms with strong augmentation
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                p=0.6
            ),
            
            # Color/brightness transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(std_range=(0.04, 0.2)),  # Normalized to [0, 1] range
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            
            # Cutout for regularization
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                p=0.3
            ),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    else:
        # Validation/Test transforms (no augmentation)
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def create_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    image_dir,
    batch_size=32,
    num_workers=4,
    image_size=224,
    seed = 42
):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_csv, val_csv, test_csv: Paths to CSV files
        image_dir: Root image directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get transforms
    train_transform = get_transforms(image_size=image_size, augment=True)
    val_transform = get_transforms(image_size=image_size, augment=False)
    
    # Create datasets
    train_dataset = DRBinaryDataset(train_csv, image_dir, train_transform)
    val_dataset = DRBinaryDataset(val_csv, image_dir, val_transform)
    test_dataset = DRBinaryDataset(test_csv, image_dir, val_transform)
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"  Distribution: {train_dataset.get_class_distribution()}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"  Distribution: {val_dataset.get_class_distribution()}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"  Distribution: {test_dataset.get_class_distribution()}")
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Worker initialization function
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Create dataloaders with reproducibility
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,  # Add generator
        worker_init_fn=seed_worker  # Add worker init
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker  # Add worker init
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker  # Add worker init
    )
    
    return train_loader, val_loader, test_loader


# Test the dataset
if __name__ == "__main__":
    # Test loading
    train_csv = "archive/train_binary.csv"
    image_dir = "archive/gaussian_filtered_images"
    
    # Create dataset
    transform = get_transforms(image_size=224, augment=True)
    dataset = DRBinaryDataset(train_csv, image_dir, transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test loading a sample
    image, label = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label.item()}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    batch_images, batch_labels = next(iter(loader))
    print(f"\nBatch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels: {batch_labels}")