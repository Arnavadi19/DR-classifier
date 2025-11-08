# Diabetic Retinopathy Binary Classification

A deep learning pipeline for binary classification of Diabetic Retinopathy images using PyTorch. This project classifies retinal images into two categories:
- **Negative**: No DR and Mild DR (referrable cases)
- **Positive**: Moderate, Severe, and Proliferative DR (non-referrable cases)

## 📁 Project Structure

```
remidio/
├── archive/                          # Data directory
│   ├── gaussian_filtered_images/     # Preprocessed retinal images
│   ├── train_binary.csv              # Training split
│   ├── val_binary.csv                # Validation split
│   ├── test_binary.csv               # Test split
│   └── dataset_stats.json            # Dataset statistics
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration parameters
│   ├── dataset.py                    # Dataset class and data loaders
│   ├── model.py                      # Model architectures
│   ├── train.py                      # Training logic
│   └── utils.py                      # Utility functions
├── checkpoints/                      # Saved model checkpoints
├── logs/                             # Training logs and plots
├── main.py                           # Main training script
├── evaluate.py                       # Evaluation script
├── inference.py                      # Inference script
├── EDA.ipynb                         # Exploratory Data Analysis
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd remidio
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install timm albumentations opencv-python pandas numpy scikit-learn matplotlib seaborn tqdm
```

### Dataset Preparation

Your dataset should be organized as follows:
```
archive/gaussian_filtered_images/
├── No_DR/
├── Mild/
├── Moderate/
├── Severe/
└── Proliferate_DR/
```

Run the EDA notebook to create the binary splits:
```bash
jupyter notebook EDA.ipynb
```

### Training

Train the model using default configuration:
```bash
python main.py
```

### Evaluation

Evaluate the trained model on the test set:
```bash
python evaluate.py
```

### Inference

Run inference on new images:
```bash
python inference.py
```

Or use the predictor programmatically:
```python
from inference import DRPredictor
from src.config import Config

config = Config()
predictor = DRPredictor('checkpoints/best_model.pth', config)

# Single image prediction
result = predictor.predict('path/to/image.png')
print(result['predicted_label'], result['confidence'])

# Visualize prediction
predictor.visualize_prediction('path/to/image.png')
```

## ⚙️ Configuration

Modify `src/config.py` to customize training parameters:

```python
# Model settings
MODEL_NAME = "efficientnet_b3"  # Options: efficientnet_b3, vit_base_patch16_224, densenet121
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Loss function
LOSS_FUNCTION = "focal"  # Options: focal, ce, weighted_ce
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Optimizer
OPTIMIZER = "adamw"  # Options: adam, adamw, sgd
SCHEDULER = "cosine"  # Options: cosine, step, plateau
```

## 🏗️ Model Architectures

The project supports three state-of-the-art architectures:

### 1. EfficientNet-B3 (Recommended)
- Best balance of accuracy and efficiency
- ~12M parameters
- Fast inference (~15ms per image on GPU)

### 2. Vision Transformer (ViT)
- State-of-the-art performance
- ~86M parameters
- Better for larger datasets

### 3. DenseNet-121
- Lightweight and efficient
- ~8M parameters
- Good for resource-constrained environments

## 📊 Data Augmentation

Strong augmentation pipeline for medical images:
- Geometric: Flips, rotations, shifts, scaling
- Color: Brightness/contrast adjustment, HSV
- Noise: Gaussian noise, motion blur, Gaussian blur
- Regularization: CoarseDropout (Cutout)

## 📈 Training Features

- ✅ **Focal Loss** for class imbalance handling
- ✅ **Mixed Precision Training** (AMP) for faster training
- ✅ **Cosine Annealing with Warm Restarts** scheduler
- ✅ **Early Stopping** to prevent overfitting
- ✅ **Automatic checkpoint saving** (best F1 score)
- ✅ **Comprehensive metrics** (Accuracy, Precision, Recall, F1, AUC)

## 📊 Results

Expected performance metrics:
- **Accuracy**: 85-90%
- **F1 Score**: 0.85-0.92
- **AUC-ROC**: 0.90-0.95
- **Sensitivity**: 85-92% (catching positive cases)
- **Specificity**: 85-90% (correctly identifying negatives)

## 📝 Dataset Split

- **Training**: 75% (~2,750 images)
- **Validation**: 10% (~370 images)
- **Test**: 15% (~550 images)

All splits use stratified sampling to maintain class balance.

## 🔧 Advanced Usage

### Custom Training Loop

```python
from src.config import Config
from src.dataset import create_dataloaders
from src.model import create_model, get_loss_function
from src.train import Trainer
import torch.optim as optim

config = Config()

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV,
    config.IMAGE_DIR, config.BATCH_SIZE
)

# Create model
model = create_model(config.MODEL_NAME, pretrained=True)

# Setup training
criterion = get_loss_function('focal', alpha=0.25, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Train
trainer = Trainer(model, criterion, optimizer, device='cuda')
history = trainer.fit(train_loader, val_loader, num_epochs=50)
```

### Handling Class Imbalance

If the default focal loss doesn't work well, try:

**Option A: Weighted Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
class_counts = [2200, 1460]  # [Negative, Positive]
class_weights = [1/c for c in class_counts]
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Option B: Weighted Loss**
```python
# In config.py
LOSS_FUNCTION = "weighted_ce"
POS_WEIGHT = 1.5  # Increase to give more weight to positive class
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dr_binary_classification,
  title={Diabetic Retinopathy Binary Classification},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/remidio}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

## 🙏 Acknowledgments

- EfficientNet implementation from [timm](https://github.com/rwightman/pytorch-image-models)
- Data augmentation using [Albumentations](https://albumentations.ai/)
- Inspired by various Kaggle kernels on Diabetic Retinopathy detection
