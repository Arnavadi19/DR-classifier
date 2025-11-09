# Diabetic Retinopathy Binary Classification

A production-ready deep learning pipeline for binary classification of Diabetic Retinopathy (DR) fundus images using PyTorch. This system classifies retinal images into two clinically relevant categories:

- **Negative**: No DR and Mild DR (non-referrable cases)
- **Positive**: Moderate, Severe, and Proliferative DR (referrable cases requiring specialist evaluation)

The project includes a complete machine learning workflow from training to deployment, featuring a FastAPI-based REST API for real-time inference.

## Project Structure

```
remidio/
├── api/                              # Production API
│   ├── app.py                        # FastAPI application
│   ├── model_handler.py              # Model loading and inference
│   ├── Dockerfile                    # Container configuration
│   ├── requirements.txt              # API dependencies
│   └── best_model.pth                # Trained model checkpoint
├── src/                              # Core ML pipeline
│   ├── __init__.py
│   ├── config.py                     # Training configuration
│   ├── dataset.py                    # Data loading and augmentation
│   ├── model.py                      # Model architectures
│   ├── train.py                      # Training logic
│   ├── utils.py                      # Utility functions
│   └── results_summary.py            # Evaluation metrics
├── archive/                          # Dataset storage
│   ├── gaussian_filtered_images/     # Preprocessed images (5 classes)
│   ├── train_binary.csv              # Training split (75%)
│   ├── val_binary.csv                # Validation split (10%)
│   ├── test_binary.csv               # Test split (15%)
│   └── dataset_stats.json            # Dataset statistics
├── test_images/                      # Separated test images
│   ├── Negative/                     # Ground truth negative cases
│   └── Positive/                     # Ground truth positive cases
├── checkpoints/                      # Model checkpoints
│   └── best_model.pth                # Best performing model
├── logs/                             # Training logs and visualizations
├── results/                          # Evaluation results
├── main.py                           # Training script
├── evaluate.py                       # Model evaluation
├── inference.py                      # Command-line inference
├── test_setup.py                     # Environment verification
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB RAM minimum (16GB recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/Arnavadi19/DR-classifier.git
cd remidio
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations opencv-python-headless pandas numpy scikit-learn matplotlib seaborn tqdm
pip install fastapi uvicorn python-multipart
```

### Dataset Preparation

Organize your dataset in the following structure:

```text
archive/gaussian_filtered_images/
├── No_DR/
├── Mild/
├── Moderate/
├── Severe/
└── Proliferate_DR/
```

The images should be preprocessed using the Ben Graham filter (Gaussian unsharp masking) to enhance microaneurysms and other DR lesions.

### Training

Train the model using the default configuration:

```bash
python main.py
```

The training script will:

- Load and preprocess the dataset with stratified splits
- Apply comprehensive data augmentation
- Train using mixed precision (AMP) for faster convergence
- Save the best model based on validation F1 score
- Generate training history plots and logs

### Evaluation

Evaluate the trained model on the held-out test set:

```bash
python evaluate.py
```

This generates comprehensive metrics including:

- Classification report (Precision, Recall, F1-Score)
- Confusion matrix visualization
- ROC curve and AUC score
- Precision-Recall curve
- Clinical interpretation of results

Results are saved in the `results/` directory.

## REST API

### Overview

The project includes a production-ready FastAPI application for real-time DR classification. The API provides RESTful endpoints for single and batch image inference.

### Running the API Locally

Navigate to the API directory:

```bash
cd api
```

Ensure the trained model is present:

```bash
cp ../checkpoints/best_model.pth .
```

Start the server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`. Interactive documentation is accessible at `http://localhost:8000/docs`.

### API Endpoints

#### Health Check

```http
GET /
```

Returns server status and model loading state.

**Response:**

```json
{
  "message": "DR Classification API is running",
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### Single Image Prediction

```http
POST /predict
```

**Request:**

- Content-Type: `multipart/form-data`
- Body: Form field `file` containing the image (PNG, JPG, JPEG)

**Example:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@fundus_image.png"
```

**Response:**

```json
{
  "prediction": "Positive (Moderate+ DR)",
  "confidence": 0.923,
  "class_probabilities": {
    "Negative": 0.077,
    "Positive": 0.923
  },
  "interpretation": "High likelihood of moderate or worse DR. Ophthalmologist referral strongly recommended."
}
```

#### Batch Prediction

```http
POST /batch_predict
```

**Request:**

- Content-Type: `multipart/form-data`
- Body: Multiple form fields named `files` (max 10 images)

**Example:**

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -F "files=@image1.png" \
  -F "files=@image2.png"
```

**Response:**

```json
{
  "results": [
    {
      "filename": "image1.png",
      "prediction": "Negative (No/Mild DR)",
      "confidence": 0.891,
      "class_probabilities": {
        "Negative": 0.891,
        "Positive": 0.109
      },
      "interpretation": "Very low risk of moderate or worse DR. Regular screening recommended."
    },
    {
      "filename": "image2.png",
      "prediction": "Positive (Moderate+ DR)",
      "confidence": 0.945,
      "class_probabilities": {
        "Negative": 0.055,
        "Positive": 0.945
      },
      "interpretation": "High likelihood of moderate or worse DR. Ophthalmologist referral strongly recommended."
    }
  ]
}
```

### Docker Deployment

Build the Docker image:

```bash
cd api
docker build -t dr-classifier-api .
```

Run the container:

```bash
docker run -p 8000:8000 dr-classifier-api
```

### AWS Lambda Deployment

The API is designed to be deployed on AWS Lambda using container images:

Push the Docker image to Amazon ECR:

```bash
aws ecr create-repository --repository-name dr-classifier
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag dr-classifier-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/dr-classifier:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/dr-classifier:latest
```

Then:

- Create a Lambda function from the ECR image via AWS Console or CLI
- Configure function URL for public access

The serverless deployment provides cost-effective inference with automatic scaling.

## Command-Line Inference

For single-image predictions without the API:

```bash
python inference.py --image path/to/fundus_image.png
```

Programmatic usage:

```python
from inference import DRPredictor
from src.config import Config

config = Config()
predictor = DRPredictor('checkpoints/best_model.pth', config)

result = predictor.predict('path/to/image.png')
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Visualize with CAM overlay
predictor.visualize_prediction('path/to/image.png', save_path='output.png')
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

## Data Augmentation

Strong augmentation pipeline for medical images:

- Geometric: Flips, rotations, shifts, scaling
- Color: Brightness/contrast adjustment, HSV
- Noise: Gaussian noise, motion blur, Gaussian blur
- Regularization: CoarseDropout (Cutout)

## Training Features

- Focal Loss for class imbalance handling
- Mixed Precision Training (AMP) for faster training
- Cosine Annealing with Warm Restarts scheduler
- Early Stopping to prevent overfitting
- Automatic checkpoint saving (best F1 score)
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)

## Results

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

Option A: Weighted Sampling

```python
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
class_counts = [2200, 1460]  # [Negative, Positive]
class_weights = [1/c for c in class_counts]
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

Option B: Weighted Loss

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
  author={Arnav Aditya},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Arnavadi19/DR-classifier}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [arnavdt@gmail.com].

## 🙏 Acknowledgments

- EfficientNet implementation from [timm](https://github.com/rwightman/pytorch-image-models)
- Data augmentation using [Albumentations](https://albumentations.ai/)
- Inspired by various Kaggle kernels on Diabetic Retinopathy detection
