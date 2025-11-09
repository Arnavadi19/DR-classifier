"""
Model loading and inference handler
Optimized for AWS Lambda deployment
"""

import os
import logging
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

logger = logging.getLogger(__name__)

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "dr-classifier-models-arnav")
MODEL_KEY = os.getenv("MODEL_KEY", "models/best_model.pth")
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/best_model.pth")

class DRClassifier(nn.Module):
    """
    DR Classification Model - Must match training architecture EXACTLY
    Uses ViT backbone with custom classifier head
    """
    
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, dropout=0.3):
        super().__init__()
        
        # Create backbone (ViT)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0  # Remove default classifier
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            num_features = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            num_features = self.backbone.embed_dim
        else:
            num_features = 768  # Default for ViT-B
        
        # Custom classifier head - EXACT match to your training
        # Based on checkpoint:
        #   0: BatchNorm1d(768)
        #   1: ReLU
        #   2: Linear(768, 128)
        #   3: ReLU
        #   4: Dropout
        #   5: Linear(128, 2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),              # 0: BatchNorm
            nn.ReLU(),                                 # 1: ReLU
            nn.Linear(num_features, 128),              # 2: 768 -> 128
            nn.ReLU(),                                 # 3: ReLU
            nn.Dropout(dropout),                       # 4: Dropout
            nn.Linear(128, num_classes)                # 5: 128 -> 2
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class DRModelHandler:
    """Handler for DR classification model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize model handler
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        requested_device = device.lower()
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if requested_device == "cuda":
                logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model_path = model_path or MODEL_PATH
        self._ensure_model_available()
        self.model = self._load_model(self.model_path)
        self.transform = self._get_transform()
        
        self.class_names = {
            0: "Negative (No/Mild DR)",
            1: "Positive (Moderate+ DR)"
        }
        
        logger.info(f"Model loaded on {self.device}")
    
    def _ensure_model_available(self) -> None:
        """Make sure the model checkpoint exists locally, downloading from S3 if needed."""

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if os.path.exists(self.model_path):
            logger.info("Model checkpoint found locally. Skipping download.")
            return

        logger.info(
            "Downloading model from s3://%s/%s to %s...",
            MODEL_BUCKET,
            MODEL_KEY,
            self.model_path,
        )

        try:
            s3 = boto3.client("s3")
            s3.download_file(MODEL_BUCKET, MODEL_KEY, self.model_path)
            logger.info("Model download complete.")
        except (BotoCoreError, ClientError) as exc:
            logger.error("Failed to download model from S3: %s", exc)
            raise

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        
        # Create model architecture (MUST match training)
        model = DRClassifier(
            model_name='vit_base_patch16_224',
            num_classes=2,
            dropout=0.3
        )
        
        # Load checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Load weights with strict=False to handle BatchNorm buffers
        # BatchNorm's running_mean and running_var will be initialized automatically
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False
        )
        
        # Log any issues (should only be BatchNorm buffers)
        if missing_keys:
            logger.info(f"Missing keys (expected for BatchNorm buffers): {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        model.to(self.device)
        model.eval()  # This will initialize BatchNorm buffers properly
        
        if isinstance(checkpoint, dict):
            logger.info(
                "✓ Loaded checkpoint from epoch %s",
                checkpoint.get("epoch", "unknown")
            )
            best_f1 = checkpoint.get("best_f1")
            if best_f1 is not None:
                logger.info("✓ Best F1 Score: %.4f", best_f1)
        
        return model
    
    def _get_transform(self):
        """Get preprocessing transform (must match training)"""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Preprocessed tensor
        """
        # Convert to numpy
        image_np = np.array(image)
        
        # Apply transforms
        augmented = self.transform(image=image_np)
        image_tensor = augmented['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image: Image.Image) -> dict:
        """
        Make prediction on image
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        image_tensor = self._preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Format results
        result = {
            "prediction": self.class_names[predicted_class],
            "confidence": float(confidence),
            "class_probabilities": {
                "Negative": float(probabilities[0]),
                "Positive": float(probabilities[1])
            },
            "interpretation": self._get_interpretation(predicted_class, confidence)
        }
        
        return result
    
    def _get_interpretation(self, predicted_class: int, confidence: float) -> str:
        """Generate human-readable interpretation"""
        
        if predicted_class == 0:  # Negative
            if confidence >= 0.95:
                return "Very low risk of moderate or worse DR. Regular screening recommended."
            elif confidence >= 0.80:
                return "Low risk of moderate or worse DR. Continue routine monitoring."
            else:
                return "Uncertain classification. Consider ophthalmologist referral for confirmation."
        
        else:  # Positive
            if confidence >= 0.95:
                return "High likelihood of moderate or worse DR. Ophthalmologist referral strongly recommended."
            elif confidence >= 0.80:
                return "Moderate likelihood of moderate or worse DR. Ophthalmologist referral recommended."
            else:
                return "Possible moderate or worse DR detected. Ophthalmologist evaluation advised."