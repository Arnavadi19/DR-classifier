import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.config import Config
from src.model import create_model
from src.dataset import get_transforms


class DRPredictor:
    """
    Predictor class for inference on DR images
    
    Args:
        model_path: Path to trained model checkpoint
        config: Configuration object
        device: Device to run inference on
    """
    
    def __init__(self, model_path, config, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = create_model(
            model_name=config.MODEL_NAME,
            pretrained=False,
            dropout=config.DROPOUT,
            device=self.device
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load transforms
        self.transform = get_transforms(image_size=config.IMAGE_SIZE, augment=False)
        
        # Class names
        self.class_names = {
            0: 'Negative (No DR / Mild DR)',
            1: 'Positive (Moderate / Severe / PDR)'
        }
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for inference
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    
    def predict(self, image_path, return_probs=True):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
        
        result = {
            'predicted_class': pred_class,
            'predicted_label': self.class_names[pred_class],
            'confidence': float(probs[pred_class]),
            'image_path': str(image_path)
        }
        
        if return_probs:
            result['class_probabilities'] = {
                self.class_names[i]: float(probs[i])
                for i in range(len(probs))
            }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction result
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Load image
        image = Image.open(image_path)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Add prediction text
        pred_label = result['predicted_label']
        confidence = result['confidence']
        color = 'green' if result['predicted_class'] == 0 else 'red'
        
        title = f"Prediction: {pred_label}\nConfidence: {confidence:.2%}"
        ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=20)
        
        # Add probability bars
        probs = result['class_probabilities']
        prob_text = "\n".join([f"{k}: {v:.2%}" for k, v in probs.items()])
        ax.text(0.02, 0.98, prob_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return result


def main():
    """Example usage"""
    config = Config()
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running main.py")
        return
    
    # Initialize predictor
    predictor = DRPredictor(model_path, config)
    
    # Example: predict on a single image
    print("\n" + "="*60)
    print("INFERENCE EXAMPLE")
    print("="*60)
    
    # You can change this to any image path
    example_image = config.IMAGE_DIR / "No_DR" / "10_left.png"
    
    if example_image.exists():
        print(f"\nMaking prediction on: {example_image}")
        result = predictor.predict(example_image)
        
        print("\nPrediction Result:")
        print(f"  Class: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("\n  Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"    {class_name}: {prob:.2%}")
        
        # Visualize
        predictor.visualize_prediction(
            example_image,
            save_path=config.LOG_DIR / 'inference_example.png'
        )
    else:
        print(f"\nExample image not found: {example_image}")
        print("Please provide a valid image path.")
    
    # Example: batch prediction
    print("\n" + "="*60)
    print("BATCH INFERENCE EXAMPLE")
    print("="*60)
    
    # Get a few sample images
    sample_images = []
    for class_folder in ["No_DR", "Moderate", "Severe"]:
        class_path = config.IMAGE_DIR / class_folder
        if class_path.exists():
            images = list(class_path.glob("*.png"))[:2]
            sample_images.extend(images)
    
    if sample_images:
        print(f"\nPredicting on {len(sample_images)} images...")
        results = predictor.predict_batch(sample_images)
        
        print("\nBatch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {Path(result['image_path']).name}")
            print(f"   Prediction: {result['predicted_label']}")
            print(f"   Confidence: {result['confidence']:.2%}")
    else:
        print("\nNo sample images found for batch inference.")


if __name__ == "__main__":
    main()
