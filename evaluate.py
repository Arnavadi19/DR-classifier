import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)

from src.config import Config
from src.dataset import create_dataloaders
from src.model import create_model
from src.utils import plot_confusion_matrix, set_seed
from src.results_summary import analyze_results


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model_path, config):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to saved model checkpoint
        config: Configuration object
    """
    set_seed(config.RANDOM_SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        train_csv=config.TRAIN_CSV,
        val_csv=config.VAL_CSV,
        test_csv=config.TEST_CSV,
        image_dir=config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        seed = config.RANDOM_SEED
    )
    
    # Load model
    print("\nLoading model...")
    model = create_model(
        model_name=config.MODEL_NAME,
        pretrained=False,
        dropout=config.DROPOUT,
        device=device
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    
    # Inference
    print("\nRunning inference on test set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Use comprehensive results analyzer
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION RESULTS")
    print("="*80)
    
    metrics, summary_df = analyze_results(
        y_true=all_labels,
        y_pred=all_preds,
        y_probs=all_probs,
        save_dir=results_dir
    )
    
    # Plot confusion matrix
    class_names = ['Negative (No/Mild DR)', 'Positive (Moderate+ DR)']
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        classes=class_names,
        save_path=results_dir / 'confusion_matrix.png'
    )
    
    # Plot ROC curve
    plot_roc_curve(
        all_labels, 
        all_probs,
        save_path=results_dir / 'roc_curve.png'
    )
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(
        all_labels,
        all_probs,
        save_path=results_dir / 'pr_curve.png'
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("✓ All results saved to: results/")
    print("  ├─ metrics_summary.csv       (All metrics in table format)")
    print("  ├─ confusion_matrix.png      (Visual confusion matrix)")
    print("  ├─ roc_curve.png             (ROC curve)")
    print("  └─ pr_curve.png              (Precision-Recall curve)")
    print("="*80)
    
    return metrics, summary_df


def main():
    config = Config()
    model_path = config.CHECKPOINT_DIR / 'best_model.pth'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running main.py")
        return
    
    evaluate_model(model_path, config)


if __name__ == "__main__":
    main()
