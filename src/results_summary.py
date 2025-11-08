"""
Comprehensive results analysis and visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path


class ResultsAnalyzer:
    """Comprehensive results analysis and visualization"""
    
    def __init__(self, y_true, y_pred, y_probs, save_dir="results"):
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (for positive class)
            save_dir: Directory to save results
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_probs = np.array(y_probs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Calculate confusion matrix elements
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        self.tn, self.fp, self.fn, self.tp = cm.ravel()
        
    def calculate_all_metrics(self):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {
            # Basic metrics
            'Accuracy': accuracy_score(self.y_true, self.y_pred),
            'Precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'Recall (Sensitivity)': recall_score(self.y_true, self.y_pred, zero_division=0),
            'Specificity': self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0,
            'F1 Score': f1_score(self.y_true, self.y_pred, zero_division=0),
            'AUROC': roc_auc_score(self.y_true, self.y_probs),
            
            # Additional metrics
            'NPV (Negative Predictive Value)': self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0,
            'PPV (Positive Predictive Value)': self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0,
            
            # Confusion matrix elements
            'True Negatives': int(self.tn),
            'False Positives': int(self.fp),
            'False Negatives': int(self.fn),
            'True Positives': int(self.tp),
            
            # Sample counts
            'Total Samples': len(self.y_true),
            'Positive Samples': int(self.y_true.sum()),
            'Negative Samples': int(len(self.y_true) - self.y_true.sum()),
        }
        
        return metrics
    
    def print_summary(self):
        """Print comprehensive results summary"""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Section 1: Dataset Overview
        print("\n" + "─"*80)
        print("DATASET OVERVIEW")
        print("─"*80)
        print(f"Total Test Samples:     {metrics['Total Samples']}")
        print(f"  ├─ Negative (No/Mild DR):     {metrics['Negative Samples']} samples")
        print(f"  └─ Positive (Moderate+ DR):   {metrics['Positive Samples']} samples")
        neg_pct = metrics['Negative Samples']/(metrics['Negative Samples']+metrics['Positive Samples'])*100
        pos_pct = metrics['Positive Samples']/(metrics['Negative Samples']+metrics['Positive Samples'])*100
        print(f"Class Balance: {neg_pct:.1f}% Negative, {pos_pct:.1f}% Positive")
        
        # Section 2: Core Performance Metrics
        print("\n" + "─"*80)
        print("CORE PERFORMANCE METRICS")
        print("─"*80)
        print(f"Accuracy:               {metrics['Accuracy']*100:.2f}%")
        print(f"AUROC:                  {metrics['AUROC']*100:.2f}%")
        print(f"F1 Score:               {metrics['F1 Score']*100:.2f}%")
        print()
        print(f"Precision (PPV):        {metrics['Precision']*100:.2f}%  ← Of predicted positive, how many are correct?")
        print(f"Recall (Sensitivity):   {metrics['Recall (Sensitivity)']*100:.2f}%  ← Of actual positive, how many detected?")
        print(f"Specificity:            {metrics['Specificity']*100:.2f}%  ← Of actual negative, how many detected?")
        print(f"NPV:                    {metrics['NPV (Negative Predictive Value)']*100:.2f}%  ← Of predicted negative, how many are correct?")
        
        # Section 3: Confusion Matrix
        print("\n" + "─"*80)
        print("CONFUSION MATRIX BREAKDOWN")
        print("─"*80)
        print(f"True Negatives  (TN):   {metrics['True Negatives']:4d}  ✓ Correctly identified as Negative")
        print(f"False Positives (FP):   {metrics['False Positives']:4d}  ✗ Wrongly identified as Positive")
        print(f"False Negatives (FN):   {metrics['False Negatives']:4d}  ✗ Wrongly identified as Negative")
        print(f"True Positives  (TP):   {metrics['True Positives']:4d}  ✓ Correctly identified as Positive")
        
        # Section 4: Clinical Interpretation
        print("\n" + "─"*80)
        print("CLINICAL INTERPRETATION")
        print("─"*80)
        
        # Calculate error rates
        total = metrics['Total Samples']
        fp_rate = (metrics['False Positives'] / total) * 100
        fn_rate = (metrics['False Negatives'] / total) * 100
        
        print(f"Model Performance Summary:")
        print(f"  ├─ Overall Accuracy: {metrics['Accuracy']*100:.1f}% ({int(self.tp + self.tn)}/{total} correct)")
        print(f"  ├─ Correctly identified {metrics['True Positives']} out of {metrics['Positive Samples']} DR cases")
        print(f"  └─ Correctly identified {metrics['True Negatives']} out of {metrics['Negative Samples']} healthy cases")
        print()
        print(f"Error Analysis:")
        print(f"  ├─ False Positives: {metrics['False Positives']} ({fp_rate:.1f}%) - Healthy patients flagged for referral")
        print(f"  │  → Impact: Unnecessary specialist visits, patient anxiety")
        print(f"  └─ False Negatives: {metrics['False Negatives']} ({fn_rate:.1f}%) - DR cases missed")
        print(f"     → Impact: Delayed treatment, potential vision loss")
        
        # Section 5: Model Quality Assessment
        print("\n" + "─"*80)
        print(" MODEL QUALITY ASSESSMENT")
        print("─"*80)
        
        # Assess performance
        acc = metrics['Accuracy']
        auc = metrics['AUROC']
        sens = metrics['Recall (Sensitivity)']
        spec = metrics['Specificity']
        
        ratings = []
        if acc >= 0.90: ratings.append("✅ Excellent Accuracy")
        elif acc >= 0.80: ratings.append("✓ Good Accuracy")
        else: ratings.append("Moderate Accuracy")
        
        if auc >= 0.90: ratings.append("✅ Excellent Discrimination (AUROC)")
        elif auc >= 0.80: ratings.append("✓ Good Discrimination (AUROC)")
        else: ratings.append("Moderate Discrimination (AUROC)")
        
        if sens >= 0.90: ratings.append("✅ Excellent at Detecting DR")
        elif sens >= 0.80: ratings.append("✓ Good at Detecting DR")
        else: ratings.append("May miss some DR cases")
        
        if spec >= 0.90: ratings.append("✅ Excellent at Identifying Healthy")
        elif spec >= 0.80: ratings.append("✓ Good at Identifying Healthy")
        else: ratings.append("May over-refer healthy patients")
        
        for rating in ratings:
            print(f"  {rating}")
        
        
        print("\n" + "="*80)
        
        return metrics
    
    def create_summary_table(self, save=True):
        """Create a formatted summary table"""
        metrics = self.calculate_all_metrics()
        
        # Create DataFrame
        data = {
            'Metric': [
                'Accuracy',
                'AUROC',
                'F1 Score',
                'Precision (PPV)',
                'Recall (Sensitivity)',
                'Specificity',
                'NPV',
                '',  # Separator
                'True Positives',
                'True Negatives',
                'False Positives',
                'False Negatives',
            ],
            'Value': [
                f"{metrics['Accuracy']*100:.2f}%",
                f"{metrics['AUROC']*100:.2f}%",
                f"{metrics['F1 Score']*100:.2f}%",
                f"{metrics['Precision']*100:.2f}%",
                f"{metrics['Recall (Sensitivity)']*100:.2f}%",
                f"{metrics['Specificity']*100:.2f}%",
                f"{metrics['NPV (Negative Predictive Value)']*100:.2f}%",
                '',
                f"{metrics['True Positives']}",
                f"{metrics['True Negatives']}",
                f"{metrics['False Positives']}",
                f"{metrics['False Negatives']}",
            ],
            'Interpretation': [
                'Overall correctness',
                'Ability to discriminate classes',
                'Balance of precision & recall',
                'When model says positive, % correct',
                'Of actual positives, % detected',
                'Of actual negatives, % detected',
                'When model says negative, % correct',
                '',
                'Correctly identified DR cases',
                'Correctly identified healthy cases',
                'Healthy flagged as DR (over-referral)',
                'DR cases missed (under-referral)',
            ]
        }
        
        df = pd.DataFrame(data)
        
        if save:
            csv_path = self.save_dir / 'metrics_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Metrics table saved to: {csv_path}")
        
        return df


def analyze_results(y_true, y_pred, y_probs, save_dir="results"):
    """
    Main function to analyze and display results
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        y_probs: Prediction probabilities
        save_dir: Directory to save results
    
    Returns:
        metrics: Dictionary of all metrics
        df: Summary DataFrame
    """
    analyzer = ResultsAnalyzer(y_true, y_pred, y_probs, save_dir)
    
    # Print comprehensive summary
    metrics = analyzer.print_summary()
    
    # Create and save summary table
    df = analyzer.create_summary_table(save=True)
    
    # Display table
    print("\n" + "="*80)
    print("📋 METRICS SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return metrics, df
