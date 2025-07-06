import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay
)


def evaluate_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_probs: Optional[np.ndarray] = None,
    class_names: List[str] = ['Control', 'ASD']
) -> Dict[str, Any]:
    """
    Model evaluation with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for AUC calculation)
        class_names: Names of the classes
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # AUC if probabilities are provided
    auc = None
    if y_probs is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_probs)
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'auc': float(auc) if auc is not None else None,
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names)
    }
    
    return results


def create_cv_visualizations(
    cv_results: List[Dict],
    output_dir: Path,
    experiment_name: str = "experiment"
) -> None:
    """
    Create visualizations for cross-validation results.
    
    Args:
        cv_results: List of fold results
        output_dir: Directory to save visualizations
        experiment_name: Name of the experiment
    """
    # Extract metrics
    cv_metrics = {
        'accuracy': [r['test_accuracy'] for r in cv_results],
        'balanced_accuracy': [r['test_balanced_accuracy'] for r in cv_results],
        'auc': [r['test_auc'] for r in cv_results]
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Box plot of metrics
    ax1 = fig.add_subplot(gs[0, :2])
    metrics_df = pd.DataFrame(cv_metrics)
    box_plot = metrics_df.boxplot(ax=ax1, patch_artist=True)
    
    # Customize box plot
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot.artists, colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Cross-Validation Metrics Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, (metric, values) in enumerate(cv_metrics.items()):
        mean_val = np.mean(values)
        ax1.text(i+1, mean_val + 0.02, f'μ={mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Learning curves (from last fold if available)
    if 'history' in cv_results[-1]:
        ax2 = fig.add_subplot(gs[1, 0])
        history = cv_results[-1]['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax2.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, color='blue')
        ax2.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training History (Last Fold)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy progression
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(epochs, history['train_accuracy'], label='Train Acc', linewidth=2, color='blue')
        ax3.plot(epochs, history['val_accuracy'], label='Val Acc', linewidth=2, color='red')
        if 'val_balanced_accuracy' in history:
            ax3.plot(epochs, history['val_balanced_accuracy'], label='Val Bal Acc', linewidth=2, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Progression (Last Fold)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning rate schedule
        ax4 = fig.add_subplot(gs[1, 2])
        if 'lr' in history:
            ax4.plot(epochs, history['lr'], 'g-', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
    
    # 5. Performance comparison bar plot
    ax5 = fig.add_subplot(gs[2, :2])
    metrics_mean = {k: np.mean(v) for k, v in cv_metrics.items()}
    metrics_std = {k: np.std(v) for k, v in cv_metrics.items()}
    
    x = np.arange(len(metrics_mean))
    width = 0.6
    bars = ax5.bar(x, metrics_mean.values(), width, yerr=metrics_std.values(),
                   capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    
    ax5.set_ylabel('Score')
    ax5.set_title('Average Performance Metrics')
    ax5.set_xticks(x)
    ax5.set_xticklabels([name.replace('_', ' ').title() for name in metrics_mean.keys()])
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, (name, value) in zip(bars, metrics_mean.items()):
        height = bar.get_height()
        ax5.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 6. Results summary table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Create summary statistics
    summary_data = []
    for metric, values in cv_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        summary_data.append([
            metric.replace('_', ' ').title(),
            f'{mean_val:.3f}',
            f'{std_val:.3f}',
            f'{min_val:.3f}',
            f'{max_val:.3f}'
        ])
    
    table = ax6.table(
        cellText=summary_data,
        colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.suptitle(f'{experiment_name.replace("_", " ").title()} - Cross-Validation Results', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / f'{experiment_name}_cv_results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_results(
    cv_results: List[Dict],
    config: Any,
    output_dir: Path,
    experiment_name: str = "experiment"
) -> None:
    """
    Save results to JSON file.
    
    Args:
        cv_results: Cross-validation results
        config: Configuration object
        output_dir: Output directory
        experiment_name: Name of the experiment
    """
    # Calculate summary statistics
    cv_metrics = {
        'accuracy': [r['test_accuracy'] for r in cv_results],
        'balanced_accuracy': [r['test_balanced_accuracy'] for r in cv_results],
        'auc': [r['test_auc'] for r in cv_results]
    }
    
    # Convert config to dict (handle dataclass)
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
    else:
        config_dict = dict(config)
    
    # Prepare final results
    final_results = {
        'experiment_name': experiment_name,
        'config': config_dict,
        'cv_results': cv_results,
        'summary': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
            for metric, values in cv_metrics.items()
        },
        'num_folds': len(cv_results)
    }
    
    # Save to JSON
    output_file = output_dir / f'{experiment_name}_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'-'*40}")
    print(f"{experiment_name.replace('_', ' ').title()} - Results")
    print(f"{'-'*40}")
    
    for metric, stats in final_results['summary'].items():
        print(f"{metric.title()}:")
        print(f"  Mean +- Std: {stats['mean']:.4f} +- {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    print(f"Results saved to: {output_file}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray, 
    class_names: List[str] = ['Control', 'ASD'],
    output_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix with percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        output_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i + 0.3, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_probs: Prediction probabilities for positive class
        output_path: Path to save plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # AUC if probabilities provided
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity and sensitivity  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    } 