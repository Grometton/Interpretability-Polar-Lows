import os
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import model_builder as model_builder
from dataloaders import create_dataloaders
from utils import get_device
from dataloaders import get_transforms


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, 
                   test_loader: DataLoader, 
                   device: torch.device) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    Returns dictionary with all metrics and predictions.
    """
    model.eval()
    
    # Pre-allocate lists
    y_true_list = []
    y_pred_labels_list = []
    y_pred_probs_list = []
    
    start_time = time.time()
    
    for X, y in tqdm(test_loader, desc="Evaluating", unit="batch"):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Forward pass
        logits = model(X)
        
        # Get predictions and probabilities in one go
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(logits, dim=1)
        
        # Move to CPU and store
        y_true_list.append(y.cpu())
        y_pred_labels_list.append(pred_labels.cpu())
        y_pred_probs_list.append(probs[:, 1].cpu())  # Probability of positive class
    
    # Concatenate all results 
    y_true = torch.cat(y_true_list).numpy()
    y_pred_labels = torch.cat(y_pred_labels_list).numpy()
    y_pred_probs = torch.cat(y_pred_probs_list).numpy()
    
    evaluation_time = time.time() - start_time
    total_samples = len(y_true)  # Fixed: get actual sample count
    
    # Calculate metrics with error handling
    classification_rep = classification_report(y_true, y_pred_labels, output_dict=True)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC metrics: {e}")
        roc_auc = None
        pr_auc = None

    results = {
        'classification_report': classification_rep,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_true': y_true,
        'y_pred_labels': y_pred_labels,
        'y_pred_probs': y_pred_probs,
        'evaluation_time': evaluation_time,
        'total_samples': total_samples
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total samples evaluated: {results['total_samples']}")
    print(f"Evaluation time: {results['evaluation_time']:.2f} seconds")
    
    print("\nClassification Report:")
    print("-" * 40)
    cr = results['classification_report']
    for class_name, metrics in cr.items():
        if isinstance(metrics, dict):
            print(f"{class_name:>12}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1-score']:.3f} Support={metrics['support']}")
    
    # Print AUC metrics if available
    if results['roc_auc'] is not None:
        print(f"\nAUC Metrics:")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"PR-AUC:  {results['pr_auc']:.4f}")


def generate_results_path(model_path: str, output_dir: str = "results") -> str:
    """Generate output filename based on model name."""
    model_name = Path(model_path).stem  # Gets filename without extension
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    return f"{output_dir}/{model_name}_evaluation_{timestamp}.json"


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {
        'classification_report': results['classification_report'],
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc'],
        'evaluation_time': results['evaluation_time'],
        'total_samples': results['total_samples'],
        'y_true': results['y_true'].tolist(),
        'y_pred_labels': results['y_pred_labels'].tolist(),
        'y_pred_probs': results['y_pred_probs'].tolist()
    }
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():     
    parser = argparse.ArgumentParser(description="Evaluate trained PyTorch model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--data-path", type=str, default="~/uncertainty/data/polar-lows", 
                       help="Path to data directory")

    args = parser.parse_args()

    # Expand user path if needed
    model_path = Path(args.model_path).expanduser()
    data_path = Path(args.data_path).expanduser()
    
    # Validate paths
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)
    
    device = get_device('auto')
    print(f"Using device: {device}")

    train_dir = data_path / 'train'
    test_dir = data_path / 'test' 

    train_transform, test_transform = get_transforms()
    
    try:
        train_loader, test_loader, n_pos_train, n_neg_train, class_to_idx_dict = create_dataloaders(
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            train_transform=train_transform,
            test_transform=test_transform
        )
        
        print(f"Class mapping: {class_to_idx_dict}")
        print(f"Training set: {n_pos_train} positive, {n_neg_train} negative samples")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        sys.exit(1)

    # Load model
    try:
        model = model_builder.XceptionCustom(input_channels=3)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Model loaded successfully from: {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("\nStarting evaluation...")
    results = evaluate_model(model, test_loader, device)
    print_results(results)
    
    output_path = generate_results_path(str(model_path))
    save_results(results, output_path)


if __name__ == '__main__': 
    main()