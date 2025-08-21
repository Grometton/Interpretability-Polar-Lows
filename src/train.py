import torch 
from typing import Dict, List
import os 
from collections import Counter
from torchvision import datasets, transforms
from typing import Tuple, Optional
import copy
from tqdm import tqdm
from urllib import request
import zipfile
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary 
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc 
from utils import get_device

from model_builder import XceptionCustom
from dataloaders import get_transforms, create_dataloaders
from utils import f1_score
import argparse
import sys

"""
Contains functions for making training and testing PyTorch dataloaders 
for image classification data from train and test folders 
"""

def train_step(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            device: torch.device,
            steps_per_epoch: int) -> Tuple[float, float, int, int, float, float]: 

    model.to(device)
    model.train()

    all_preds = []
    all_labels = []
    total_loss, total_correct, total_samples = 0.0, 0.0, 0.0
    
    # Manually limit batches per epoch to match TF's steps_per_epoch
    for batch_num, (X, y) in enumerate(dataloader):
        if batch_num >= steps_per_epoch:
            break

        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1) # Predicted classes (0 or 1)

        loss_batch = loss_fn(y_logits, y)
        batch_size_actual = y.shape[0] # Actual batch size, might be smaller at the end
        total_loss += loss_batch.item() * batch_size_actual # .item() to get scalar
        total_samples += batch_size_actual
        total_correct += (y_preds == y).sum().item()

        # Detach tensors before moving to CPU and appending to lists
        all_preds.append(y_preds.detach().cpu())
        all_labels.append(y.detach().cpu())

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    true_pos = ((all_preds == 1) & (all_labels == 1)).sum().item()
    false_pos = ((all_preds == 1) & (all_labels == 0)).sum().item()
    false_neg = ((all_preds == 0) & (all_labels == 1)).sum().item()

    binary_accuracy = (all_preds == all_labels).sum().item() / total_samples
    f1 = f1_score(true_pos, false_pos, false_neg)

    train_loss = float(total_loss / total_samples)
    train_accuracy = float(total_correct) / float(total_samples)
    
    return train_loss, train_accuracy, int(false_pos), int(false_neg), float(binary_accuracy), float(f1)

def train_model(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                device: torch.device,
                checkpoint_path: str,
                n_neg_train: int,
                loss_fn=torch.nn.CrossEntropyLoss(),
                num_epochs=200,
                batch_size=16,
                patience=20
                ): 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    steps_per_epoch = int(np.ceil(2.0 * n_neg_train / batch_size))
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_metric = float('inf') 
    early_stopping_counter = 0
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    history = {
        'train_loss': [],
        'train_acc': [],
        'f1': [],
    }

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
    
        # Pass steps_per_epoch_tf to train_step_fn
        train_loss, train_acc, _, _, _, f1 = train_step(model, 
                                                        train_loader, 
                                                        loss_fn, 
                                                        optimizer, 
                                                        device, 
                                                        steps_per_epoch
                                                        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['f1'].append(f1)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - F1: {f1:.4f}")

        # Learning rate scheduler step
        lr_scheduler.step(train_loss)

        # Save best model
        if (train_loss < best_metric):
            best_metric = train_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model updated.")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience or train_loss < 0.001:
            print("Early stopping triggered.")
            break

    # Restore best model
    model.load_state_dict(best_model_weights)
    return model, history

##### parser ####

def main(): 
    parser = argparse.ArgumentParser(description='Training pipeline for polar low Xception model')

    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    

    # to improve: use data directoryt in parser and subsequently validate train and test directories 
    # Make train-dir and test-dir required arguments
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training directory')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to test directory')
    
    # Fix the type specification
    parser.add_argument('--checkpoint-path', type=str, default='outputs/new_model.pth', 
                        help='Path for newly trained model - use format: outputs/DD.MM.YY_polar_lows_customex.pth (Default is new_model.pth)')

    args = parser.parse_args()

    # Validate that directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory '{args.train_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist.")
        sys.exit(1)

    device = get_device(args.device)
    train_dir = args.train_dir
    test_dir = args.test_dir 
    
    print(f'Using device: {device}')
    print(f'Train directory: {train_dir}')
    print(f'Test directory: {test_dir}')
    print(f'Checkpoint path: {args.checkpoint_path}')

    print('Creating dataloaders...')
    train_transform, test_transform = get_transforms()
    train_loader, test_loader, n_pos_train, n_neg_train, class_to_idx_dict = create_dataloaders(
        train_dir,
        test_dir, 
        train_transform,
        test_transform
    )
    
    print(f"Training samples: {n_pos_train + n_neg_train} (pos: {n_pos_train}, neg: {n_neg_train})")
    print(f"Class mapping: {class_to_idx_dict}")
    
    print('Creating model...')
    model = XceptionCustom().to(device)
    
    print('Starting training...')
    # FIXED: Added missing comma and proper argument passing
    trained_model, history = train_model(
        model,
        train_loader,
        device,
        checkpoint_path=args.checkpoint_path,
        n_neg_train=n_neg_train  # Fixed: added comma and proper assignment
    )

    print("Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final F1 score: {history['f1'][-1]:.4f}")

if __name__ == '__main__':
    main()