import os, json
from urllib import request
import zipfile
import numpy as np
from collections import Counter
import copy
from typing import Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary 
from torch.autograd import Variable
import torch.nn.functional as F
from dataloaders import create_dataloaders

from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc # Import for full evaluation metrics

import model_builder
import matplotlib.pyplot as plt
 

device = torch.device("mps")



if __name__ == '__main__':
    device = torch.device("mps")
    print(f"Using device: {device}")

    # --- define transforms and create datasets ---

    # --- train and test transforms ---
    CROP_HEIGHT = 512
    CROP_WIDTH = 512
    IMAGE_SIZE = (288 + 512, 288 + 512) # (800, 800) for initial resize
    BATCH_SIZE = 16
    NUM_WORKERS = os.cpu_count() or 1  # Ensure NUM_WORKERS is always an int

    data_path = 'data'

    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')


    train_transform = transforms.Compose([
                                        transforms.Resize(IMAGE_SIZE), 
                                        transforms.RandomAffine( # RandomTranslation and RandomRotation
                                                    degrees=22.9, # 0.4 radians * (180/pi)
                                                    translate=(0.1, 0.1), # Max absolute fraction of translation
                                                    scale=(0.85, 1.15), # Equivalent to zoom (-0.15, 0.15)
                                                    fill=0), # Fill value with 0 for areas outside the image after transform
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.CenterCrop((CROP_HEIGHT, CROP_WIDTH)), # CenterCrop to 512x512
                                        transforms.ToTensor(), # Converts PIL Image to Tensor and scales pixels to [0, 1]
                                                    ])

    test_transform = transforms.Compose([
                                        transforms.Resize(IMAGE_SIZE), # First resize to the expected input size
                                        transforms.CenterCrop((CROP_HEIGHT, CROP_WIDTH)), # CenterCrop
                                        transforms.ToTensor(), # Converts PIL Image to Tensor and scales pixels to [0, 1]
                                        ])


    # --- train and test loaders 
    train_loader, test_loader, n_pos_train, n_neg_train, class_to_idx_dict = create_dataloaders(train_dir=train_dir,
                                                                                    test_dir=test_dir, 
                                                                                    train_transform=train_transform,
                                                                                    test_transform=test_transform,
                                                                                    batch_size=BATCH_SIZE,
                                                                                    num_workers=NUM_WORKERS)
    
    print(class_to_idx_dict)


    # Load the best model weights
    model = model_builder.XceptionCustom(input_channels=3)
    model.load_state_dict(torch.load('models/trained_model.pth'))

    y_pred_labels = [] # Stores predicted class labels (0 or 1)
    y_true_labels = [] # Stores true class labels (0 or 1)
    y_pred_probs = [] # Stores probabilities for the positive class (for AUC-PR)

    model.to(device)
    model.eval() # Set model to evaluation mode
    with torch.inference_mode(): # Equivalent to torch.no_grad() but explicitly for inference
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            
            # For classification_report, we need hard labels
            y_pred_batch_labels = torch.argmax(y_logits, dim=1)
            y_pred_labels.append(y_pred_batch_labels.cpu())
            y_true_labels.append(y.cpu())

            # For AUC-PR, we need probabilities of the positive class
            y_probs_batch = torch.softmax(y_logits, dim=1)
            y_pred_probs.append(y_probs_batch[:, 1].cpu()) # Probability of class 1

    # Concatenate all collected tensors
    y_true_np = torch.cat(y_true_labels).numpy()
    y_pred_labels_np = torch.cat(y_pred_labels).numpy()
    y_pred_probs_np = torch.cat(y_pred_probs).numpy()

    # Print classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(y_true_np, y_pred_labels_np))

