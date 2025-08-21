import os
from collections import Counter
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
 



# --- train and test transforms ---
def get_transforms(CROP_HEIGHT = 512, 
               CROP_WIDTH = 512,
               IMAGE_SIZE = (288 + 512, 288 + 512), # (800, 800) for initial resize
               BATCH_SIZE = 16,
               NUM_WORKERS = os.cpu_count() or 1,
               train_dir = os.path.join('data', 'train'),
               test_dir = os.path.join('data', 'test')
               ): 

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


    return train_transform, test_transform


# --- dataloaders ---
def create_dataloaders(train_dir: str,
                       test_dir: str, 
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size = 16,
                       num_workers = os.cpu_count() or 1): 
    """
    Create training and testing dataloaders.
    Takes in a training directory and a testing directory path 
    and turns them into PyTorch datasets and subsequently into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: Torchvision transform to apply to training data.
        test_transform: Torchvision transform to apply to testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.


    Returns:
        A tutle of (train_dataloader, test_dataloader, n_pos, n_neg, class_to_idx_dict),
        Example usage: train_dataloader, test_dataloader, n_pos, n_neg, class_to_idx_dict = \
                                            = create_dataloaders(train_dir=path/to/train_dir,
                                                                 test_dir=path/to/test_dir,
                                                                 transform=some_transform,
                                                                 batch_size=16,
                                                                 num_workers=os.cpu_count())
    """

    
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

    targets = [label for _, label in train_data]
    class_counts = Counter(targets)
    class_to_idx_dict = train_data.class_to_idx

    n_pos_train = class_counts.get(train_data.class_to_idx.get('pos', -1), 0) 
    n_neg_train = class_counts.get(train_data.class_to_idx.get('neg', -1), 0) 

    total = n_pos_train + n_neg_train
    print(f"Positive training samples: {n_pos_train} ({int(n_pos_train/total*100)}%), Negative training samples: {n_neg_train} ({int(n_neg_train/total*100)}%)")
    

    # Create a balanced sampler for oversampling
    class_weights = []
    for _, label in train_data:
        class_weights.append(1.0 / class_counts[label])
    
    # Convert tensor to a list of floats for WeightedRandomSampler
    sampler = WeightedRandomSampler(class_weights, num_samples=len(train_data) * 2, replacement=True)
        
    # define dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #print(f"\nTrain DataLoader batch size: {train_loader.batch_size}")
    #print(f"Test DataLoader batch size: {test_loader.batch_size}")

    return train_loader, test_loader, n_pos_train, n_neg_train, class_to_idx_dict


