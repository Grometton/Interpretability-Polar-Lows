from PIL import Image
import os
import torch 
import torch.nn.functional as F
from sklearn.metrics import classification_report



def get_image(path):
    """For an input image path, returns a PIL Image object."""
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
def ImagePred(model, img_path, transform, device):
    """
    Predict the class of a single image using the provided model.
    
    Args:
        model: The trained PyTorch model.
        image_path: Path to the image file.
        transform: Transformations to apply to the image.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        Input image as tensor 
        Class probability vector 
    """
    img = get_image(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model (img_tensor)
        probs = F.softmax(logits, dim=1)

    return img_tensor, probs

    # write code to predict image, finish  writing full LIME predictions 




def evaluate_model(model, test_loader, device):

    y_pred_labels = [] # Stores predicted class labels (0 or 1)
    y_true_labels = [] # Stores true class labels (0 or 1)
    y_pred_probs = [] # Stores probabilities for the positive class (for AUC-PR)
    misclassified_images = []
    misclassified_true_labels = []
    misclassified_pred_labels = []
    misclassified_prob_vectors = []

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

            # Identify misclassified images
            misclassified_mask = (y_pred_batch_labels != y)
            if misclassified_mask.any():
                misclassified_X = X[misclassified_mask].cpu()
                misclassified_y_true = y[misclassified_mask].cpu()
                misclassified_y_pred = y_pred_batch_labels[misclassified_mask].cpu()
                misclassified_y_probs = y_probs_batch[misclassified_mask].cpu()
                
                # Append to lists
                for i in range(misclassified_X.shape[0]):
                    misclassified_images.append(misclassified_X[i])
                    misclassified_true_labels.append(misclassified_y_true[i])
                    misclassified_pred_labels.append(misclassified_y_pred[i])
                    misclassified_prob_vectors.append(misclassified_y_probs[i])

    # Concatenate all collected tensors
    y_true_np = torch.cat(y_true_labels).numpy()
    y_pred_labels_np = torch.cat(y_pred_labels).numpy()
    y_pred_probs_np = torch.cat(y_pred_probs).numpy()

    return y_pred_labels_np, y_true_np, y_pred_probs_np, misclassified_images, misclassified_true_labels, misclassified_pred_labels, misclassified_prob_vectors




def f1_score(true_pos, false_pos, false_neg):
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)