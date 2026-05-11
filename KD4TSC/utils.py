"""Utility functions for metrics, logging and model management"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred, duration):
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        duration: Training duration in seconds

    Returns:
        DataFrame with metrics
    """

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'duration': duration
    }

    df = pd.DataFrame([metrics])

    return df



def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model and return predictions.

    Args:
        model: Trained model
        test_loader: Tets data loader
        device: Device to use

    Returns:
        Tuple of (y_true, y_pred)
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            # Handle one-hot encoded labels
            if labels.dim() > 1:
                labels = labels.argmax(dim=1)
            labels = labels.numpy()

            all_preds.extend(predictions)
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds)



def save_logs(model, test_loader, output_dir, history, duration, device='cuda'):
    """
    Save all logs and metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        output_dir: Output directory
        history: Training history dictionary
        duration: Training duration
        device: Device
    """    
    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(output_dir, 'history.csv'), index=False)

    # Load best model and evaluate
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get predictions
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Calculate metrics
    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(os.path.join(output_dir, 'df_metrics.csv'), index=False)
    
    # Save best model info
    # Find index of best model (minimum loss after 2/3 of epochs)
    loss_column = 'student_loss' if 'student_loss' in history else 'train_loss'
    start_idx = int(len(history['train_loss']) * 2 / 3)
    
    if loss_column in history:
        best_idx = start_idx + np.argmin(history[loss_column][start_idx:])
    else:
        best_idx = start_idx + np.argmin(history['train_loss'][start_idx:])
    
    best_model_info = {
        'best_model_train_loss': history['train_loss'][best_idx],
        'best_model_train_acc': history['train_acc'][best_idx],
        'best_model_val_loss': history['val_loss'][best_idx] if history['val_loss'][best_idx] > 0 else 0.0,
        'best_model_val_acc': history['val_acc'][best_idx] if history['val_acc'][best_idx] > 0 else 0.0,
        'best_model_learning_rate': history['lr'][best_idx],
        'best_model_nb_epoch': best_idx + 1
    }
    
    df_best_model = pd.DataFrame([best_model_info])
    df_best_model.to_csv(os.path.join(output_dir, 'df_best_model.csv'), index=False)
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {df_metrics['accuracy'].values[0]:.4f}")
    print(f"  Precision: {df_metrics['precision'].values[0]:.4f}")
    print(f"  Recall: {df_metrics['recall'].values[0]:.4f}")



def get_best_teacher(dataset_name, teacher_root_path, num_iterations=5):
    """
    Find the best teacher model based on training loss.
    
    Args:
        dataset_name: Name of the dataset
        teacher_root_path: Root path to teacher models
        num_iterations: Number of iterations to consider
        
    Returns:
        Path to best teacher model
    """
    train_losses = []
    val_accuracies = []
    teacher_paths = []
    
    for i in range(1, num_iterations + 1):
        iter_name = f'UCRArchive_2018_itr_{i}'
        dataset_path = os.path.join(teacher_root_path, iter_name, dataset_name)
        
        # Check if path exists
        if not os.path.exists(dataset_path):
            print(f"Warning: Path {dataset_path} does not exist")
            continue
        
        # Read metrics
        try:
            df_best_model = pd.read_csv(os.path.join(dataset_path, 'df_best_model.csv'))
            train_loss = df_best_model['best_model_train_loss'].values[0]
            train_losses.append(train_loss)
            
            df_metrics = pd.read_csv(os.path.join(dataset_path, 'df_metrics.csv'))
            val_acc = df_metrics['accuracy'].values[0]
            val_accuracies.append(val_acc)
            
            model_path = os.path.join(dataset_path, 'best_model.pth')
            teacher_paths.append(model_path)
        except Exception as e:
            print(f"Warning: Could not read metrics for {dataset_path}: {e}")
    
    if not train_losses:
        raise ValueError(f"No teacher models found for {dataset_name}")
    
    # Select teacher with minimum training loss
    best_idx = np.argmin(train_losses)
    best_teacher_path = teacher_paths[best_idx]
    
    print(f"\nBest teacher for {dataset_name}:")
    print(f"  Path: {best_teacher_path}")
    print(f"  Train Loss: {train_losses[best_idx]:.4f}")
    print(f"  Val Accuracy: {val_accuracies[best_idx]:.4f}")
    
    return best_teacher_path


def create_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def check_done(output_dir):
    """Check if experiment is already completed."""
    done_file = os.path.join(output_dir, 'DONE')
    return os.path.exists(done_file)


def mark_done(output_dir):
    """Mark experiment as completed."""
    done_file = os.path.join(output_dir, 'DONE')
    open(done_file, 'w').close()


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_shape):
    """Print model summary."""
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {count_parameters(model):,}")
    print("="*60)
    
    # Try to print a forward pass shape
    try:
        dummy_input = torch.randn(1, input_shape[1], input_shape[0])
        with torch.no_grad():
            output = model(dummy_input.to(next(model.parameters()).device))
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not run forward pass: {e}")
    
    print("="*60 + "\n")


