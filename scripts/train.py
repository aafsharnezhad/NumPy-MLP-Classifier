import numpy as np
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from models.model import cross_entropy_loss , compute_total_loss



def calculate_accuracy(predictions, labels_one_hot):
    """
    Calculates the classification accuracy.

    Args:
        predictions (np.ndarray): The predicted probabilities from the model (Shape: [N, num_classes]).
        labels_one_hot (np.ndarray): The true ground truth one-hot encoded labels (Shape: [N, num_classes]).

    Returns:
        float: The accuracy as a ratio (e.g., 0.95 for 95% accuracy).
    """
    pred_indices = np.argmax(predictions, axis=1)
    true_indices = np.argmax(labels_one_hot, axis=1)
    return np.mean(pred_indices == true_indices)



def train_model(model, X_train, y_train_oh, X_val, y_val_oh, config):
    """
    Executes the training loop for the neural network using Mini-Batch Gradient Descent.

    Args:
        model (MLP): The instantiated neural network model.
        X_train (np.ndarray): Training data features.
        y_train_oh (np.ndarray): Training one-hot encoded labels.
        X_val (np.ndarray): Validation data features.
        y_val_oh (np.ndarray): Validation one-hot encoded labels.
        config (dict): Dictionary containing training hyperparameters.

    Returns:
        dict: A dictionary containing training history (loss and accuracy per epoch).
    """
    # Extract hyperparameters from configuration
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    
    num_samples = X_train.shape[0]
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\n--- Starting Training Process ---")
    print(f"Hyperparameters -> Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

  

    for epoch in range(epochs):
        # 1. Shuffle data at the beginning of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_oh[indices]
        
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # 2. Mini-batch iteration
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            y_batch = y_train_shuffled[i : i + batch_size]
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            
            ### batch_loss = cross_entropy_loss(predictions, y_batch)
            lambda_val = config.get('lambda_reg')
            batch_loss = cross_entropy_loss(y_pred=predictions, y_true=y_batch)
     
   
            batch_acc = calculate_accuracy(predictions, y_batch)
            
            epoch_train_loss += batch_loss * X_batch.shape[0]
            epoch_train_acc += batch_acc * X_batch.shape[0]
            
            # Backward pass (Compute gradients)
            model.backward(learning_rate = learning_rate, y_true = y_batch, y_predicted = predictions)
            
            
            
            
            
        
        epoch_train_loss /= num_samples
        epoch_train_acc /= num_samples
        
        
        val_predictions = model.forward(X_val)
        ###epoch_val_loss = cross_entropy_loss(val_predictions, y_val_oh)

        epoch_val_loss = cross_entropy_loss(val_predictions, y_val_oh)
        epoch_val_acc = calculate_accuracy(val_predictions, y_val_oh)
        
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    return history
