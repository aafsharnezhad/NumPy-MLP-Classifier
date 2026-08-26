import sys
import os 
import pickle
import numpy as np
import yaml # Useful if you move the config dict to config.yaml later

current_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_path, "..")
sys.path.append(root_dir)

from data.data_loader import get_dataloader, standardize_data
from utils.visualization import showing_samples, plot_class_distribution
from models.model import MLP
from scripts.train import train_model

def load_data(data_dir):
    with open(data_dir, "rb") as f :
        data = pickle.load(f, encoding="latin1")
    return data

def to_one_hot(y, num_classes=10):
    """Converts integer labels to one-hot encoded arrays."""
    return np.eye(num_classes)[y]

def main():
    # 1. Load the Dataset
    data_dir = os.path.join(root_dir, "data/dataset/mnist.pkl")
    data = load_data(data_dir=data_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = standardize_data(data)

    # (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_dataloader(data)

    # Optional Visualizations
    # showing_samples(X_train, y_train)
    # plot_class_distribution(y_train)    

    # 2. Preprocess Labels
    # The train_model function expects one-hot encoded labels
    num_classes = 10
    y_train_oh = to_one_hot(y_train, num_classes)
    y_val_oh = to_one_hot(y_val, num_classes)
    y_test_oh = to_one_hot(y_test, num_classes)

    # 3. Define the Configuration
    # This matches the keys expected by train_model 
    config = {
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-3
        },
        'model': {
            'layers': [16] # Example: One hidden layer with 16 neurons
        },
        'lambda_reg': 0
    }

    # 4. Instantiate the Model
    print("Initializing Model...")
    model = MLP(
        layers=config['model']['layers'], 
        X=X_train, 
        batch_size=config['training']['batch_size'], 
        num_classes=num_classes
    )

    # 5. Train the Model
    # Executes the mini-batch gradient descent loop
    history = train_model(
        model=model, 
        X_train=X_train, 
        y_train_oh=y_train_oh, 
        X_val=X_val, 
        y_val_oh=y_val_oh, 
        config=config
    )

    # 6. Evaluate on Test Data (Optional)
    print("\n--- Evaluating on Test Set ---")
    test_predictions = model.forward(X_test)
    pred_indices = np.argmax(test_predictions, axis=1)
    test_acc = np.mean(pred_indices == y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()