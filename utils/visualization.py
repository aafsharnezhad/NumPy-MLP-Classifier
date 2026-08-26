import numpy as np 

import matplotlib.pyplot as plt

import collections

def showing_samples(X, y, num_classes = 10, samples_per_class = 5):
    
    """
    Plots a grid of random samples for each class.
    Each row represents a class, and each column is a random sample.
    
    Args:
        X (np.ndarray): The input feaures, shape (num_samples, 784).
        y (np.ndarray): The labels, shape (num_samples,).
        samples_per_class (int): Number of samples to show per class.
        num_classes (int): Total number of unique classes (0-9 for MNIST).
    """
    # Create a figure with 10 rows and 5 columns
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 15))
    
    # Add a main title to the plot
    fig.suptitle('Random Samples per Class', fontsize=16)

    for class_idx in range(num_classes):
        
        class_indices = np.where(y == class_idx)[0]
        
        # 2. Randomly select 5 indices from this specific class
        random_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        
        for col_idx, img_idx in enumerate(random_indices):

            
            img_flat = X[img_idx]
            img_2d = img_flat.reshape(28, 28)
            
            
            ax = axes[class_idx, col_idx]
            
            
            ax.imshow(img_2d, cmap='gray')
            ax.axis('off') 
            
            
            if col_idx == 0:
                ax.set_title(f"Class {class_idx}", fontsize=12)

    # Adjust layout so subplots don't overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Make room for the main title
    plt.show()


def plot_class_distribution(y, title="Class Distribution in Dataset"):

    class_proportion = {}
    counts = list(range(10))
    class_proportion = collections.Counter(y)

    for key, value in class_proportion.items() :
        counts[key] = value

    classes = range(10)
    
    

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='black', alpha=0.8)
    
    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Class (Digits 0-9)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    
    # Ensure all class numbers (0 to 9) are displayed on the x-axis
    plt.xticks(classes)
    
    # Add exact count numbers above each bar for better readability
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), 
                 ha='center', va='bottom', fontsize=10)
    
    # Add a subtle grid for easier reading of values
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    # print("class_proportion : ", class_proportion)
    
def plot_training_history(history, save_path=None):
    """
    Plots the training and validation loss and accuracy curves over epochs.

    Args:
        history (dict): The training history returned by the train_model function.
        save_path (str, optional): If provided, saves the plot to this path. Defaults to None.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    plt.title('Cross-Entropy Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='red', linestyle='--')
    plt.title('Classification Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()



