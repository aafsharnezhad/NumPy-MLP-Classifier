import numpy as np 

import matplotlib.pyplot as plt

def showing_samples(X, y, num_classes = 10, samples_per_class = 5):
    
    """
    Plots a grid of random samples for each class.
    Each row represents a class, and each column is a random sample.
    
    Args:
        X (np.ndarray): The input features, shape (num_samples, 784).
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