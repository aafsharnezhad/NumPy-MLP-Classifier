

import numpy as np 


def relu(x):
    """
    Applies the Rectified Linear Unit (ReLU) activation function element-wise.

    Args:
        x (np.ndarray): Input array or matrix.

    Returns:
        np.ndarray: The activated array where all negative values are set to 0.
    """
    return np.maximum(0, x)  


def soft_max(Z): 
    """
    Computes the Softmax activation for a batch of predictions, converting logits 
    into a probability distribution. Includes a numerical stability shift.

    Args:
        Z (np.ndarray): Raw logits from the output layer. Shape: (Batch_Size, Num_Classes).

    Returns:
        np.ndarray: Probability distribution for each sample. Shape: (Batch_Size, Num_Classes).
    """
    # Subtracting the maximum value in each row prevents np.exp() from overflowing
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    
    # Normalize by the sum of exponentials for each row independently
    predictions = expZ / np.sum(expZ, axis=1, keepdims=True)

    return predictions


def cross_entropy_loss(y_pred, y_true):
    """
    Calculates the categorical Cross-Entropy loss for a batch of predictions.

    Args:
        y_pred (np.ndarray): Predicted probabilities (from Softmax). Shape: (B, C).
        y_true (np.ndarray): True one-hot encoded labels. Shape: (B, C).

    Returns:
        float: The scalar average loss over the batch.
    """
    B = y_pred.shape[0] # Batch Size

    # Clip probabilities to prevent log(0) which causes NaN errors
    epsilon = 1e-4
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # Compute loss: multiply true labels by log of predictions and average over the batch
    loss = -1/B * (np.sum(y_true * (np.log(y_pred))))
    return loss    


def compute_total_loss(): 
    """
    Placeholder for computing total loss (e.g., adding L2 regularization penalty to the base loss).
    """
    pass   


def relu_backward(Z):
    """
    Computes the derivative of the ReLU function.

    Args:
        Z (np.ndarray): The pre-activation values (logits) of a hidden layer.

    Returns:
        np.ndarray: The ReLU gradients (1 where Z > 0, else 0).
    """
    # astype(float) converts boolean True/False array into 1.0/0.0
    return (Z > 0).astype(float)



class DenseLayer():
    """
    A standard fully connected (dense) neural network layer.
    
    Handles both forward propagation and backpropagation, and maintains its own 
    learnable parameters (Weights and Biases) along with momentum variables.
    """
    
    def __init__(self, input_dim, neurons):
        """
        Initializes the DenseLayer with randomly scaled weights and zero biases.

        Args:
            input_dim (int): Number of features/neurons from the previous layer.
            neurons (int): Number of neurons in this current layer.
        """
        self.neurons = neurons
        
        # Initialize weights randomly and biases to zero
        
        self.W = np.random.randn(input_dim, neurons) * 0.1
        self.b = np.zeros(shape=(1, neurons))
        
        # Placeholders for gradients
        self.dW = 0
        self.dz = 0
        
        # Cache for storing forward pass values needed during backpropagation
        self.cache = {}
        
        # Initialize velocity matrices for momentum-based SGD
        self.V_dW = np.zeros_like(self.W)
        self.V_db = np.zeros_like(self.b)


    def forward(self, A_prev):
        """
        Performs the linear forward pass: Z = A_prev @ W + b.

        Args:
            A_prev (np.ndarray): Activations from the previous layer. Shape: (B, input_dim).

        Returns:
            np.ndarray: The pre-activation linear output (Z). Shape: (B, neurons).
        """
        # Linear transformation
        z = A_prev @ self.W + self.b 
        
        # Store variables in cache for use during the backward pass
        self.cache['current_z'] = z
        self.cache['A_prev'] = A_prev 
        
        return z 


    def backward(self, learning_rate, w_next=None, beta=0.9, current_z=None, 
                 Y_true=None, Y_predicted=None, dz_next=None, lambda_reg=0):
        """
        Computes the backward pass to calculate gradients and update layer parameters.

        Args:
            learning_rate (float): The step size for gradient descent.
            w_next (np.ndarray, optional): Weights of the next layer.
            beta (float, optional): Momentum coefficient. Defaults to 0.9.
            current_z (np.ndarray, optional): Cached Z values of this layer.
            Y_true (np.ndarray, optional): True one-hot labels (if output layer).
            Y_predicted (np.ndarray, optional): Model predictions (if output layer).
            dz_next (np.ndarray, optional): Gradients from the next layer.
            lambda_reg (float, optional): L2 regularization hyperparameter.

        Returns:
            np.ndarray: The gradient with respect to the pre-activations of this layer (dz).
        """
        A_prev = self.cache["A_prev"]

        # Determine gradient calculation based on layer position (Output vs. Hidden)
        if self.neurons == 10: 
            # Simplified gradient calculation for Softmax + Cross-Entropy output layer
            m = A_prev.shape[0]
            self.dz = (Y_predicted - Y_true) / m 
             
            
        else: 
            # Standard hidden layer backpropagation chain rule
            self.dz = (dz_next @ w_next.T) * relu_backward(current_z)

        # Calculate Gradients for Weights and Biases
        self.dW = A_prev.T @ self.dz
        
        # Sum across the batch dimension to maintain correct bias shape (1, neurons)
        self.db = np.sum(self.dz, axis=0, keepdims=True)

        # Apply L2 Regularization penalty to weight gradients if specified
        if lambda_reg > 0:
            self.dW += ((lambda_reg / m) * self.W)

        # Calculate gradient to pass back to the previous layer
        self.dA_prev = self.dz @ self.W.T   

        # 2. Update Weights and Biases using Momentum Optimization
        self.V_dW = (beta * self.V_dW) + ((1 - beta) * self.dW)
        self.V_db = (beta * self.V_db) + ((1 - beta) * self.db)

        # Apply the update step
        self.W -= learning_rate * self.V_dW
        self.b -= learning_rate * self.V_db

        return self.dz
        

class MLP():
    """
    Multi-Layer Perceptron (Neural Network) class that chains together multiple DenseLayers.
    """
    
    def __init__(self, layers, X, batch_size=32, num_classes=10):
        """
        Constructs the neural network architecture based on the provided configuration.

        Args:
            layers (list of int): A list specifying the number of neurons in each hidden layer.
            X (np.ndarray): The input dataset, used here to infer input dimensions (if needed).
            batch_size (int, optional): The training batch size. Defaults to 32.
            num_classes (int, optional): The number of output classes. Defaults to 10.
        """
        self.first_layer_neurons = layers[0]
        self.num_classes = num_classes
        self.model_layers = []

        # Iterate through the configuration list to instantiate hidden layers
        for i in range(len(layers)): 
            if i == 0:
                # Assume standard flattened image input size (e.g., 28x28 = 784)
                input_dim = 784
            else: 
                # Input dimension matches output dimension of previous layer
                input_dim = layers[i-1]    

            self.model_layers.append(DenseLayer(input_dim=input_dim, neurons=layers[i]))
            
        # Append the final output layer using the specified number of classes
        self.model_layers.append(DenseLayer(input_dim=layers[-1], neurons=num_classes))

        self.X = X
            
    def forward(self, X):
        """
        Executes the full forward pass through the entire network architecture.

        Args:
            X (np.ndarray): Input batch of data.

        Returns:
            np.ndarray: Predicted probability distributions for the batch.
        """
        A_prev = X
        
        # Propagate through all hidden layers with ReLU activations
        for i in range(len(self.model_layers) - 1):
           layer = self.model_layers[i]
           Z = layer.forward(A_prev=A_prev)
           A = relu(Z)
           A_prev = A

        # Propagate through the final layer with a Softmax activation
        layer = self.model_layers[-1]
        Z_out = layer.forward(A_prev=A_prev)
        A_out = soft_max(Z_out)   
        
        y_predicted = A_out
        
        return y_predicted  

    def backward(self, learning_rate, y_true, y_predicted):
        """
        Executes backpropagation through the network, updating the weights 
        and biases of every layer sequentially from output to input.

        Args:
            learning_rate (float): The step size for gradient descent updates.
            y_true (np.ndarray): The ground truth labels (one-hot encoded).
            y_predicted (np.ndarray): The predictions outputted by the forward pass.
        """
        # 1. Start with the Output Layer
        last_layer = self.model_layers[-1]
        dZ = last_layer.backward(
            learning_rate, 
            w_next=None, 
            beta=0.9, 
            current_z=None, 
            Y_true=y_true, 
            Y_predicted=y_predicted, 
            dz_next=None, 
            lambda_reg=0
        )
        
        # Store weights and gradients to pass to the preceding layer
        w_next = last_layer.W
        dz_next = dZ

        # 2. Iterate backward through all hidden layers
        for i in reversed(range(len(self.model_layers) - 1)):
            layer = self.model_layers[i]
            current_z = layer.cache['current_z']
            
            dZ = layer.backward(
                learning_rate, 
                w_next=w_next, 
                beta=0.9, 
                current_z=current_z, 
                Y_true=None, 
                Y_predicted=None, 
                dz_next=dz_next, 
                lambda_reg=0
            )
            
            # Update running trackers for the next step in the reverse loop
            w_next = layer.W
            dz_next = dZ