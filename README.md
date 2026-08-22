# NumPy-MLP-Classifier
A multi-layer perceptron (MLP) built entirely from scratch using only NumPy. This project classifies MNIST images to demonstrate the foundational mechanics of backpropagation, mini-batch gradient descent, and neural network architecture without relying on deep learning frameworks.

## About The Project:
This repository contains a fully functional Feed-Forward Neural Network (Multi-Layer Perceptron) implemented entirely from scratch using Python and NumPy.
## The Purpose

The primary goal of this project is educational. While modern frameworks like PyTorch and TensorFlow abstract away the math, this project was built to gain a deep, under-the-hood understanding of backpropagation, weight initialization, and gradient descent mechanics. 

The network is currently configured to classify images from the MNIST dataset, but the architecture is modular and dynamic.

### Key Features Implemented:
* **Zero External ML Frameworks:** Relies solely on NumPy for matrix operations.
* **Dynamic Architecture:** The network dynamically generates hidden layers based on a customizable YAML configuration file. 
* **Advanced Optimization:** Includes an implementation of Stochastic Gradient Descent (SGD) with Momentum. 
* **Regularization:** Features L2 regularization (weight decay) to prevent overfitting. 
* **Comprehensive Evaluation:** Includes custom visualization modules for tracking training history (loss/accuracy curves), plotting confusion matrices, and inspecting misclassified samples.