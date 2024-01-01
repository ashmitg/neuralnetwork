# Building a Neural Network from Scratch

## Introduction
In this project, I built a simple neural network from scratch without relying on external deep learning libraries such as PyTorch, Keras, or TensorFlow. The goal was to create a basic neural network to recognize handwritten digits using the famous MNIST dataset.

## Dataset
The dataset used for this project is the "Digit Recognizer" dataset from Kaggle, which contains labeled images of handwritten digits (0-9). The dataset is split into training and testing sets.

## Model Architecture
The neural network consists of an input layer, a hidden layer with a ReLU activation function, and an output layer. The architecture is as follows:
- **Input Layer:** 784 neurons (28x28 pixels of flattened images)
- **Hidden Layer:** 196 neurons with ReLU activation function
- **Output Layer:** 10 neurons representing digits 0-9

Weights and biases for both layers are randomly initialized.

## Training
The model is trained using a simple backpropagation algorithm with stochastic gradient descent (SGD) optimization. The ReLU activation function is employed for both the hidden and output layers. Training is performed on a subset of the dataset, and the model is updated iteratively using the backpropagation algorithm.

## Results
After training the model, it achieved an accuracy of 94% on a validation set of 1000 samples. This demonstrates the learning capability of a basic neural network and explains the theory of vector similarity in how machine learning works.

## Testing the Model
To assess the model's predictions, a sample image from the validation set is chosen. The input image is fed into the trained neural network, and the predicted class is compared with the correct class. The image along with the predicted and correct labels is displayed using Matplotlib.

## Conclusion
Building a neural network from scratch provides a fundamental understanding of the underlying principles of deep learning. While this implementation is basic and lacks the optimizations and features of established libraries, it serves as a valuable educational exercise. Further improvements could involve adding more layers, implementing different activation functions, or exploring advanced optimization techniques.
