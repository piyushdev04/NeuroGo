# NeuroGo: Neural Network Implementation in Go

A minimal neural network framework.

NeuroGo lets you build, train, and interactively explore feedforward neural networks with features like ReLU, Softmax, Dropout, Backpropagation, Class Weights, Early Stopping, and more.

## Features

- **Simple Architecture**: Single hidden layer neural network with configurable layer sizes
- **Efficient Training**: Implements mini-batch gradient descent with momentum
- **Learning Rate Adaptation**: Dynamic learning rate adjustment based on training progress
- **High Performance**: Achieves 93.33% accuracy on the Iris dataset
- **Interactive CLI**: User-friendly command-line interface for model training and evaluation

## Usage

The CLI provides an interactive menu with the following options:

1. Load dataset
2. Define architecture
3. Train model
4. Evaluate
5. Exit

### Training Process

The model is trained on the Iris dataset with the following parameters:
- Input layer: 4 neurons (sepal length, sepal width, petal length, petal width)
- Hidden layer: 6 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (Setosa, Versicolor, Virginica)
- Learning rate: Starts at 0.005 with dynamic reduction
- Batch size: 4
- Epochs: 400

### Performance Results

The model achieves the following performance on the Iris dataset:

- **Test Accuracy**: 93.33%
- **Test Loss**: 0.5214
- **Training Time**: ~400 epochs
- **Confusion Matrix**:
  ```
  Predicted      Setosa  Versicolor   Virginica
  Actual
  Setosa              5          0          0
  Versicolor          0         14          2
  Virginica           0          0          9
  ```

### Sample Predictions

The model shows high confidence in predictions:
- Setosa samples: ~94-95% confidence
- Versicolor samples: ~46-59% confidence
- Virginica samples: ~55% confidence

## Architecture

The current implementation uses a simple yet effective architecture:
- Input layer: 4 neurons (sepal length, sepal width, petal length, petal width)
- Hidden layer: 6 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (Setosa, Versicolor, Virginica)

## Training Parameters

- Learning rate: 0.005 (initial) with dynamic reduction
- Batch size: 4
- Epochs: 400
- Patience: 8 epochs
- Learning rate reduction factor: 0.3

## Current Implementation Status

### ✅ Completed Features

1. **Core Neural Network**
   - Feedforward network with ReLU + Softmax
   - Training loop with batching
   - Early stopping
   - Learning rate scheduling
   - L2 regularization
   - He initialization for weights

2. **Training & Evaluation**
   - Train/Test data separation
   - Predict support
   - Evaluation metrics
   - Confusion matrix generation

3. **CLI Interface**
   - Interactive command-line interface
   - Dataset loading and preprocessing
   - Model training and evaluation
   - Real-time training progress display

## License

This project is licensed under the MIT License - see the LICENSE file for details. 