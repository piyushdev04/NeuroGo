# NeuroGo: Neural Network Implementation in Go

A lightweight neural network implementation in Go, designed for educational purposes.

## Features

- **Simple Architecture**: Single hidden layer neural network with configurable layer sizes
- **Efficient Training**: Implements mini-batch gradient descent with momentum
- **Learning Rate Adaptation**: Dynamic learning rate adjustment based on training progress
- **High Performance**: Achieves 100% accuracy on the Iris dataset
- **Command-line Interface**: User-friendly CLI for model training, evaluation, and prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurogo.git
cd neurogo

# Build the project
go build

# Install globally (optional)
go install
```

## Usage

The CLI provides three main commands:

### Training a Model

```bash
# On Windows PowerShell
.\neurogo train --dataset data/iris.csv --epochs 400 --batch-size 4 --learning-rate 0.005 --output model.json

# On Unix-like systems
./neurogo train --dataset data/iris.csv --epochs 400 --batch-size 4 --learning-rate 0.005 --output model.json
```

Options:
- `--dataset, -d`: Path to dataset CSV file (required)
- `--epochs, -e`: Number of training epochs (default: 400)
- `--batch-size, -b`: Batch size for training (default: 4)
- `--learning-rate, -l`: Initial learning rate (default: 0.005)
- `--output, -o`: Path to save the trained model (default: model.json)

### Making Predictions

```bash
# On Windows PowerShell
.\neurogo predict --model model.json --input data/test.csv --output predictions.csv

# On Unix-like systems
./neurogo predict --model model.json --input data/test.csv --output predictions.csv
```

Options:
- `--model, -m`: Path to trained model file (required)
- `--input, -i`: Path to input CSV file (required)
- `--output, -o`: Path to save predictions (default: predictions.csv)

### Evaluating a Model

```bash
# On Windows PowerShell
.\neurogo evaluate --model model.json --test data/test.csv

# On Unix-like systems
./neurogo evaluate --model model.json --test data/test.csv
```

Options:
- `--model, -m`: Path to trained model file (required)
- `--test, -t`: Path to test dataset CSV file (required)

## Performance

The model has been tested on the Iris dataset with the following results:

- **Test Accuracy**: 100%
- **Test Loss**: 0.0609
- **Training Time**: ~400 epochs
- **Confusion Matrix**:
  ```
  Predicted      Setosa  Versicolor   Virginica
  Actual
  Setosa             14          0          0
  Versicolor          0          7          0
  Virginica           0          0          9
  ```

## Architecture

The current implementation uses a simple yet effective architecture:
- Input layer: 4 neurons (sepal length, sepal width, petal length, petal width)
- Hidden layer: 6 neurons with ReLU activation
- Output layer: 3 neurons with softmax activation (Setosa, Versicolor, Virginica)

## Training Parameters

- Learning rate: 0.005 (initial)
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
   - Command-line interface using Cobra
   - Separate commands for train/predict/evaluate
   - Command-line flags for parameters

### 🚧 Planned Improvements

1. **Data Handling**
   - [ ] Support for more dataset formats
   - [ ] Data augmentation
   - [ ] Cross-validation
   - [ ] Feature scaling options

2. **Model Management**
   - [ ] Model versioning
   - [ ] Training checkpoints
   - [ ] Model comparison tools

3. **Training Improvements**
   - [ ] Validation set support
   - [ ] Learning rate finder
   - [ ] Gradient clipping
   - [ ] More optimization algorithms

4. **Architecture Enhancements**
   - [ ] Multiple hidden layers
   - [ ] Different activation functions
   - [ ] Dropout regularization
   - [ ] Batch normalization

## License

This project is licensed under the MIT License - see the LICENSE file for details. 