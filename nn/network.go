package nn

import (
	"math"
	"math/rand"
)

// NeuralNet represents a feedforward neural network
type NeuralNet struct {
	Layers       [][]float64   // Neuron values for each layer
	Weights      [][][]float64 // Weights between layers
	Biases       [][]float64   // Biases for each layer
	LearningRate float64       // Learning rate for gradient descent
	DropoutRate  float64       // Dropout rate for regularization
}

// NewNeuralNet creates a new neural network with the specified architecture
func NewNeuralNet(layerSizes []int, learningRate float64) *NeuralNet {
	nn := &NeuralNet{
		Layers:       make([][]float64, len(layerSizes)),
		Weights:      make([][][]float64, len(layerSizes)-1),
		Biases:       make([][]float64, len(layerSizes)-1),
		LearningRate: learningRate,
		DropoutRate:  0.2, // 20% dropout rate
	}

	// Initialize layers
	for i, size := range layerSizes {
		nn.Layers[i] = make([]float64, size)
	}

	// Initialize weights and biases using He initialization
	for i := 0; i < len(layerSizes)-1; i++ {
		nn.Weights[i] = make([][]float64, layerSizes[i])
		nn.Biases[i] = make([]float64, layerSizes[i+1])

		// He initialization: weights ~ N(0, sqrt(2/n))
		scale := math.Sqrt(2.0 / float64(layerSizes[i]))
		for j := range nn.Weights[i] {
			nn.Weights[i][j] = make([]float64, layerSizes[i+1])
			for k := range nn.Weights[i][j] {
				nn.Weights[i][j][k] = rand.NormFloat64() * scale
			}
		}

		// Initialize biases to small random values
		for j := range nn.Biases[i] {
			nn.Biases[i][j] = rand.NormFloat64() * 0.01
		}
	}

	return nn
}

// Forward performs forward propagation through the network
func (nn *NeuralNet) Forward(input []float64) []float64 {
	// Set input layer
	copy(nn.Layers[0], input)

	// Propagate through hidden layers with ReLU and dropout
	for i := 0; i < len(nn.Layers)-2; i++ {
		for j := range nn.Layers[i+1] {
			sum := nn.Biases[i][j]
			for k := range nn.Layers[i] {
				sum += nn.Layers[i][k] * nn.Weights[i][k][j]
			}
			nn.Layers[i+1][j] = relu(sum)

			// Apply dropout during training
			if rand.Float64() < nn.DropoutRate {
				nn.Layers[i+1][j] = 0
			}
		}
	}

	// Output layer with softmax
	lastLayer := len(nn.Layers) - 1
	for j := range nn.Layers[lastLayer] {
		sum := nn.Biases[lastLayer-1][j]
		for k := range nn.Layers[lastLayer-1] {
			sum += nn.Layers[lastLayer-1][k] * nn.Weights[lastLayer-1][k][j]
		}
		nn.Layers[lastLayer][j] = sum
	}

	// Apply softmax to output layer
	softmax(nn.Layers[lastLayer])

	return nn.Layers[lastLayer]
}

// relu activation function
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDerivative returns the derivative of the ReLU function
func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// softmax applies the softmax function to the output layer
func softmax(xs []float64) {
	// Find max value for numerical stability
	max := xs[0]
	for _, x := range xs {
		if x > max {
			max = x
		}
	}

	// Compute softmax
	sum := 0.0
	for i := range xs {
		xs[i] = math.Exp(xs[i] - max)
		sum += xs[i]
	}
	for i := range xs {
		xs[i] /= sum
	}
}

// Train performs one training step using backpropagation
func (nn *NeuralNet) Train(input, target []float64) float64 {
	output := nn.Forward(input)

	// Calculate error (Cross Entropy)
	error := make([]float64, len(output))
	for i := range output {
		error[i] = output[i] - target[i]
	}

	// Backpropagation
	deltas := make([][]float64, len(nn.Layers))

	// Output layer delta (Cross Entropy + Softmax)
	deltas[len(deltas)-1] = make([]float64, len(output))
	copy(deltas[len(deltas)-1], error)

	// Hidden layers
	for l := len(nn.Layers) - 2; l > 0; l-- {
		deltas[l] = make([]float64, len(nn.Layers[l]))
		for j := range nn.Layers[l] {
			sum := 0.0
			for k := range nn.Layers[l+1] {
				sum += deltas[l+1][k] * nn.Weights[l][j][k]
			}
			deltas[l][j] = sum * reluDerivative(nn.Layers[l][j])
		}
	}

	// Update weights and biases with momentum and L2 regularization
	momentum := 0.9
	weightUpdates := make([][][]float64, len(nn.Weights))
	biasUpdates := make([][]float64, len(nn.Biases))
	l2Lambda := 0.0001 // L2 regularization strength

	for l := 0; l < len(nn.Layers)-1; l++ {
		weightUpdates[l] = make([][]float64, len(nn.Weights[l]))
		biasUpdates[l] = make([]float64, len(nn.Biases[l]))

		for i := range nn.Layers[l] {
			weightUpdates[l][i] = make([]float64, len(nn.Layers[l+1]))
			for j := range nn.Layers[l+1] {
				// Calculate weight update with momentum and L2 regularization
				weightUpdates[l][i][j] = momentum*weightUpdates[l][i][j] -
					nn.LearningRate*(deltas[l+1][j]*nn.Layers[l][i]+l2Lambda*nn.Weights[l][i][j])
				nn.Weights[l][i][j] += weightUpdates[l][i][j]
			}
		}

		for j := range nn.Layers[l+1] {
			// Calculate bias update with momentum
			biasUpdates[l][j] = momentum*biasUpdates[l][j] -
				nn.LearningRate*deltas[l+1][j]
			nn.Biases[l][j] += biasUpdates[l][j]
		}
	}

	// Return Cross Entropy loss
	loss := 0.0
	for i := range target {
		if target[i] > 0 {
			loss -= target[i] * math.Log(output[i]+1e-10)
		}
	}
	return loss
}

// Predict returns the network's prediction for the given input
func (nn *NeuralNet) Predict(input []float64) []float64 {
	// Disable dropout during prediction
	originalDropoutRate := nn.DropoutRate
	nn.DropoutRate = 0
	defer func() { nn.DropoutRate = originalDropoutRate }()

	return nn.Forward(input)
}

// Evaluate computes accuracy and average loss on a dataset
func (nn *NeuralNet) Evaluate(inputs, targets [][]float64) (float64, float64) {
	correct := 0
	totalLoss := 0.0
	for i := range inputs {
		pred := nn.Predict(inputs[i])
		targetIdx := argmax(targets[i])
		predIdx := argmax(pred)
		if predIdx == targetIdx {
			correct++
		}
		// Cross Entropy loss
		for j := range pred {
			if targets[i][j] > 0 {
				totalLoss -= targets[i][j] * math.Log(pred[j]+1e-10)
			}
		}
	}
	accuracy := float64(correct) / float64(len(inputs))
	avgLoss := totalLoss / float64(len(inputs))
	return accuracy, avgLoss
}

func argmax(arr []float64) int {
	maxIdx := 0
	for i, v := range arr {
		if v > arr[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
