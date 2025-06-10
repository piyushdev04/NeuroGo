package cmd

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"neurogo/nn"
)

// TrainModel trains the neural network on the given dataset
func TrainModel(net *nn.NeuralNet, inputs, targets [][]float64) {
	fmt.Println("Training model...")
	epochs := 400   // Increased epochs for better convergence
	batchSize := 4  // Smaller batch size for better generalization
	patience := 8   // Shorter patience for faster adaptation
	lrFactor := 0.3 // More aggressive learning rate reduction
	bestLoss := math.MaxFloat64
	noImprovement := 0

	// Calculate class weights to handle imbalance
	classCounts := make([]int, len(targets[0]))
	for _, target := range targets {
		for i, val := range target {
			if val == 1 {
				classCounts[i]++
			}
		}
	}
	maxCount := float64(classCounts[0])
	for _, count := range classCounts {
		if float64(count) > maxCount {
			maxCount = float64(count)
		}
	}
	classWeights := make([]float64, len(classCounts))
	for i, count := range classCounts {
		classWeights[i] = maxCount / float64(count)
	}

	// Create indices for shuffling
	indices := make([]int, len(inputs))
	for i := range indices {
		indices[i] = i
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		epochStart := time.Now()

		// Shuffle indices
		rand.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// Train in batches
		totalLoss := 0.0
		for i := 0; i < len(inputs); i += batchSize {
			end := i + batchSize
			if end > len(inputs) {
				end = len(inputs)
			}

			// Train on batch
			batchLoss := 0.0
			for j := i; j < end; j++ {
				idx := indices[j]
				// Apply class weights to the loss
				target := make([]float64, len(targets[idx]))
				copy(target, targets[idx])
				for k := range target {
					target[k] *= classWeights[k]
				}
				loss := net.Train(inputs[idx], target)
				batchLoss += loss
			}
			totalLoss += batchLoss / float64(end-i)
		}

		avgLoss := totalLoss / float64(len(inputs))
		duration := time.Since(epochStart)

		fmt.Printf("Epoch %d: Loss: %.4f, Time: %v\n", epoch, avgLoss, duration)

		// Early stopping with learning rate reduction
		if avgLoss < bestLoss {
			bestLoss = avgLoss
			noImprovement = 0
		} else {
			noImprovement++
			if noImprovement >= patience {
				// Reduce learning rate more gradually
				net.LearningRate *= lrFactor
				fmt.Printf("Reducing learning rate to %.6f\n", net.LearningRate)
				noImprovement = 0
			}
		}
	}
}
