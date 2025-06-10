package cmd

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"neurogo/nn"
)

// Global variables to store state
var (
	trainInputs  [][]float64
	trainTargets [][]float64
	testInputs   [][]float64
	testTargets  [][]float64
	net          *nn.NeuralNet
)

// Class names for Iris dataset
var irisClasses = []string{"Setosa", "Versicolor", "Virginica"}

// Execute starts the interactive CLI menu.
func Execute() error {
	var trainInputs, trainTargets, testInputs, testTargets [][]float64

	for {
		fmt.Println("\nWelcome to Go NeuralNet CLI 🚀")
		fmt.Println("1. Load dataset")
		fmt.Println("2. Define architecture")
		fmt.Println("3. Train model")
		fmt.Println("4. Evaluate")
		fmt.Println("5. Exit")

		fmt.Print("\nEnter choice: ")
		var choice int
		fmt.Scan(&choice)

		switch choice {
		case 1:
			// Load dataset
			fmt.Println("Loading dataset (iris.csv)...")
			trainInputs, trainTargets, testInputs, testTargets = loadDataset()
		case 2:
			// Define architecture
			if len(trainInputs) == 0 {
				fmt.Println("Please load dataset first")
				continue
			}
			defineArchitecture()
		case 3:
			// Train model
			if net != nil && len(trainInputs) > 0 {
				TrainModel(net, trainInputs, trainTargets)
			} else {
				fmt.Println("Please load dataset and define architecture first")
			}
		case 4:
			// Evaluate
			if net != nil && len(testInputs) > 0 {
				fmt.Println("Evaluating model...")
				start := time.Now()
				accuracy, loss := net.Evaluate(testInputs, testTargets)
				duration := time.Since(start)
				fmt.Printf("Test Accuracy: %.2f%%, Loss: %.4f, Time: %v\n\n", accuracy*100, loss, duration)

				// Display sample predictions
				fmt.Println("Sample Predictions:")
				fmt.Println("-------------------")
				for i := 0; i < 6; i++ {
					pred := net.Predict(testInputs[i])
					targetIdx := argmax(testTargets[i])
					predIdx := argmax(pred)
					classNames := []string{"Setosa", "Versicolor", "Virginica"}
					fmt.Printf("Sample (Class: %s):\n", classNames[targetIdx])
					fmt.Printf("  Features: %v\n", testInputs[i])
					fmt.Printf("  Predicted: %s (confidence: %.2f%%)\n", classNames[predIdx], pred[predIdx]*100)
					fmt.Printf("  Actual: %s\n", classNames[targetIdx])
					fmt.Printf("  Correct: %v\n\n", predIdx == targetIdx)
				}

				// Display confusion matrix
				fmt.Println("Confusion Matrix:")
				fmt.Println("----------------")
				confusion := make([][]int, 3)
				for i := range confusion {
					confusion[i] = make([]int, 3)
				}
				for i := range testInputs {
					pred := net.Predict(testInputs[i])
					targetIdx := argmax(testTargets[i])
					predIdx := argmax(pred)
					confusion[targetIdx][predIdx]++
				}
				fmt.Println("   Predicted      Setosa  Versicolor   Virginica")
				fmt.Println("Actual")
				classNames := []string{"Setosa", "Versicolor", "Virginica"}
				for i, name := range classNames {
					fmt.Printf("%-10s %10d %10d %10d\n", name, confusion[i][0], confusion[i][1], confusion[i][2])
				}
			} else {
				fmt.Println("Please load dataset and train model first")
			}
		case 5:
			fmt.Println("Exiting...")
			return nil
		default:
			fmt.Println("Invalid choice")
		}
	}
}

// splitData splits the dataset into training and testing sets
func splitData(inputs, targets [][]float64, trainRatio float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	// Create indices array
	indices := make([]int, len(inputs))
	for i := range indices {
		indices[i] = i
	}

	// Shuffle indices
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Calculate split point
	splitPoint := int(float64(len(inputs)) * trainRatio)

	// Split the data
	trainInputs := make([][]float64, splitPoint)
	trainTargets := make([][]float64, splitPoint)
	testInputs := make([][]float64, len(inputs)-splitPoint)
	testTargets := make([][]float64, len(inputs)-splitPoint)

	for i := 0; i < splitPoint; i++ {
		trainInputs[i] = inputs[indices[i]]
		trainTargets[i] = targets[indices[i]]
	}
	for i := splitPoint; i < len(inputs); i++ {
		testInputs[i-splitPoint] = inputs[indices[i]]
		testTargets[i-splitPoint] = targets[indices[i]]
	}

	return trainInputs, trainTargets, testInputs, testTargets
}

func loadDataset() ([][]float64, [][]float64, [][]float64, [][]float64) {
	file, err := os.Open("data/iris.csv")
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, nil, nil, nil
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return nil, nil, nil, nil
	}

	// Skip header
	records = records[1:]

	// Create temporary slices for all data
	allInputs := make([][]float64, len(records))
	allTargets := make([][]float64, len(records))

	for i, record := range records {
		// Convert features to float64 (skip ID column)
		features := make([]float64, 4)
		for j := 0; j < 4; j++ {
			val, err := strconv.ParseFloat(record[j+1], 64) // Skip ID column
			if err != nil {
				fmt.Printf("Error parsing feature %d in record %d: %v\n", j, i, err)
				return nil, nil, nil, nil
			}
			features[j] = val
		}

		// Validate feature ranges (typical Iris dataset ranges)
		if features[0] < 4.0 || features[0] > 8.0 || // sepal length
			features[1] < 2.0 || features[1] > 4.5 || // sepal width
			features[2] < 1.0 || features[2] > 7.0 || // petal length
			features[3] < 0.1 || features[3] > 2.5 { // petal width
			fmt.Printf("Warning: Unusual feature values in record %d: %v\n", i, features)
		}

		allInputs[i] = features

		// Convert target to one-hot encoding
		target := make([]float64, 3)
		classLabel := record[5] // Class label is in the last column
		var targetIndex int
		switch classLabel {
		case "Iris-setosa":
			targetIndex = 0
		case "Iris-versicolor":
			targetIndex = 1
		case "Iris-virginica":
			targetIndex = 2
		default:
			fmt.Printf("Error: Unknown class label '%s' in record %d\n", classLabel, i)
			return nil, nil, nil, nil
		}
		target[targetIndex] = 1
		allTargets[i] = target
	}

	// Split data into training and test sets (80% training, 20% testing)
	trainInputs, trainTargets, testInputs, testTargets = splitData(allInputs, allTargets, 0.8)

	fmt.Printf("Dataset loaded successfully:\n")
	fmt.Printf("- Total samples: %d\n", len(allInputs))
	fmt.Printf("- Training samples: %d\n", len(trainInputs))
	fmt.Printf("- Test samples: %d\n", len(testInputs))
	fmt.Printf("- Input features: %d (sepal length, sepal width, petal length, petal width)\n", len(allInputs[0]))
	fmt.Printf("- Output classes: %d (Setosa, Versicolor, Virginica)\n", len(allTargets[0]))

	// Print sample of first few records
	fmt.Println("\nSample of first 3 records:")
	for i := 0; i < min(3, len(allInputs)); i++ {
		fmt.Printf("Record %d: Features: [%.1f, %.1f, %.1f, %.1f], Class: %s\n",
			i+1,
			allInputs[i][0], allInputs[i][1], allInputs[i][2], allInputs[i][3],
			irisClasses[argmax(allTargets[i])])
	}

	return trainInputs, trainTargets, testInputs, testTargets
}

func defineArchitecture() {
	// Define network architecture
	layerSizes := []int{4, 6, 3}             // Input -> Hidden -> Output
	net = nn.NewNeuralNet(layerSizes, 0.005) // Increased learning rate for faster learning

	fmt.Printf("Architecture defined: %v\n", layerSizes)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
