


package main

import (
	"fmt"
	"image"
	"os"

	"github.com/kelvins/lbph"
	"github.com/kelvins/lbph/metric"
)

func main() {

	// Prepare the training data
	var paths []string
	// paths = append(paths, "./train/s1/pr_test_6.jpg")
	// paths = append(paths, "./train/s2/ishita_test_2.jpg")
	paths = append(paths, "./train/s1/pr_train_1.jpg")
	paths = append(paths, "./train/s2/ishita1.jpg")
	// paths = append(paths, "./train/s3/papa_test_1.jpg")
	paths = append(paths, "./train/s3/papa2.jpg")

	var labels []string
	labels = append(labels, "pushpraj")
	labels = append(labels, "ishita")
	labels = append(labels, "papa")

	var images []image.Image

	for index := 0; index < len(paths); index++ {
		img, err := loadImage(paths[index])
		checkError(err)
		images = append(images, img)
	}

	// Define the LBPH parameters
	// This is optional, if you not set the parameters using
	// the Init function, the LBPH will use the default ones
	params := lbph.Params{
		Radius:    1,
		Neighbors: 8,
		GridX:     8,
		GridY:     8,
	}

	// Set the parameters
	lbph.Init(params)

	// Train the algorithm
	err := lbph.Train(images, labels)
	checkError(err)

	// Prepare the testing data
	paths = nil
	paths = append(paths, "./pr_test_6.jpg")
	paths = append(paths, "./ishita_test_2.jpg")
	paths = append(paths, "./papa_test_1.jpg")
	// paths = append(paths, "./papa2.jpg")

	var expectedLabels []string
	expectedLabels = append(expectedLabels, "pushpraj")
	expectedLabels = append(expectedLabels, "ishita")
	expectedLabels = append(expectedLabels, "papa")

	// Select the metric used to compare the histograms
	// This is optional, the default is EuclideanDistance
	lbph.Metric = metric.EuclideanDistance

	// For each data in the training dataset
	for index := 0; index < len(paths); index++ {
		// Load the image
		img, err := loadImage(paths[index])
		checkError(err)

		// Call the Predict function
		label, distance, err := lbph.Predict(img)
		checkError(err)

		// Check the results
		if label == expectedLabels[index] {
			fmt.Println("Image correctly predicted")
		} else {
			fmt.Println("Image wrongly predicted")
		}
		fmt.Printf("Predicted as %s expected %s\n", label, expectedLabels[index])
		fmt.Printf("Distance: %f\n\n", distance)
	}
}

// loadImage function is used to load an image based on a file path
func loadImage(filePath string) (image.Image, error) {
	fImage, err := os.Open(filePath)
	checkError(err)

	defer fImage.Close()

	img, _, err := image.Decode(fImage)
	checkError(err)

	return img, nil
}

// checkError functions is used to check for errors
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
