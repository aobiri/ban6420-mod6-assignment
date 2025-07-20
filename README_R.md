# FashionMNISTClassifier.R

## Overview
This script implements a classifier for the Fashion MNIST dataset using machine learning techniques. The Fashion MNIST dataset consists of 70,000 grayscale images of clothing items, categorized into 10 classes.

## Dependencies
- Required libraries: `tensorflow`, `keras` etc.
```bash
    if (!require("keras")) install.packages("keras")
    if (!require("R6")) install.packages("R6")
```

## Functions

### load_data()
Loads the Fashion MNIST dataset and preprocesses it for training and testing.
```bash
    data <- dataset_fashion_mnist()
```

### Pre-process and build_model() the model with 6 layers
Defines and compiles the neural network architecture for the classifier.
```bash
    self$preprocess_data()
    self$build_model()
```

### train_model(model, train_data, train_labels)
Trains the model using the training dataset and labels.
```bash
    cnn$train_model(epochs = 5)
```

### evaluate_model(model, test_data, test_labels)
Evaluates the trained model on the test dataset and returns accuracy metrics.
```bash
    cnn$evaluate_model()
```

### predict(model, new_data)
Generates predictions for new data using the trained model.

## Usage
1. Load the dataset using `load_data()`.
2. Build the model with `build_model()`.
3. Train the model using `train_model()`.
4. Evaluate the model's performance with `evaluate_model()`.
5. Make predictions on new data with `predict()`.
