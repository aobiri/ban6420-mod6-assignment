# ban6420-mod6-assignment
Nexford - Programming in R and Python, Module 6 Assignment

# Context

As a junior machine learning researcher at Microsoft AI, you have been assigned the task of classifying images using profile images to target marketing for different products. To prepare for this project, you will begin by working with the  in Keras and later adapt the code for user profile classification.

Convolutional Neural Network (CNN):

Using Keras and classes in both Python and R, develop a CNN with six layers to classify the Fashion MNIST dataset.
Prediction:

Make predictions for at least two images from the Fashion MNIST dataset.


## Fashion MNIST Classifier

This README provides a step-by-step guide to using the `fashionMNISTClassifier.py` script for classifying images from the Fashion MNIST dataset.

### Prerequisites

1. **Python Installation**: Ensure you have Python 3.x installed on your machine.
2. **Required Libraries**: Install the necessary libraries using pip:
    ```bash
    pip install numpy tensorflow keras matplotlib
    ```

### Step 1: Download the Fashion MNIST Dataset

The Fashion MNIST dataset can be easily loaded using Keras. The dataset consists of 60,000 training images and 10,000 testing images.

### Step 2: Load the Dataset

In your `fashionMNISTClassifier.py`, load the dataset with the following code:
```python
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

### Step 3: Preprocess the Data

Normalize the images to a range of 0 to 1 for better performance:
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

### Step 4: Build the CNN Model

Define a Convolutional Neural Network with six layers:
```python
from keras import layers, models

build_model(self)
```

### Step 5: Compile the Model

Compile the model with an optimizer and loss function:
```python
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
```

### Step 6: Train the Model

Train the model using the training data:
```python
    classifier.train(train_images, train_labels, epochs=5)
```

### Step 7: Evaluate the Model

Evaluate the model's performance on the test dataset:
```python
    classifier.evaluate(test_images, test_labels)
```

### Step 8: Make Predictions

Use the model to make predictions on 2 sample images:
```python
  classifier.predict_images(test_images, test_labels, class_names, count=2)
```

### Author
Albert Obiri-Yeboah
