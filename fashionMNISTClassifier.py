import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


class FashionMNISTClassifier:
    def __init__(self):
        self.model = None
        self.input_shape = (28, 28, 1)

    # Load and preprocess the Fashion MNIST dataset
    def load_and_preprocess_data(self):
        # Load Fashion MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # Normalize images
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

        # Reshape for CNN input
        train_images = train_images.reshape((-1, 28, 28, 1))
        test_images = test_images.reshape((-1, 28, 28, 1))

        # One-hot encode labels
        train_labels = to_categorical(train_labels, 10)
        test_labels = to_categorical(test_labels, 10)

        return train_images, train_labels, test_images, test_labels

    # Build the CNN model
    def build_model(self):

        # Define the CNN model with 6 layers
        self.model = models.Sequential([
            # 1st layer: Conv
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),

            # 2nd layer: Conv
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # 3rd layer: Conv
            layers.Conv2D(64, (3, 3), activation='relu'),

            # 4th layer: Flatten
            layers.Flatten(),

            # 5th layer: Dense
            layers.Dense(64, activation='relu'),

            # 6th layer: Output
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    # train the model    
    def train(self, train_images, train_labels, epochs=5):
        self.model.fit(train_images, train_labels, epochs=epochs, batch_size=64, verbose=2)

    # Evaluate the model on test data
    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")

    # Predict and visualize some images
    def predict_images(self, images, labels, class_names, count=2):
        predictions = self.model.predict(images[:count])

        for i in range(count):
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title(f"Actual: {class_names[np.argmax(labels[i])]} | Predicted: {class_names[np.argmax(predictions[i])]}")
            plt.axis('off')
            plt.show()
    
if __name__ == "__main__":
    classifier = FashionMNISTClassifier()
    train_images, train_labels, test_images, test_labels = classifier.load_and_preprocess_data()
    classifier.build_model()
    # Train the model
    classifier.train(train_images, train_labels, epochs=5)
    classifier.evaluate(test_images, test_labels)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Predict 2 sample images
    classifier.predict_images(test_images, test_labels, class_names, count=2)
  
    

