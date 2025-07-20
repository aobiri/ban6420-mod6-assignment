# Install required packages if not already done
if (!require("keras")) install.packages("keras")
if (!require("R6")) install.packages("R6")

library(keras)
library(R6)

# Install TensorFlow backend if not already installed
#keras::install_keras()  # Uncomment if running for the first time

CNNClassifier <- R6Class("CNNClassifier",
  public = list(
    model = NULL,
    train_images = NULL,
    train_labels = NULL,
    test_images = NULL,
    test_labels = NULL,
    
    initialize = function() {
      cat("Initializing CNN Classifier...\n")
      self$load_data()
      self$preprocess_data()
      self$build_model()
    },
    
    load_data = function() {
      cat("Loading Fashion MNIST dataset...\n")
      data <- dataset_fashion_mnist()
      self$train_images <- data$train$x
      self$train_labels <- data$train$y
      self$test_images <- data$test$x
      self$test_labels <- data$test$y
    },
    
    preprocess_data = function() {
      cat("Preprocessing data...\n")
      self$train_images <- self$train_images / 255
      self$test_images <- self$test_images / 255
      
      self$train_images <- array_reshape(self$train_images, c(nrow(self$train_images), 28, 28, 1))
      self$test_images <- array_reshape(self$test_images, c(nrow(self$test_images), 28, 28, 1))
      
      self$train_labels <- to_categorical(self$train_labels, 10)
      self$test_labels <- to_categorical(self$test_labels, 10)
    },
    
    build_model = function() {
      cat("Building CNN model with 6 layers...\n")
      self$model <- keras_model_sequential() %>%
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        layer_flatten() %>%
        layer_dense(units = 64, activation = 'relu') %>%
        layer_dense(units = 10, activation = 'softmax')
      
      self$model %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer_adam(),
        metrics = 'accuracy'
      )
    },
    
    train_model = function(epochs = 5, batch_size = 64) {
      cat("Training model...\n")
      self$model %>% fit(
        self$train_images, self$train_labels,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.2
      )
    },
    
    evaluate_model = function() {
      cat("Evaluating model on test data...\n")
      score <- self$model %>% evaluate(self$test_images, self$test_labels)
      #cat("Test accuracy:", score$accuracy, "\n")
      cat("Test accuracy:", score[["accuracy"]])
    },
    
    predict_image = function(index = 1) {
        cat("Making prediction on test image", index, "...\n")
        #dim(self$test_images)
        image <- self$test_images[index,,,,drop = FALSE]
        prediction <- self$model %>% predict(image)
        print(prediction)
        cat("Predicted class:", which.max(prediction) - 1, "\n")  # zero-indexed
    }
  )
)

cnn <- CNNClassifier$new()
cnn$train_model(epochs = 5)
cnn$evaluate_model()
cnn$predict_image(index = 1)
cnn$predict_image(index = 2)