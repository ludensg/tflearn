"""
cifar_experiment.py

Description:
    This script demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) 
    on the CIFAR-10 dataset using TensorFlow. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, 
    with 6,000 images per class. The primary purpose of this script is to provide a hands-on introduction to TensorFlow 
    for image classification tasks.

Author: Daniel Gandelman
Date: September, 2023

Usage:
    python cifar_experiment.py
"""


import tensorflow as tf
import matplotlib.pyplot as plt

# Daniel Gandelman, September, 2023
# Exercise for learning Tensorflow, using the CIFAR-10 dataset

# Load CIFAR-10 Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# This code will automatically download the CIFAR-10 dataset the first time you run it
# and will store it locally. Subsequent runs will use the locally stored data, 
# so you won't have to redownload it.


# Preprocess the Data
# It's a good practice to normalize the image data to have values between 0 and 1.
train_images = train_images / 255.0     # Since these are images, the RGB intensity is represented from 0-255
test_images = test_images / 255.0

# Define the CNN Model:
# 
# In summary, this CNN model consists of three convolutional layers interspersed 
# with max-pooling layers, followed by a flattening step, and then two dense layers. 
# The model is designed to take in images of shape (32, 32, 3) and output predictions
#  for the 10 classes in the CIFAR-10 dataset. 

model = tf.keras.models.Sequential([ #1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #2
    tf.keras.layers.MaxPooling2D((2, 2)), #3
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(), #4
    tf.keras.layers.Dense(64, activation='relu'), #5
    tf.keras.layers.Dense(10) #6
])

#region <Model Detail Explanation>
#1: This initializes a linear stack of layers.
# You can think of it as a list where you can add layers in sequence.
# The data flows from the input, through each layer in the order they're added,
# and finally to the output.

#2: Convolutional Layers
#
# These layers perform convolution operations. Convolutional layers are the major 
# building blocks used in CNNs and are designed to automatically and adaptively
# learn spatial hierarchies of features from input images.
#
# The first argument (e.g., 32 or 64) is the number of filters (or kernels) 
# the layer will learn. Filters are small, learnable weight matrices which the layer 
# slides or "convolves" around the input image to produce a feature map or 
# convolutional layer.
#
# (3, 3) specifies the height and width of the convolution window. 
# In this case, each filter is 3x3 pixels.
#
# activation='relu': The Rectified Linear Unit (ReLU) activation function is used.
# It introduces non-linearity to the model, allowing it to learn from the error and
# make adjustments, which is essential for learning complex patterns.
# 
# input_shape=(32, 32, 3): This is specified only for the first layer. 
# It indicates that the input images are 32x32 pixels with 3 color channels 
# (red, green, blue).

#3: Pooling Layers
#
# These layers are used to reduce the spatial dimensions of the output volume. 
# It works by sliding a 2x2 pooling filter across the input volume and taking 
# the max value among the 4 values in the 2x2 filter. 
# This reduces the height and width by half, making computations faster and 
# reducing the chance of overfitting.

#4: Flattening Layer
#
# After the convolutional and pooling layers, the high-level reasoning 
# in the neural network is done via fully connected layers. 
# Before connecting the layers, you'll flatten the 2D output of the previous 
# layers to 1D.

#5: Dense (Fully Connected) Layers
#
# These are standard layers of neurons in a neural network. 
# Each neuron receives input from all the neurons in the previous layer, 
# thus densely connected.
#
# 64, activation='relu': The first dense layer has 64 neurons.


#6: "10": The last dense layer has 10 neurons (one for each class in CIFAR-10). 
# This layer doesn't have an activation function because we're using it with the 
# SparseCategoricalCrossentropy(from_logits=True) loss function, which expects 
# raw logits as its input.
#endregion

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#region <Model Compilation Detail Explanation>
    # "Adam" is a popular optimization algorithm that adapts 
    # the learning rate during training. It combines the advantages of 
    # two other extensions of stochastic gradient descent: AdaGrad and RMSProp.

    # The loss function measures how well the model is doing during training, 
    # and the goal is to minimize this function.
    # SparseCategoricalCrossentropy is a loss function suitable for classification
    # tasks with multiple classes. It's used when the labels are integers 
    # (e.g., 0, 1, 2, ...). If the labels were one-hot encoded, 
    # you'd use CategoricalCrossentropy instead. 
    # 
    # The from_logits=True argument indicates that the model's output is not passed
    # through a softmax activation function. The softmax function would convert 
    # the model's logits (raw predictions) into probabilities for each class. 
    # If from_logits is set to True, the loss function will apply the softmax 
    # itself. If you had a softmax activation in the last layer of your model, 
    # you'd set from_logits to False.

    # metrics=['accuracy']: Metrics are used to monitor the training and 
    # testing steps. In this case, we're monitoring the "accuracy," 
    # which calculates how often the model's predictions match the true labels.
#endregion

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#region <Model Training Detail Explanation>
# model.fit(): 
#       This method trains the model for a fixed number of epochs (iterations on the dataset).
#
# train_images, train_labels:
#       These are the training data and corresponding labels. The model will learn from this data.
#
# epochs=10:
#       An epoch is one complete forward and backward pass of all the training examples. 
#   Setting epochs=10 means the entire dataset will be passed through the model 10 times. 
#   With each epoch, the model should improve its performance, reducing the loss and 
#   increasing the accuracy.
#
# validation_data=(test_images, test_labels):
#       This argument provides data that the model hasn't been trained on. 
#   After each epoch, the model will evaluate its performance on this validation data. 
#   It's essential to monitor the model's performance on unseen data to ensure 
#   it's not just memorizing the training data (overfitting).
# 
# 
#  The fit method returns a History object, which contains a record of training loss values 
# and metrics values at successive epochs, as well as validation loss values and validation
# metrics values (if applicable). We'll use this later to visualize the training and validation 
# accuracy and loss.
#
#endregion

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)


# Visualize Training and Validation Accuracy & Loss

plt.figure(figsize=(12, 4))

# Plot for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Save the figure as an image file
plt.savefig('training_plots.png')

