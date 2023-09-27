# CIFAR-10 Classification Experiment with TensorFlow

This document provides an overview of a simple experiment conducted on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in TensorFlow.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a well-known dataset in the machine learning community, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Model Architecture

The model is a simple CNN with the following layers:

- Three convolutional layers with ReLU activation.
- Two max-pooling layers.
- A fully connected layer with 64 units and ReLU activation.
- An output layer with 10 units (one for each class).

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10

## Results

The model was trained for 10 epochs. Below are the key metrics observed during training:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| 1     | 1.4879        | 45.81%            | 1.3466          | 50.77%              |
| 2     | 1.1226        | 60.08%            | 1.0799          | 61.84%              |
| ...   | ...           | ...               | ...             | ...                 |
| 10    | 0.5565        | 80.23%            | 0.8679          | 71.16%              |

### Final Evaluation

After 10 epochs, the model achieved:

- **Test Loss**: 0.8679
- **Test Accuracy**: 71.16%

This means the model correctly classified approximately 71.16% of the images in the test dataset.

## Analysis

1. **Performance**: The model achieved a test accuracy of ~71%, which is a decent result for a simple CNN on CIFAR-10. More advanced architectures or training techniques might yield better performance.

2. **Overfitting**: The model's training accuracy is higher than its validation accuracy, which is a typical sign of overfitting. However, the difference isn't drastic. The slight increase in validation loss towards the end of training might be an early indication of overfitting.

3. **Training Progress**: The model's accuracy on both training and validation data increased over epochs, indicating beneficial training. The losses decreased, which is also a good sign.

## Conclusion

This experiment provided a hands-on experience with TensorFlow, showcasing the process of building, training, and evaluating a CNN on the CIFAR-10 dataset. The results offer a baseline that can be improved upon with further experimentation and optimization.
