
# CNN Model Plugin

## Overview

This model plugin defines a PyTorch neural network model tailored for the classification of image datasets.

## Model Architecture

The model consists of the following layers:

- Two convolutional layers (`conv1` and `conv2`) for extracting features from the input images.
- A dropout layer (`conv2_drop`) applied after the second convolutional layer to prevent overfitting.
- Two fully connected layers (`fc1` and `fc2`) to perform classification based on the features extracted by the convolutional layers. The final layer outputs predictions for the `n_classes`, which is the number of digit classes (0-9) in the MNIST dataset.

## Requirements

- Python 3.8
- PyTorch

Ensure you have PyTorch installed in your environment, as it is required to define and utilize the model.

