
# Federated MNIST Classification Workflow Template

## Overview

This template enables the classification of the MNIST dataset using 
[federated deep learning app](https://github.com/FeatureCloud/fc-deep-learning).

## MNIST Dataset

The MNIST dataset is a handwritten digits that is widely used for training and testing in the field of machine learning. It contains 70,000 images of handwritten digits from 0 to 9, each of which is a 28x28 pixel grayscale image. The dataset is divided into two parts: 60,000 training images and 10,000 testing images. This dataset is utilized for various tasks in image processing, especially for training and testing in the field of machine learning and computer vision.

![MNIST Sample](images/mnist.webp)

## Federated Learning on MNIST

In this project, we apply federated learning to classify handwritten digits from the MNIST dataset. Instead of training a single model on a centralized dataset, federated learning allows models to be trained across multiple clients that each have a portion of the overall dataset. These local models are then aggregated into a global model, which can achieve high accuracy without any single entity ever having access to the full dataset.

### Steps to Run the Project

1. Instantiate MNIST classification workflow template on featurecloud app store.
2. Share tokens with other partners and execute the workflow. 
3. Upload the sample data and plugins into the local container.
4. Observe the logs and download the results.

### Requirements

- Python 3.8
- PyTorch
- FeatureCloud

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.