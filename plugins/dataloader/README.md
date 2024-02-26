
# CustomDataLoader Plugin for loading image dataset saved in NumPy formats

## Overview

The `CustomDataLoader` plugin is designed to facilitate the loading and preprocessing of MNIST dataset images for classification tasks in a federated learning environment. This plugin extends the `AbstractDataLoader` class, providing functionality to read dataset files, preprocess images, and create PyTorch `DataLoader` instances for efficient training.

## Features

- Supports `.npz` and `.npy` file formats for dataset loading.
- Implements image preprocessing including normalization and transformation into PyTorch tensors.
- Facilitates the creation of PyTorch `DataLoader` instances with customized batch sizes, shuffling, and multiprocessing.
- Integrates with federated learning workflows, allowing for the decentralized processing of data.

## Requirements

- Python 3.8
- NumPy
- PyTorch
- torchvision
- PIL (Python Imaging Library)

Ensure these dependencies are installed in your environment to use the `CustomDataLoader` plugin.
