
# Federated MNIST Classification Workflow Template

## Overview

This template enables the classification of the MNIST dataset using 
[federated deep learning app](https://github.com/FeatureCloud/fc-deep-learning).

## MNIST Dataset

The MNIST dataset is a handwritten digits that is widely used for training and testing in the field of machine learning. It contains 70,000 images of handwritten digits from 0 to 9, each of which is a 28x28 pixel grayscale image. The dataset is divided into two parts: 60,000 training images and 10,000 testing images. This dataset is utilized for various tasks in image processing, especially for training and testing in the field of machine learning and computer vision.

![MNIST Sample](https://github.com/FeatureCloud/fc-mnist-workflow-template/blob/master/images/mnist.webp?raw=true)

## Federated Learning on MNIST

In this project, we apply federated learning to classify handwritten digits from the MNIST dataset. Instead of training a single model on a centralized dataset, federated learning allows models to be trained across multiple clients that each have a portion of the overall dataset. These local models are then aggregated into a global model, which can achieve high accuracy without any single entity ever having access to the full dataset.

The data is distributed in an IId fashion across two clients:

![Train set](https://github.com/FeatureCloud/fc-mnist-workflow-template/blob/master/images/mnist_train-hist.png?raw=true)

![Test set](https://github.com/FeatureCloud/fc-mnist-workflow-template/blob/master/images/mnist_test-hist.png?raw=true)

### Steps to Run the Project

1. Instantiate MNIST classification workflow template on featurecloud app store.
2. Share tokens with other partners and execute the workflow. 
3. Upload the sample data and plugins into the local container.
4. Observe the logs and download the results.

## Configuration File Overview

The configuration file, typically named `config.yml`, plays a crucial role in setting up the federated learning environment for the MNIST classification project. It specifies the plugins to be used, along with their respective parameters, ensuring seamless integration and execution of the workflow.

### Plugins

- **Data Loader Plugin**: Manages the loading and preprocessing of the MNIST dataset across federated clients.
- **Model Plugin**: Defines the neural network architecture used for digit classification.
- **Optimizer Plugin**: Configures the federated learning optimizer, including learning rate and aggregation strategies.
- **Aggregator Plugin**: Specifies the method for aggregating model updates from federated clients.

### Parameters

Each plugin requires specific parameters for its operation. These parameters include paths to data, model specifications (e.g., number of classes, input features), optimizer settings (e.g., learning rate, epochs), and aggregation rules (e.g., weighted average, secure aggregation techniques).

### Example Configuration

```yaml
fc_deep:
  local_dataset:
    train: "train.npz"
    test: "test.npz"
    central_test: mnist_central_test.npz
    detail:  {}


  logic:
    mode: "file"
    dir: "."

  result:
    pred: "y_pred.csv"
    target: "y_test.csv"
    central_pred: "central_pred.csv"
    central_target: "central_target.csv"
    model: "model.pt"

  fed_hyper_params:
    max_iter: 20
    n_classes: 10
    federated_model: 'FedAvg.py'
    global_updates: WEIGHTS_STOPPING
    param: { }


  use_smpc: False


  trainer:
    name: 'BasicTrainer'
    param: {}
    local_updates: WEIGHTS_N_SAMPLES

    data_loader: ImageLoader # True: using torchvision Dataloader, False using custom Dataloader
    optimizer:
      name: 'SGD'
      param:
        lr: 0.1
    loss:
      name: 'CrossEntropyLoss'

    metrics:
      - name: 'Accuracy'
        package: 'torchmetrics.classification'
        param:
          task: 'multiclass'
          num_classes: 10

  train_config:
    verbose: True
    batch_size: 32
    test_batch_size: 32
    epochs: 1
    lr: 0.1
    batch_count: 1
    device: 'gpu'

  model: # 'cnn' or 'cnn.py' or layers_dict
    name: 'cnn.py'
    n_classes: 10
    in_features: 1
```

This section provides guidance on configuring the system for federated learning tasks, ensuring that users can customize their setup according to their requirements.



### Requirements

- Python 3.8
- PyTorch
- FeatureCloud

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.