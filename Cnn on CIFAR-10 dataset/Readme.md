# CNN on CIFAR-10 Dataset

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images in the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cnn-on-cifar10.git
    cd cnn-on-cifar10
    ```

2. Install the required packages:
    ```bash
    pip install torch torchvision numpy matplotlib scikit-learn
    ```

### Usage

1. Run the training script:
    ```bash
    python cnn_on_cifar10_dataset.py
    ```

### Code Overview

The main script `cnn_on_cifar10_dataset.py` includes the following steps:

1. **Dataset Preparation**:
    - Download and load the CIFAR-10 dataset.
    - Apply necessary transformations.

2. **DataLoader**:
    - Create data loaders for training and testing datasets.

3. **Model Definition**:
    - Define a CNN architecture with three convolutional layers and two fully connected layers.

4. **Training**:
    - Train the model using the training dataset.
    - Calculate and display the training and testing losses for each epoch.

5. **Evaluation**:
    - Calculate and display the training and testing accuracy.
    - Generate and display a confusion matrix for the test dataset.

### Model Architecture

- **Convolutional Layers**:
    - Conv1: 3 input channels, 32 output channels, 3x3 kernel, stride 2
    - Conv2: 32 input channels, 64 output channels, 3x3 kernel, stride 2
    - Conv3: 64 input channels, 128 output channels, 3x3 kernel, stride 2

- **Fully Connected Layers**:
    - FC1: 128*3*3 input features, 1024 output features
    - FC2: 1024 input features, 10 output features (for 10 classes)

### Results

- Training and testing loss over epochs are plotted and displayed.
- Training and testing accuracy are printed.
- Confusion matrix of the test dataset is generated and displayed.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- The initial version of this code was generated using Google Colab.
- The CIFAR-10 dataset is provided by [Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html).

---

