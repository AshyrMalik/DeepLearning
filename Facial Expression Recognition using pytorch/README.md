# Facial Expression Recognition using PyTorch

This repository contains the implementation of a **Facial Expression Recognition** system using **PyTorch**. The goal of this project is to classify facial expressions into different categories using deep learning techniques.

## Features
- **Deep Learning Framework**: Built using PyTorch for easy experimentation and flexibility.
- **Dataset Preprocessing**: Includes preprocessing steps for handling facial images.
- **Model Training**: Trains a Convolutional Neural Network (CNN) for facial expression classification.
- **Evaluation**: Provides metrics to evaluate the performance of the model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AshyrMalik/DeepLearning.git
   cd DeepLearning/Facial%20Expression%20Recognition%20using%20pytorch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses a public dataset for facial expression recognition. To use the dataset:

1. Download the dataset and place it in the `data/` directory.
2. Ensure the dataset is structured as follows:
   ```
   data/
   ├── train/
   │   ├── class_1/
   │   ├── class_2/
   │   └── ...
   └── test/
       ├── class_1/
       ├── class_2/
       └── ...
   ```

## Usage

### Training the Model

To train the model, run the following command:
```bash
python train.py --epochs <num_epochs> --batch_size <batch_size> --learning_rate <lr>
```

### Evaluating the Model

To evaluate the model on the test set, use:
```bash
python evaluate.py --model_path <path_to_saved_model>
```

### Inference

To make predictions on new images:
```bash
python predict.py --image_path <path_to_image> --model_path <path_to_saved_model>
```


## Results

- **Accuracy**: Achieved an accuracy of XX% on the test set.
- **Loss**: Final test loss was YY.
- Confusion matrix and classification report are available in the `results/` directory.

## Future Improvements
- Enhance the model architecture for better accuracy.
- Experiment with different datasets and preprocessing techniques.
- Deploy the model as a web application.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [PyTorch Documentation](https://pytorch.org/docs/)
- Publicly available datasets used for facial expression recognition.


