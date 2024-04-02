# Transformer from scratch using PyTorch

This is an implementation of the Transformer model for machine translation using PyTorch. The Transformer model is a neural network architecture that has been highly successful in various natural language processing tasks, including machine translation.

## Features

- Transformer architecture with multi-head attention and feed-forward layers
- Positional encoding for handling sequential data
- Greedy decoding for inference
- Evaluation with character error rate (CER), word error rate (WER), and BLEU score
- Tensorboard integration for tracking training metrics
- Support for preloading and saving trained models

## Requirements

- Python 3.6 or higher
- PyTorch
- TorchText
- Tokenizers
- TorchMetrics
- Tqdm

## Usage

1. Configure the training settings in `config.py`.
2. Run `train.py` to start the training process.

During training, the script will:

- Load or build tokenizers for the source and target languages
- Split the dataset into train and validation sets
- Create data loaders for the train and validation sets
- Build the Transformer model
- Train the model with the specified number of epochs
- Validate the model on the validation set and log metrics to Tensorboard
- Save the trained model weights at the end of each epoch

## Code Structure

- `config.py`: Contains configurations for training, such as batch size, learning rate, and model parameters.
- `dataset.py`: Defines the `BilingualDataset` class for loading and preprocessing the dataset.
- `model.py`: Implements the Transformer model architecture and its components.
- `train.py`: The main script for training the model and handling validation and model saving.

## Credits

This implementation is based the tutorials by Umar Jamil on YouTube. This Transformer model is described in the paper "Attention Is All You Need" by Vaswani et al. (2017).
