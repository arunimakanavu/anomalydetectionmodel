# Casting Defect Detection with Autoencoder

A fine-tuned deep learning model for detecting defects in metal casting products using convolutional autoencoders. This model is adapted from Intel's Edge AI Suite PCB anomaly detection architecture and specialized for industrial casting quality inspection.

## Overview

This project implements an autoencoder-based anomaly detection system specifically trained to identify defects in metal casting products. The model learns to reconstruct normal casting patterns and flags anomalies based on reconstruction error, achieving excellent performance with a final training loss of 0.0005 and a suggested anomaly threshold of 0.0004.

## Model Architecture

The autoencoder consists of:

- **Encoder**: 3-layer convolutional encoder (3→16→32→64 channels)
- **Decoder**: 3-layer transposed convolutional decoder (64→32→16→3 channels)
- **Input**: 304×304 RGB images (converted from grayscale)
- **Output**: Reconstructed images with sigmoid activation.


## Training Results

```
Epoch [6/10] - Loss: 0.0007
Epoch [7/10] - Loss: 0.0007
Epoch [8/10] - Loss: 0.0006
Epoch [9/10] - Loss: 0.0005
Epoch [10/10] - Loss: 0.0005
Model saved to casting_autoencoder.pth
⚡ Suggested anomaly threshold: 0.0004
```


## Dataset

The model was trained using the [Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle, which contains high-quality images of casting products with and without defects.

### Dataset Structure

```
casting_data/
├── train/          # Normal casting images only
│   └── ok/
└── test/           # Defective casting images for threshold calibration
    └── def_front/
```


## Installation

### Requirements

```bash
pip install torch torchvision numpy
```


### Dependencies

- PyTorch
- torchvision
- NumPy


## Usage

### Training

```python
python train_model.py
```


### Key Configuration Parameters

- **Image Size**: 304×304 pixels
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Epochs**: 10
- **Loss Function**: MSE Loss
- **Optimizer**: Adam


## Model Outputs

The training script generates:

- `casting_autoencoder.pth` - PyTorch model weights
- `casting_autoencoder.onnx` - ONNX export for deployment
- Calibrated anomaly threshold based on defective samples


## Anomaly Detection Process

1. **Training Phase**: Model learns to reconstruct normal casting images
2. **Threshold Calibration**: Uses defective samples to determine optimal threshold
3. **Inference**: Images with reconstruction error > threshold are flagged as defective

## Performance

- **Final Training Loss**: 0.0005
- **Suggested Threshold**: 0.0004
- **Model Type**: Unsupervised anomaly detection
- **Architecture**: Convolutional Autoencoder


## Applications

This model is designed for:

- Quality control in metal casting manufacturing
- Real-time defect detection on production lines
- Automated visual inspection systems
- Edge deployment in industrial environments


## Model Features

- **Unsupervised Learning**: Trained only on normal samples
- **Real-time Capable**: Optimized for edge deployment
- **ONNX Compatible**: Ready for production deployment
- **Automatic Thresholding**: Self-calibrating anomaly detection
- **Industrial Grade**: Tested on real manufacturing data


## Technical Details

The model uses a symmetric encoder-decoder architecture with:

- Stride-2 convolutions for downsampling
- Transposed convolutions for upsampling
- ReLU activation in hidden layers
- Sigmoid output activation for pixel reconstruction.


## Future Enhancements

- Multi-scale feature extraction
- Attention mechanisms for defect localization
- Integration with production line systems
- Real-time inference optimization


## Citation

If you use this model in your research or production system, please cite:

- Original PCB anomaly detection framework from Intel Edge AI Suite
- Casting product dataset from Kaggle (ravirajsinh45)


## License

This project builds upon open-source frameworks and datasets. Please ensure compliance with respective licenses for commercial use.

