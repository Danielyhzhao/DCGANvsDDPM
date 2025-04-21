# MNISTGenComp

This project implements and compares **DCGAN** and **DDPM** on the MNIST dataset, with training, evaluation (FID, Diversity), and visualization of learning curves.

## Features
- Train DCGAN and DDPM on MNIST with configurable hyperparameters  
- Grid search over learning rates, `beta1` (DCGAN), and `T` (DDPM)  
- Evaluate with FID and Diversity metrics  
- Visualize learning curves and compare metrics  

## Installation

### Prerequisites
- Python 3.8+  
- PyTorch  
- CUDA (optional, for GPU)  

### Setup
1. Clone the repository:

git clone https://github.com/your-username/MNISTGenComp.git

cd MNISTGenComp
2. Install dependencies:
pip install -r requirements.txt

torch
torchvision
torchmetrics
numpy
matplotlib
3. MNIST dataset will be auto-downloaded to `data/` when running the script.

## Usage

### Training Models
Train DCGAN and DDPM models:  python train.py

- Trains DCGAN with `lr=[0.0001, 0.0002, 0.0004]`, `beta1=[0.3, 0.5, 0.7]`  
- Trains DDPM with `lr=[1e-5, 2e-5, 5e-5]`, `T=[50, 100, 200]`  
- Saves images, learning curves, metrics to `images/`  
- Saves best models to `dcgan_generator_best.pth`, `dcgan_discriminator_best.pth`, `ddpm_unet_best.pth`  

**Output**:  
- Learning curves: `images/dcgan_lr0.0001_beta1=0.3/learning_curves_epoch49.png`  
- Generated images: `images/dcgan_lr0.0001_beta1=0.3/dcgan_epoch49.png`  
- Metrics: `metrics_comparison.txt`  

## Project Details

### Models
- **DCGAN**: Conditional DCGAN with label smoothing, gradient clipping, learning rate warmup, and TTUR  
- **DDPM**: Diffusion model with configurable diffusion steps (`T`)  

### Evaluation Metrics
- **FID**: Measures similarity between generated and real images  
- **Diversity**: Measures variance of generated images  

### Visualization
- Learning curves for Discriminator/Generator Loss (DCGAN) and Loss (DDPM)  
- Parameter comparison plots for FID, Diversity, and training time

## License
MIT License

## Acknowledgments
- MNIST dataset from torchvision  
- Inspired by DCGAN and DDPM research  
- Thanks to PyTorch community
