# MNIST Generation using DCGAN and DDPM

This repository contains code for training and comparing two generative models on the MNIST dataset: a conditional DCGAN and a conditional DDPM (Denoising Diffusion Probabilistic Model). In addition, a Flask web application is provided to generate and display MNIST images using pre-trained models interactively.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
  - [DCGAN Training](#dcgan-training)
  - [DDPM Training](#ddpm-training)
- [Web Application](#web-application)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Usage Notes](#usage-notes)
- [License](#license)

## Overview

The project focuses on generating MNIST digit images conditioned on their labels. Two main approaches are implemented:

1. **Conditional DCGAN**:  
   Uses a generator with transposed convolutions and an embedding layer for labels. A discriminator is built to distinguish between real and fake images while conditioning on the digit labels.

2. **Conditional DDPM (Denoising Diffusion Probabilistic Model)**:  
   Implements a conditional U-Net that guides the reverse diffusion process. The forward process gradually adds noise to input images, and the reverse process iteratively denoises an initial noise tensor.

The training process includes logging the following:
- Loss curves for the generator/discriminator (DCGAN) and DDPM.
- FID (Fréchet Inception Distance) for generated images.
- Diversity and training time evaluation.

After training, the best models (based on FID scores) are saved and compared using various metrics. The Flask web application can then load the pre-trained models for online MNIST generation.

## Project Structure

```
├── README.md
├── train.py                 # Main training script that runs DCGAN and DDPM training pipelines.
├── model.py                 # Contains the model definitions for Generator, Discriminator, ConditionalUNet, and DiffusionModel.
├── dataset.py               # Contains the MNIST dataset loader function.
├── utils.py                 # Utility functions for saving images, plotting learning curves, computing FID/diversity, etc.
├── app.py                   # Flask web application to generate images from the trained models.
└── requirements.txt         # List of required Python packages.
```

> **Note:** Some parts of the code provided (e.g., model definitions) may appear in multiple files (one for training and one for the Flask app). Adjust the file structure as needed for your project.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Danielyhzhao/MNISTGenCompare.git
   cd MNISTGenCompare
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   The required packages include:

   - `torch`
   - `torchvision`
   - `torchmetrics`
   - `matplotlib`
   - `numpy`
   - `Pillow`
   - `Flask`

## Training

### DCGAN Training

The training code for the conditional DCGAN is implemented in `main.py`. The model accepts a noise vector and label embeddings to generate MNIST images.

Parameters to adjust in the script:

- `z_dim` (dimensionality of the noise vector)
- `batch_size`
- Learning rate (`lr`) and optimizer parameters (`beta1`)
- Number of epochs (`num_epochs_dcgan`)

During training, samples are saved and FID scores computed for evaluation. Once training is complete, the best model is saved as `dcgan_generator_best.pth` and `dcgan_discriminator_best.pth` in the specified save directory.

### DDPM Training

Training for the DDPM is also included in `main.py`. The DDPM uses a conditional U-Net model that learns to predict the added noise during the forward diffusion process. Key parameters include:

- Learning rate for the diffusion model (`ddpm_lr_grid`)
- Number of time steps (`T`) for the forward diffusion process
- Number of epochs (`num_epochs_ddpm`)

The best DDPM U-Net is saved as `ddpm_unet_best.pth`.

To start training, simply run:

```bash
python train.py
```

This will train both models on the MNIST dataset using a grid search on defined hyperparameters and save the best performing models along with logs and visualizations in the specified directories.

## Web Application

A Flask web application is provided for interactive image generation:

1. **Run the Flask App:**

   ```bash
   python app.py
   ```

2. **Usage:**
   - Open your browser and navigate to `http://localhost:5000/`.
   - Choose the generation model (DCGAN or DDPM).
   - Optionally specify a digit (0-9) or leave blank to generate random digits.
   - Enter the number of images to generate.
   - Click "Generate Images" to see results.

The Flask app loads the pre-trained models and generates a grid of MNIST images which is then displayed on the web page.

## Evaluation and Metrics

The project includes utilities to evaluate generated images:
- **FID (Fréchet Inception Distance):** Measures how close the generated image distribution is to the real distribution.
- **Diversity:** Computes the variance across image pixels.
- **Training Time:** Records the training time of each model.

Plots for learning curves and parameter comparisons are saved during training for visual inspection.

## Usage Notes

- **Data Directory:**  
  The MNIST dataset is automatically downloaded to the directory specified by the `data_dir` parameter (adjust as needed).

- **Device Configuration:**  
  The training and inference scripts automatically use GPU if available.

- **Hyperparameter Tuning:**  
  The code uses grid search loops for both DCGAN and DDPM settings; you can modify the learning rate and other parameters as needed.

- **Saving Results:**  
  Generated images, model checkpoints, and metrics are saved in the directories specified by the code (`save_dir`, `images_base_dir`, and `best_dir`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- MNIST dataset from torchvision  
- Inspired by DCGAN and DDPM research  
- Thanks to PyTorch community
