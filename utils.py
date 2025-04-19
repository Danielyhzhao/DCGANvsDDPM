# utils.py
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import os

def save_images(images, filename):
    vutils.save_image(images, filename, normalize=True, nrow=4)

def plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, ddpm_loss_history, fid_ddpm_history, epoch, save_dir, phase='dcgan'):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot losses on the left y-axis
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    if phase == 'dcgan' or phase == 'both':
        if loss_D_history:
            ax1.plot(loss_D_history, label='DCGAN Discriminator Loss', color='blue', alpha=0.5)
        if loss_G_history:
            ax1.plot(loss_G_history, label='DCGAN Generator Loss', color='green', alpha=0.5)

    if phase == 'ddpm' or phase == 'both':
        if ddpm_loss_history:
            ax1.plot(ddpm_loss_history, label='DDPM Loss', color='orange', alpha=0.5)

    ax1.legend(loc='upper left')

    # Plot FID on the right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('FID', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    if phase == 'dcgan' or phase == 'both':
        if fid_dcgan_history:
            ax2.plot(fid_dcgan_history, label='DCGAN FID', color='red', linestyle='--', alpha=0.7)

    if phase == 'ddpm' or phase == 'both':
        if fid_ddpm_history:
            ax2.plot(fid_ddpm_history, label='DDPM FID', color='purple', linestyle='--', alpha=0.7)

    ax2.legend(loc='upper right')

    plt.title(f'Learning Curves and FID at Epoch {epoch} ({phase.upper()})')
    plt.savefig(os.path.join(save_dir, 'images', f'learning_curve_fid_epoch{epoch}_{phase}.png'))
    plt.close()

def compute_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(feature=64).to(real_images.device)
    real_images = (real_images * 255).to(torch.uint8).expand(-1, 3, -1, -1)
    fake_images = (fake_images * 255).to(torch.uint8).expand(-1, 3, -1, -1)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()

def compute_diversity(images):
    images = images.view(images.size(0), -1)
    variance = torch.var(images, dim=0).mean().item()
    return variance