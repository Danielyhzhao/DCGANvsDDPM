import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import os

def save_images(images, filename, param_dir):
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)
    vutils.save_image(images, os.path.join(param_dir, filename), normalize=True, nrow=4)

def plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, ddpm_loss_history, fid_ddpm_history, epoch, param_dir, phase='dcgan', param_str=''):
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    if phase in ['dcgan', 'both']:
        if loss_D_history:
            ax1.plot(loss_D_history, label='DCGAN Discriminator Loss', color='blue', alpha=0.5)
        if loss_G_history:
            ax1.plot(loss_G_history, label='DCGAN Generator Loss', color='green', alpha=0.5)

    if phase in ['ddpm', 'both']:
        if ddpm_loss_history:
            ax1.plot(ddpm_loss_history, label='DDPM Loss', color='orange', alpha=0.5)

    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('FID', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    if phase in ['dcgan', 'both']:
        if fid_dcgan_history:
            ax2.plot(fid_dcgan_history, label='DCGAN FID', color='red', linestyle='--', alpha=0.7)

    if phase in ['ddpm', 'both']:
        if fid_ddpm_history:
            ax2.plot(fid_ddpm_history, label='DDPM FID', color='purple', linestyle='--', alpha=0.7)

    ax2.legend(loc='upper right')
    plt.title(f'Learning Curves and FID at Epoch {epoch} ({phase.upper()}) {param_str}')
    plt.savefig(os.path.join(param_dir, f'learning_curve_fid_epoch{epoch}_{phase}.png'))
    plt.close()

def plot_comparison_metrics(dcgan_metrics, ddpm_metrics, save_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    models = ['DCGAN', 'DDPM']
    fid_values = [dcgan_metrics['fid'], ddpm_metrics['fid']]
    ax1.bar(models, fid_values, color=['blue', 'orange'])
    ax1.set_title('FID Comparison')
    ax1.set_ylabel('FID')

    diversity_values = [dcgan_metrics['diversity'], ddpm_metrics['diversity']]
    ax2.bar(models, diversity_values, color=['blue', 'orange'])
    ax2.set_title('Diversity Comparison')
    ax2.set_ylabel('Diversity')

    training_times = [dcgan_metrics['training_time'], ddpm_metrics['training_time']]
    ax3.bar(models, training_times, color=['blue', 'orange'])
    ax3.set_title('Training Time Comparison')
    ax3.set_ylabel('Training Time (seconds)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'images', 'best_model_comparison.png'))
    plt.close()

def plot_parameter_comparison(results, model_name, save_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    params = [res['params'] for res in results]
    fid_values = [res['fid'] for res in results]
    ax1.bar(params, fid_values, color='blue' if model_name == 'DCGAN' else 'orange')
    ax1.set_title(f'{model_name} FID Comparison')
    ax1.set_ylabel('FID')
    ax1.tick_params(axis='x', rotation=45)

    diversity_values = [res['diversity'] for res in results]
    ax2.bar(params, diversity_values, color='blue' if model_name == 'DCGAN' else 'orange')
    ax2.set_title(f'{model_name} Diversity Comparison')
    ax2.set_ylabel('Diversity')
    ax2.tick_params(axis='x', rotation=45)

    training_times = [res['training_time'] for res in results]
    ax3.bar(params, training_times, color='blue' if model_name == 'DCGAN' else 'orange')
    ax3.set_title(f'{model_name} Training Time Comparison')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'images', f'{model_name.lower()}_parameter_comparison.png'))
    plt.close()

def compute_fid(real_images, fake_images, num_samples=8):

    real_images = real_images.cpu()[:num_samples]
    fake_images = fake_images.cpu()[:num_samples]
    real_images = ((real_images + 1) / 2 * 255).to(torch.uint8).expand(-1, 3, -1, -1)
    fake_images = ((fake_images + 1) / 2 * 255).to(torch.uint8).expand(-1, 3, -1, -1)
    fid = FrechetInceptionDistance(feature=64)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()

def compute_diversity(images):
    images = images.view(images.size(0), -1)
    variance = torch.var(images, dim=0).mean().item()
    return variance