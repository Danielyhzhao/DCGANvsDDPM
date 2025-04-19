# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator, ConditionalUNet, DiffusionModel
from dataset import load_mnist
from utils import save_images, plot_learning_curves, compute_fid, compute_diversity
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = 'C:/Users/Daniel ZHAO/AIBA/foundation_ai/Essay'

def main():
    z_dim = 100
    num_classes = 10
    batch_size = 64
    num_epochs_dcgan = 80
    num_epochs_ddpm = 80
    lr_dcgan = 0.0001
    lr_ddpm = 2e-5
    beta1, beta2 = 0.5, 0.999
    T = 100

    train_loader = load_mnist(batch_size, data_dir=os.path.join(save_dir, 'data'))

    generator = Generator(z_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)
    unet = ConditionalUNet(num_classes).to(device)
    diffusion = DiffusionModel(unet, device, T)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_dcgan, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_dcgan, betas=(beta1, beta2))
    optimizer_diffusion = optim.Adam(unet.parameters(), lr=lr_ddpm)

    if not os.path.exists(os.path.join(save_dir, 'images')):
        os.makedirs(os.path.join(save_dir, 'images'))

    loss_D_history = []
    loss_G_history = []
    fid_dcgan_history = []
    ddpm_loss_history = []
    fid_ddpm_history = []
    metrics = {'dcgan': {}, 'ddpm': {}}

    # Get a fixed batch of real images for FID computation
    real_images_fixed, _ = next(iter(train_loader))
    real_images_fixed = real_images_fixed.to(device)

    # DCGAN Training
    dcgan_start_time = time.time()
    for epoch in range(num_epochs_dcgan):
        for i, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizer_D.zero_grad()
            output = discriminator(real_images, labels)
            loss_D_real = criterion(output, real_labels)
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z, labels)
            output = discriminator(fake_images.detach(), labels)
            loss_D_fake = criterion(output, fake_labels)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            output = discriminator(fake_images, labels)
            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizer_G.step()

            loss_D_history.append(loss_D.item())
            loss_G_history.append(loss_G.item())

            if i % 100 == 0:
                print(f'DCGAN Epoch [{epoch}/{num_epochs_dcgan}], Step [{i}/{len(train_loader)}], '
                      f'Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')

        with torch.no_grad():
            z = torch.randn(8, z_dim).to(device)
            labels = torch.randint(0, num_classes, (8,)).to(device)
            fake_images = generator(z, labels).cpu()
            save_images(fake_images, os.path.join(save_dir, 'images', f'dcgan_epoch{epoch}.png'))
            # Compute FID for this epoch
            fid = compute_fid(real_images_fixed.cpu(), fake_images)
            fid_dcgan_history.append(fid)
            if epoch == num_epochs_dcgan - 1:
                diversity = compute_diversity(fake_images)
                metrics['dcgan']['fid'] = fid
                metrics['dcgan']['diversity'] = diversity
        plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, ddpm_loss_history, fid_ddpm_history, epoch, save_dir, phase='dcgan')

    dcgan_time = time.time() - dcgan_start_time
    metrics['dcgan']['training_time'] = dcgan_time

    # DDPM Training
    ddpm_start_time = time.time()
    for epoch in range(num_epochs_ddpm):
        for i, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            loss = diffusion.train_step(real_images, labels)
            optimizer_diffusion.zero_grad()
            loss.backward()
            optimizer_diffusion.step()

            ddpm_loss_history.append(loss.item())

            if i % 100 == 0:
                print(f'DDPM Epoch [{epoch}/{num_epochs_ddpm}], Step [{i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        with torch.no_grad():
            labels = torch.randint(0, num_classes, (8,)).to(device)
            ddpm_images = diffusion.sample(8, labels).cpu()
            save_images(ddpm_images, os.path.join(save_dir, 'images', f'ddpm_epoch{epoch}.png'))
            # Compute FID for this epoch
            fid = compute_fid(real_images_fixed.cpu(), ddpm_images)
            fid_ddpm_history.append(fid)
            if epoch == num_epochs_ddpm - 1:
                diversity = compute_diversity(ddpm_images)
                metrics['ddpm']['fid'] = fid
                metrics['ddpm']['diversity'] = diversity
        plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, ddpm_loss_history, fid_ddpm_history, epoch, save_dir, phase='ddpm')

    ddpm_time = time.time() - ddpm_start_time
    metrics['ddpm']['training_time'] = ddpm_time

    # Final combined plot
    plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, ddpm_loss_history, fid_ddpm_history, 'final', save_dir, phase='both')

    torch.save(generator.state_dict(), os.path.join(save_dir, 'dcgan_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'dcgan_discriminator.pth'))
    torch.save(unet.state_dict(), os.path.join(save_dir, 'ddpm_unet.pth'))

    with open(os.path.join(save_dir, 'metrics_history.txt'), 'w') as f:
        f.write(f"DCGAN Training Time: {metrics['dcgan']['training_time']:.2f} seconds\n")
        f.write(f"DCGAN FID: {metrics['dcgan']['fid']:.2f}\n")
        f.write(f"DCGAN Diversity: {metrics['dcgan']['diversity']:.4f}\n")
        f.write(f"DDPM Training Time: {metrics['ddpm']['training_time']:.2f} seconds\n")
        f.write(f"DDPM FID: {metrics['ddpm']['fid']:.2f}\n")
        f.write(f"DDPM Diversity: {metrics['ddpm']['diversity']:.4f}\n")

if __name__ == '__main__':
    main()