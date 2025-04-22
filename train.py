import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator, ConditionalUNet, DiffusionModel
from dataset import load_mnist
from utils import save_images, plot_learning_curves, compute_fid, compute_diversity, plot_comparison_metrics, plot_parameter_comparison
import time
import os
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = 'C:/Users/Daniel ZHAO/AIBA/foundation_ai/Essay'

def train_dcgan(train_loader, z_dim, num_classes, num_epochs, lr, beta1, real_images_fixed, param_str, base_dir, device):
    generator = Generator(z_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)

    param_dir = os.path.join(base_dir, f'dcgan_{param_str}')
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    loss_D_history = []
    loss_G_history = []
    fid_dcgan_history = []

    start_time = time.time()
    for epoch in range(num_epochs):
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
                print(f'DCGAN Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                      f'Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, Params: {param_str}')

        with torch.no_grad():
            z = torch.randn(8, z_dim).to(device)
            labels = torch.randint(0, num_classes, (8,)).to(device)
            fake_images = generator(z, labels)
            save_images(fake_images.cpu(), f'dcgan_epoch{epoch}.png', param_dir)
            fid = compute_fid(real_images_fixed, fake_images)
            fid_dcgan_history.append(fid)

        plot_learning_curves(loss_D_history, loss_G_history, fid_dcgan_history, [], [], epoch, param_dir, phase='dcgan', param_str=param_str)

    training_time = time.time() - start_time
    diversity = compute_diversity(fake_images.cpu())

    return generator, discriminator, fid, diversity, training_time, param_dir

def train_ddpm(train_loader, num_classes, num_epochs, lr, T, real_images_fixed, param_str, base_dir, device):
    unet = ConditionalUNet(num_classes).to(device)
    diffusion = DiffusionModel(unet, device, T)
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    param_dir = os.path.join(base_dir, f'ddpm_{param_str}')
    ddpm_loss_history = []
    fid_ddpm_history = []

    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            loss = diffusion.train_step(real_images, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ddpm_loss_history.append(loss.item())

            if i % 100 == 0:
                print(f'DDPM Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Params: {param_str}')

        with torch.no_grad():
            labels = torch.randint(0, num_classes, (8,)).to(device)
            ddpm_images = diffusion.sample(8, labels)
            save_images(ddpm_images.cpu(), f'ddpm_epoch{epoch}.png', param_dir)
            fid = compute_fid(real_images_fixed, ddpm_images)
            fid_ddpm_history.append(fid)

        plot_learning_curves([], [], [], ddpm_loss_history, fid_ddpm_history, epoch, param_dir, phase='ddpm', param_str=param_str)

    training_time = time.time() - start_time
    diversity = compute_diversity(ddpm_images.cpu())

    return unet, diffusion, fid, diversity, training_time, param_dir

def save_best_model_results(generator, unet, diffusion, num_classes, best_dir, device):
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    with torch.no_grad():
        z = torch.randn(8, 100).to(device)
        labels = torch.randint(0, num_classes, (8,)).to(device)
        fake_images = generator(z, labels)
        save_images(fake_images.cpu(), 'best_dcgan.png', best_dir)

    with torch.no_grad():
        labels = torch.randint(0, num_classes, (8,)).to(device)
        ddpm_images = diffusion.sample(8, labels)
        save_images(ddpm_images.cpu(), 'best_ddpm.png', best_dir)

def main():
    z_dim = 100
    num_classes = 10
    batch_size = 1024
    num_epochs_dcgan = 50
    num_epochs_ddpm = 50

    dcgan_lr_grid = [0.0001, 0.0002, 0.0004]
    dcgan_beta1_grid = [0.3, 0.5, 0.7]
    ddpm_lr_grid = [1e-5, 2e-5, 5e-5]
    ddpm_T_grid = [50, 100, 200]

    train_loader = load_mnist(batch_size, data_dir=os.path.join(save_dir, 'data'))
    images_base_dir = os.path.join(save_dir, 'images')
    best_dir = os.path.join(save_dir, 'best_models')

    if not os.path.exists(images_base_dir):
        os.makedirs(images_base_dir)

    real_images_fixed, _ = next(iter(train_loader))
    real_images_fixed = real_images_fixed.to(device)

    dcgan_results = []
    for lr, beta1 in itertools.product(dcgan_lr_grid, dcgan_beta1_grid):
        param_str = f'lr{lr}_beta1{beta1}'
        print(f"Training DCGAN with {param_str}")
        generator, discriminator, fid, diversity, training_time, param_dir = train_dcgan(
            train_loader, z_dim, num_classes, num_epochs_dcgan, lr, beta1, real_images_fixed, param_str, images_base_dir, device
        )
        dcgan_results.append({
            'params': param_str,
            'fid': fid,
            'diversity': diversity,
            'training_time': training_time,
            'generator': generator,
            'discriminator': discriminator,
            'param_dir': param_dir
        })

    best_dcgan = min(dcgan_results, key=lambda x: x['fid'])
    print(f"Best DCGAN params: {best_dcgan['params']}, FID: {best_dcgan['fid']}")
    torch.save(best_dcgan['generator'].state_dict(), os.path.join(save_dir, 'dcgan_generator_best.pth'))
    torch.save(best_dcgan['discriminator'].state_dict(), os.path.join(save_dir, 'dcgan_discriminator_best.pth'))
    plot_parameter_comparison(dcgan_results, 'DCGAN', save_dir)

    ddpm_results = []
    for lr, T in itertools.product(ddpm_lr_grid, ddpm_T_grid):
        param_str = f'lr{lr}_T{T}'
        print(f"Training DDPM with {param_str}")
        unet, diffusion, fid, diversity, training_time, param_dir = train_ddpm(
            train_loader, num_classes, num_epochs_ddpm, lr, T, real_images_fixed, param_str, images_base_dir, device
        )
        ddpm_results.append({
            'params': param_str,
            'fid': fid,
            'diversity': diversity,
            'training_time': training_time,
            'unet': unet,
            'diffusion': diffusion,
            'param_dir': param_dir
        })

    best_ddpm = min(ddpm_results, key=lambda x: x['fid'])
    print(f"Best DDPM params: {best_ddpm['params']}, FID: {best_ddpm['fid']}")
    torch.save(best_ddpm['unet'].state_dict(), os.path.join(save_dir, 'ddpm_unet_best.pth'))
    plot_parameter_comparison(ddpm_results, 'DDPM', save_dir)

    save_best_model_results(best_dcgan['generator'], best_ddpm['unet'], best_ddpm['diffusion'], num_classes, best_dir, device)

    plot_comparison_metrics(
        {'fid': best_dcgan['fid'], 'diversity': best_dcgan['diversity'], 'training_time': best_dcgan['training_time']},
        {'fid': best_ddpm['fid'], 'diversity': best_ddpm['diversity'], 'training_time': best_ddpm['training_time']},
        save_dir
    )

    with open(os.path.join(save_dir, 'metrics_comparison.txt'), 'w') as f:
        f.write(f"Best DCGAN Params: {best_dcgan['params']}\n")
        f.write(f"DCGAN FID: {best_dcgan['fid']:.2f}\n")
        f.write(f"DCGAN Diversity: {best_dcgan['diversity']:.4f}\n")
        f.write(f"DCGAN Training Time: {best_dcgan['training_time']:.2f} seconds\n")
        f.write(f"Best DDPM Params: {best_ddpm['params']}\n")
        f.write(f"DDPM FID: {best_ddpm['fid']:.2f}\n")
        f.write(f"DDPM Diversity: {best_ddpm['diversity']:.4f}\n")
        f.write(f"DDPM Training Time: {best_ddpm['training_time']:.2f} seconds\n")

if __name__ == '__main__':
    main()