from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_mnist(batch_size=64, data_dir='C:/Users/Daniel ZHAO/AIBA/foundation_ai/Essay/data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return dataloader