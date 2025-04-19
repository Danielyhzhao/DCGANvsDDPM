import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 512, 7, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_embed(labels)
        input_vec = torch.cat([z, label_embed], dim=1)
        input_vec = input_vec.view(-1, self.z_dim + self.num_classes, 1, 1)
        return self.main(input_vec)

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, num_classes * 28 * 28)
        self.main = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embed = self.label_embed(labels).view(-1, self.num_classes, 28, 28)
        input_tensor = torch.cat([img, label_embed], dim=1)
        return self.main(input_tensor)

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalUNet, self).__init__()
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.conv1 = nn.Conv2d(1 + num_classes, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, labels):
        label_embed = self.label_embed(labels)
        batch_size, _, H, W = x.shape
        label_embed = label_embed.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        x = torch.cat([x, label_embed], dim=1)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.conv6(x5)
        return x6

class DiffusionModel:
    def __init__(self, model, device, T=100):
        self.model = model.to(device)
        self.device = device
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],)).to(self.device)
        epsilon = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return xt, t, epsilon

    def train_step(self, x0, labels):
        xt, t, epsilon = self.forward(x0)
        predicted_epsilon = self.model(xt, labels)
        loss = nn.MSELoss()(predicted_epsilon, epsilon)
        return loss

    def sample(self, num_samples, labels):
        x = torch.randn(num_samples, 1, 28, 28).to(self.device)
        labels = labels.to(self.device)
        for t in range(self.T - 1, -1, -1):
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            noise = torch.randn_like(x) if t > 0 else 0
            predicted_epsilon = self.model(x, labels)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_epsilon) + torch.sqrt(beta_t) * noise
        return x