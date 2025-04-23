#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms, utils as vutils
from PIL import Image
from flask import Flask, request

# -------------------------
# Model Definitions
# -------------------------

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
        # Concatenate noise and label embedding
        label_embed = self.label_embed(labels)
        input_vec = torch.cat([z, label_embed], dim=1)
        input_vec = input_vec.view(-1, self.z_dim + self.num_classes, 1, 1)
        return self.main(input_vec)

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
        # Expand label embedding and concatenate with the input image
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

    def forward_diffusion(self, x0):
        # Randomly choose a diffusion time step and add noise to x0
        t = torch.randint(0, self.T, (x0.shape[0],)).to(self.device)
        epsilon = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return xt, t, epsilon

    def train_step(self, x0, labels):
        xt, t, epsilon = self.forward_diffusion(x0)
        predicted_epsilon = self.model(xt, labels)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_epsilon, epsilon)
        return loss

    def sample(self, num_samples, labels):
        # Generate samples using reverse diffusion process
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

# -------------------------
# Load Pretrained Models
# -------------------------

# Paths for the pretrained model files
SAVE_DIR = r"C:/Users/Daniel ZHAO/AIBA/foundation_ai/Essay"
DCGAN_MODEL_PATH = os.path.join(SAVE_DIR, "dcgan_generator_best.pth")
DDPM_MODEL_PATH = os.path.join(SAVE_DIR, "ddpm_unet_best.pth")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DCGAN Generator
dcgan_generator = Generator(z_dim=100, num_classes=10).to(device)
if os.path.exists(DCGAN_MODEL_PATH):
    dcgan_generator.load_state_dict(torch.load(DCGAN_MODEL_PATH, map_location=device))
    dcgan_generator.eval()
    print("DCGAN Generator loaded successfully!")
else:
    print(f"DCGAN model file not found: {DCGAN_MODEL_PATH}")

# Load DDPM model (load U-Net and build the diffusion model)
ddpm_unet = ConditionalUNet(num_classes=10).to(device)
if os.path.exists(DDPM_MODEL_PATH):
    ddpm_unet.load_state_dict(torch.load(DDPM_MODEL_PATH, map_location=device))
    ddpm_unet.eval()
    print("DDPM U-Net loaded successfully!")
else:
    print(f"DDPM model file not found: {DDPM_MODEL_PATH}")
# Using T=100; adjust this if needed to match your training parameters
diffusion_model = DiffusionModel(ddpm_unet, device, T=100)

# -------------------------
# Flask Application
# -------------------------

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page with enhanced styling
    html_content = '''
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>MNIST Generation App</title>
        <style>
          body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background: #f7f7f7;
            margin: 0;
            padding: 0;
          }
          .container {
            max-width: 600px;
            margin: 50px auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 30px;
          }
          h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
          }
          form {
            display: flex;
            flex-direction: column;
          }
          fieldset {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px 15px;
            margin-bottom: 20px;
          }
          legend {
            font-weight: bold;
            margin-bottom: 10px;
          }
          label {
            margin-bottom: 8px;
            color: #555;
          }
          input[type="text"],
          input[type="number"] {
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
          }
          input[type="radio"] {
            margin-right: 6px;
          }
          .radio-group label {
            display: block;
            margin-bottom: 10px;
          }
          input[type="submit"] {
            padding: 12px;
            font-size: 16px;
            color: #fff;
            background-color: #007bff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s ease;
          }
          input[type="submit"]:hover {
            background-color: #0056b3;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>MNIST Generation App</h1>
          <form action="/generate" method="post">
            <fieldset>
              <legend>Select Generation Model</legend>
              <div class="radio-group">
                <label><input type="radio" name="model" value="dcgan" checked> DCGAN</label>
                <label><input type="radio" name="model" value="ddpm"> DDPM</label>
              </div>
            </fieldset>
            <label for="digit">Specific digit to generate (0-9, leave blank for random):</label>
            <input type="text" id="digit" name="digit" placeholder="E.g., 3">
            <label for="n_samples">Number of samples (default 8):</label>
            <input type="number" id="n_samples" name="n_samples" value="8" min="1" max="64">
            <input type="submit" value="Generate Images">
          </form>
        </div>
      </body>
    </html>
    '''
    return html_content

@app.route('/generate', methods=['POST'])
def generate():
    # Get form parameters
    model_type = request.form.get('model', 'dcgan')
    digit = request.form.get('digit', '').strip()
    try:
        n_samples = int(request.form.get('n_samples', 8))
    except Exception:
        n_samples = 8

    # Parse the specific digit; if invalid use random labels
    try:
        digit_val = int(digit)
        if digit_val < 0 or digit_val > 9:
            digit_val = None
    except Exception:
        digit_val = None

    if digit_val is None:
        labels = torch.randint(0, 10, (n_samples,), device=device)
    else:
        labels = torch.full((n_samples,), digit_val, dtype=torch.long, device=device)

    # Generate images using the selected model
    if model_type == "dcgan":
        z = torch.randn(n_samples, 100, device=device)
        with torch.no_grad():
            generated = dcgan_generator(z, labels)
    else:
        with torch.no_grad():
            generated = diffusion_model.sample(n_samples, labels)

    # Create a grid of generated images then convert to PIL image
    grid_img = vutils.make_grid(generated.cpu(), nrow=4, normalize=True)
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(grid_img)

    # Encode image to base64 string for embedding in HTML
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    html_result = f'''
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Generation Result</title>
        <style>
          body {{
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background: #f7f7f7;
            margin: 0;
            padding: 0;
          }}
          .container {{
            max-width: 800px;
            margin: 50px auto;
            background: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
          }}
          h1 {{
            color: #333;
            margin-bottom: 20px;
          }}
          img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
          }}
          a {{
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007bff;
            font-weight: bold;
          }}
          a:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Generated Images using {model_type.upper()}</h1>
          <img src="data:image/png;base64,{img_base64}" alt="Generated Images">
          <br>
          <a href="/">Go Back</a>
        </div>
      </body>
    </html>
    '''
    return html_result

if __name__ == '__main__':
    # Launch the Flask application (adjust host/port as needed)
    app.run(host='0.0.0.0', port=5000, debug=True)