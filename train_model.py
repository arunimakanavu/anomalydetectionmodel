import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

# =============== 1. CONFIG =================
IMG_SIZE = 304
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MODEL_PATH = "casting_autoencoder.pth"
ONNX_PATH = "casting_autoencoder.onnx"

TRAIN_DIR = "casting_data/train"      # only OK parts
TEST_DEFECT_DIR = "casting_data/test"  # defects for thresholding

# =============== 2. DATA PIPELINE =================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# =============== 3. MODEL =================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# =============== 4. TRAINING LOOP =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(" Training started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        output = model(imgs)
        loss = criterion(output, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")

# =============== 5. THRESHOLD CALIBRATION =================
defect_data = datasets.ImageFolder(root=TEST_DEFECT_DIR, transform=transform)
defect_loader = DataLoader(defect_data, batch_size=1)

model.eval()
errors = []
with torch.no_grad():
    for img, _ in defect_loader:
        img = img.to(device)
        out = model(img)
        err = criterion(out, img).item()
        errors.append(err)

threshold = np.mean(errors) * 0.8
print(f"⚡ Suggested anomaly threshold: {threshold:.4f}")

# =============== 6. EXPORT TO ONNX =================
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)
print(f" ONNX model exported to {ONNX_PATH}")
