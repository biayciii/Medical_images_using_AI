import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoding path
        self.d1 = DoubleConv(1, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot = DoubleConv(256, 512)
        
        # Decoding path (Upsampling)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u3 = DoubleConv(128, 64)
        
        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        
        # Bottleneck
        bot = self.bot(self.pool(x3))
        
        # Decoder
        u1 = self.up1(bot)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.u1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.u2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.u3(u3)
        
        # Final output
        return torch.sigmoid(self.final(u3))

def process_and_generate(ct_dir, smri_dir, generator_weights, device):
    print("Loading GAN Generator model...")
    
    model_G = SimpleUNet()
    model_G.load_state_dict(torch.load(generator_weights, map_location=device))
    model_G.to(device)
    model_G.eval()
    
    ct_paths = []
    for root, dirs, files in os.walk(ct_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                ct_paths.append(os.path.join(root, f))
                
    print(f"Found {len(ct_paths)} CT images. Starting generation...")

    with torch.no_grad():
        for ct_path in tqdm(ct_paths, desc="Generating sMRI"):
            rel_path = os.path.relpath(ct_path, ct_dir)
            smri_path = os.path.join(smri_dir, rel_path)
            
            os.makedirs(os.path.dirname(smri_path), exist_ok=True)

            img_ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            
            if img_ct is None:
                continue
                
            img_resized = cv2.resize(img_ct, (256, 256))
            img_norm = img_resized.astype(np.float32) / 255.0
            
            img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)
            
            fake_smri_tensor = model_G(img_tensor)

            fake_smri_img = fake_smri_tensor.cpu().squeeze().numpy()
            fake_smri_img = np.clip(fake_smri_img, 0, 1)
            fake_smri_img = (fake_smri_img * 255.0).astype(np.uint8)

            cv2.imwrite(smri_path, fake_smri_img)

    print(f"Generation complete. sMRI images saved to: {smri_dir}")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CT_DIRECTORY = "/mnt/nvme2/users/utbt_sv1/ct_project/CT"
    SMRI_DIRECTORY = "/mnt/nvme2/users/utbt_sv1/ct_project/content/HaN-Seg_2D_Split/sMRI"
    GAN_WEIGHTS_PATH = "/mnt/nvme2/users/utbt_sv1/ct_project/sMRI/generator_best.pth"

    process_and_generate(CT_DIRECTORY, SMRI_DIRECTORY, GAN_WEIGHTS_PATH, DEVICE)