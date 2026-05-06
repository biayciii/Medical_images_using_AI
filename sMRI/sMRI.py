import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')

from model import SimpleUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ROOT_DIR = 'content/HaN-Seg_2D_Split'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 100
LR = 2e-4

def scan_ct_mri(root_dir):
    ct_dir = os.path.join(root_dir, 'CT')
    mri_dir = os.path.join(root_dir, 'MRI')
    pairs = []
    
    if not os.path.exists(ct_dir) or not os.path.exists(mri_dir):
        return pairs
        
    for case_folder in os.listdir(ct_dir):
        ct_case_path = os.path.join(ct_dir, case_folder)
        mri_case_path = os.path.join(mri_dir, case_folder)
        
        if os.path.isdir(ct_case_path) and os.path.isdir(mri_case_path):
            for f in os.listdir(ct_case_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ct_img_path = os.path.join(ct_case_path, f)
                    mri_img_path = os.path.join(mri_case_path, f)
                    
                    if os.path.exists(mri_img_path):
                        pairs.append((ct_img_path, mri_img_path))
    return pairs

class CTMriDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mri_path = self.pairs[idx]
        
        ct = cv2.imread(ct_path, 0)
        mri = cv2.imread(mri_path, 0)
        
        ct = cv2.resize(ct, (256, 256))
        mri = cv2.resize(mri, (256, 256))
        
        ct = torch.tensor(ct, dtype=torch.float32).unsqueeze(0) / 255.0
        mri = torch.tensor(mri, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return ct, mri

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

if __name__ == "__main__":
    print(f"Starting GAN training on {DEVICE}...")
    
    all_pairs = scan_ct_mri(ROOT_DIR)
    
    if len(all_pairs) == 0:
        print("Error: No valid paired CT-MRI data found.")
        exit()
        
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    
    train_ds = CTMriDataset(train_pairs)
    val_ds = CTMriDataset(val_pairs)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Split: {len(train_pairs)} Train | {len(val_pairs)} Val")

    generator = SimpleUNet().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    lambda_l1 = 100
    best_val_loss = float('inf')
    
    print("-" * 75)
    print(f"{'Epoch':^7} | {'D Loss':^15} | {'G Loss':^15} | {'Val L1 Loss':^15}")
    print("-" * 75)

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        d_loss_total = 0
        g_loss_total = 0
        
        for ct, mri in train_loader:
            ct, mri = ct.to(DEVICE), mri.to(DEVICE)
            
            valid = torch.ones((ct.size(0), 1, 30, 30), device=DEVICE, requires_grad=False)
            fake = torch.zeros((ct.size(0), 1, 30, 30), device=DEVICE, requires_grad=False)
            
            fake_mri = generator(ct)
            
            optimizer_d.zero_grad()
            real_loss = criterion_gan(discriminator(ct, mri), valid)
            fake_loss = criterion_gan(discriminator(ct, fake_mri.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            
            optimizer_g.zero_grad()
            g_gan_loss = criterion_gan(discriminator(ct, fake_mri), valid)
            g_l1_loss = criterion_l1(fake_mri, mri)
            g_loss = g_gan_loss + lambda_l1 * g_l1_loss
            g_loss.backward()
            optimizer_g.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            
        generator.eval()
        val_l1_total = 0
        
        with torch.no_grad():
            for ct_val, mri_val in val_loader:
                ct_val, mri_val = ct_val.to(DEVICE), mri_val.to(DEVICE)
                fake_val = generator(ct_val)
                val_l1_total += criterion_l1(fake_val, mri_val).item()
                
        avg_d_loss = d_loss_total / len(train_loader)
        avg_g_loss = g_loss_total / len(train_loader)
        avg_val_l1 = val_l1_total / len(val_loader)
        
        print(f"{epoch+1:^7} | {avg_d_loss:^15.4f} | {avg_g_loss:^15.4f} | {avg_val_l1:^15.4f}")
        
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            torch.save(generator.state_dict(), 'generator_best.pth')

    print("-" * 75)
    print(f"GAN Training Complete! Best Val L1 Loss: {best_val_loss:.4f}")