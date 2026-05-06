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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = '/mnt/nvme0/home/utbt_sv1/ct_project/content/HaN-Seg_2D_Split'
PROJECT_DIR = '/mnt/nvme0/home/utbt_sv1/ct_project/sMRI/'
SAVE_MODEL_PATH = os.path.join(PROJECT_DIR, 'best.pth')

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth))
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_metrics(pred, target, threshold=0.5):
    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target > threshold).float()
    correct = (pred_bin == target_bin).sum()
    acc = correct / torch.numel(pred_bin)
    intersection = (pred_bin * target_bin).sum()
    f1 = (2. * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)
    return acc.item(), f1.item()

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, val_acc, val_f1 = 0, 0, 0
    with torch.no_grad():
        for ct, smri, target in loader:
            ct, smri, target = ct.to(device), smri.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(ct, smri)
                loss = criterion(output, target)
            acc, f1 = calculate_metrics(output, target)
            val_loss += loss.item()
            val_acc += acc
            val_f1 += f1
    n = len(loader)
    return val_loss/n, val_acc/n, val_f1/n

def scan_triplets(root_dir):
    ct_dir = os.path.join(root_dir, 'CT')
    smri_dir = os.path.join(root_dir, 'sMRI')
    mask_dir = os.path.join(root_dir, 'Mask_2D')

    triplets = []
    if not os.path.exists(ct_dir) or not os.path.exists(smri_dir) or not os.path.exists(mask_dir):
        return triplets

    for case_folder in os.listdir(ct_dir):
        ct_case = os.path.join(ct_dir, case_folder)
        smri_case = os.path.join(smri_dir, case_folder)
        mask_case = os.path.join(mask_dir, case_folder)
        
        if os.path.isdir(ct_case) and os.path.isdir(smri_case) and os.path.isdir(mask_case):
            for f in os.listdir(ct_case):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ct_path = os.path.join(ct_case, f)
                    smri_path = os.path.join(smri_case, f)
                    mask_path = os.path.join(mask_case, f)
                    
                    if os.path.exists(smri_path) and os.path.exists(mask_path):
                        triplets.append((ct_path, smri_path, mask_path))
    return triplets

class FusionDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        ct_path, smri_path, mask_path = self.triplets[idx]
        
        ct = cv2.imread(ct_path, 0)
        smri = cv2.imread(smri_path, 0)
        mask = cv2.imread(mask_path, 0)

        ct = cv2.resize(ct, (256, 256))
        smri = cv2.resize(smri, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        ct = torch.tensor(ct, dtype=torch.float32).unsqueeze(0) / 255.0
        smri = torch.tensor(smri, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        return ct, smri, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_query, x_key_value):
        batch_size, C, width, height = x_query.size()
        
        proj_query = self.query_conv(x_query).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_key_value).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x_key_value).view(batch_size, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x_query
        return out

class ProposedLateFusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        
        self.encoder1_ct_1 = DoubleConv(1, 32)
        self.encoder1_ct_2 = DoubleConv(32, 64)
        self.encoder1_ct_3 = DoubleConv(64, 128)
        self.encoder1_ct_4 = DoubleConv(128, 256)
        
        self.encoder2_mri_1 = DoubleConv(1, 32)
        self.encoder2_mri_2 = DoubleConv(32, 64)
        self.encoder2_mri_3 = DoubleConv(64, 128)
        self.encoder2_mri_4 = DoubleConv(128, 256)
        
        self.cross_attention = CrossAttention(in_channels=256)
        
        self.fusion_layer = DoubleConv(512, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4 = DoubleConv(256 + 256 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = DoubleConv(128 + 128 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = DoubleConv(64 + 64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = DoubleConv(32 + 32 + 32, 32)
        
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, ct, mri):
        c1 = self.encoder1_ct_1(ct)
        c2 = self.encoder1_ct_2(self.pool(c1))
        c3 = self.encoder1_ct_3(self.pool(c2))
        c4 = self.encoder1_ct_4(self.pool(c3))
        
        m1 = self.encoder2_mri_1(mri)
        m2 = self.encoder2_mri_2(self.pool(m1))
        m3 = self.encoder2_mri_3(self.pool(m2))
        m4 = self.encoder2_mri_4(self.pool(m3))
        
        pool_c4 = self.pool(c4)
        pool_m4 = self.pool(m4)
        
        attn_c4 = self.cross_attention(x_query=pool_c4, x_key_value=pool_m4)
        
        fused = torch.cat([attn_c4, pool_m4], dim=1)
        b = self.fusion_layer(fused)
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, c4, m4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, c3, m3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, c2, m2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, c1, m1], dim=1)
        d1 = self.decoder1(d1)
        
        return self.out(d1)

if __name__ == "__main__":
    print(f"Starting on device: {DEVICE}...")
    
    all_triplets = scan_triplets(ROOT_DIR)
    
    if len(all_triplets) == 0:
        print(f"Data not found at: {ROOT_DIR}. Please check the path.")
        exit()
        
    train_triplets, val_triplets = train_test_split(all_triplets, test_size=0.2, random_state=42)
    
    train_ds = FusionDataset(train_triplets)
    val_ds = FusionDataset(val_triplets)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Total data: {len(train_triplets)} Train | {len(val_triplets)} Val")

    model = ProposedLateFusionUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss(smooth=1.0, bce_weight=0.5, dice_weight=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()
    
    best_score = 0
    best_train_loss = 0
    best_val_loss = 0
    best_pixel_acc = 0
    best_train_dice = 0
    
    print("-" * 90)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Train Acc %':^12} | {'Train Dice %':^12} | {'Val Dice %':^12}")
    print("-" * 90)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc_total = 0
        train_dice_total = 0
        
        for ct, smri, target in train_loader:
            ct, smri, target = ct.to(DEVICE), smri.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(ct, smri)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            acc, f1 = calculate_metrics(output, target)
            train_acc_total += acc
            train_dice_total += f1
            
        n_train = len(train_loader)
        avg_train_loss = train_loss / n_train
        avg_train_acc = (train_acc_total / n_train) * 100
        avg_train_dice = (train_dice_total / n_train) * 100
        
        avg_val_loss, avg_val_acc_raw, avg_val_f1_raw = validate(model, val_loader, criterion, DEVICE)
        avg_val_dice = avg_val_f1_raw * 100

        scheduler.step(avg_val_dice)

        print(f"{epoch+1:^7} | {avg_train_loss:^12.4f} | {avg_val_loss:^12.4f} | {avg_train_acc:^12.2f} | {avg_train_dice:^12.2f} | {avg_val_dice:^12.2f}")

        if avg_val_dice > best_score:
            best_score = avg_val_dice
            best_train_loss = avg_train_loss
            best_val_loss = avg_val_loss
            best_pixel_acc = avg_val_acc_raw * 100
            best_train_dice = avg_train_dice
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    print("-" * 90)
    print("Final Model Performance:")
    print(f"Train Loss       : {best_train_loss:.4f}")
    print(f"Val Loss         : {best_val_loss:.4f}")
    print(f"Pixel Acc (%)    : {best_pixel_acc:.2f}")
    print(f"Train Dice (%)   : {best_train_dice:.2f}")
    print(f"Best Val Dice (%): {best_score:.2f}")