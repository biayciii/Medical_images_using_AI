import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast  
import numpy as np
from tqdm import tqdm
from glob import glob
import warnings

warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True 

# --- Directory Configuration ---
CT_BASE = "/mnt/nvme2/users/utbt_sv1/ct_project/CT"
MASK_BASE = "/mnt/nvme2/users/utbt_sv1/ct_project/MASK/anh_mask"
SMRI_BASE = "/mnt/nvme2/users/utbt_sv1/ct_project/content/HaN-Seg_2D_Split/sMRI"
SAVE_PATH = "/mnt/nvme2/users/utbt_sv1/ct_project/late_fusion_best2.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16 
LR = 1e-4
EPOCHS = 100

from sMRI.model import DualStreamLateFusionUNet

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice
# ==========================================

class PDDCAFusionDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        p = self.triplets[idx]
        
        ct = cv2.imread(p['ct'], 0)
        smri = cv2.imread(p['smri'], 0)
        
        ct = cv2.resize(ct, (256, 256)).astype(np.float32) / 255.0
        smri = cv2.resize(smri, (256, 256)).astype(np.float32) / 255.0

        final_mask = np.zeros((256, 256), dtype=np.float32)
        if p['mask_list']:
            for m_path in p['mask_list']:
                m_img = cv2.imread(m_path, 0)
                if m_img is not None:
                    m_img = cv2.resize(m_img, (256, 256))
                    final_mask = np.maximum(final_mask, (m_img > 127).astype(np.float32))
        
        return (torch.tensor(ct).unsqueeze(0), 
                torch.tensor(smri).unsqueeze(0), 
                torch.tensor(final_mask).unsqueeze(0))

def get_triplets(limit_cases=10):
    triplets = []
    experts = ['oncologist', 'radiographer']
    
    case_count = 0
    for expert in experts:
        expert_ct_path = os.path.join(CT_BASE, expert)
        if not os.path.exists(expert_ct_path):
            continue
            
        patient_ids = sorted(os.listdir(expert_ct_path))
        
        for pid in patient_ids:
            if case_count >= limit_cases:
                break
                
            p_ct_dir = os.path.join(CT_BASE, expert, pid)
            p_smri_dir = os.path.join(SMRI_BASE, expert, pid)
            p_mask_root = os.path.join(MASK_BASE, expert, pid)
            
            ct_files = sorted(glob(os.path.join(p_ct_dir, "*.png")))
            smri_files = sorted(glob(os.path.join(p_smri_dir, "*.png")))
            
            if not ct_files or not smri_files:
                continue

            num_slices = min(len(ct_files), len(smri_files))
            
            mask_map = {}
            if os.path.exists(p_mask_root):
                all_mask_paths = glob(os.path.join(p_mask_root, "masks_*", "*.png"))
                for m_file in all_mask_paths:
                    fname = os.path.basename(m_file)
                    if fname not in mask_map:
                        mask_map[fname] = []
                    mask_map[fname].append(m_file)

            for i in range(num_slices):
                ct_path = ct_files[i]
                fname = os.path.basename(ct_path)
                
                triplets.append({
                    'ct': ct_path,
                    'smri': smri_files[i],
                    'mask_list': mask_map.get(fname, [])
                })
            
            print(f"Loaded: {pid} ({expert}) | Slices: {num_slices}")
            case_count += 1
            
    return triplets

if __name__ == "__main__":
    data_triplets = get_triplets(limit_cases=20) # L?y 20 ca
    
    if not data_triplets:
        print("Error: No valid data triplets found. Check your directory paths.")
        exit()

    dataset = PDDCAFusionDataset(data_triplets)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,          
        pin_memory=True,         
        prefetch_factor=2,      
        persistent_workers=True 
    )

    model = DualStreamLateFusionUNet(in_channels=1, out_classes=1).to(DEVICE)
    
    criterion = DiceLoss()
    # ==========================================
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    scaler = GradScaler()

    print(f"Starting FAST Training on {DEVICE} with {len(data_triplets)} total slices...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for ct, smri, mask in pbar:
            ct = ct.to(DEVICE, non_blocking=True)
            smri = smri.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                output = model(ct, smri)
                
            loss = criterion(output.float(), mask.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Avg Epoch Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Success! Model saved at: {SAVE_PATH}")