import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')

from train_gan import ProposedLateFusionUNet, scan_triplets

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ROOT_DIR = 'content/HaN-Seg_2D_Split'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_model_and_visualize():
    model = ProposedLateFusionUNet().to(DEVICE)
    model.load_state_dict(torch.load('late_fusion.pth', map_location=DEVICE))
    model.eval()

    all_triplets = scan_triplets(ROOT_DIR)
    
    if len(all_triplets) == 0:
        print("Error: No valid data found.")
        return

    sample = random.choice(all_triplets)
    ct_path, smri_path, mask_path = sample

    ct_img = cv2.imread(ct_path, 0)
    smri_img = cv2.imread(smri_path, 0)
    mask_img = cv2.imread(mask_path, 0)

    ct_resized = cv2.resize(ct_img, (256, 256))
    smri_resized = cv2.resize(smri_img, (256, 256))
    mask_resized = cv2.resize(mask_img, (256, 256))

    ct_tensor = torch.tensor(ct_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    smri_tensor = torch.tensor(smri_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    ct_tensor = ct_tensor.to(DEVICE)
    smri_tensor = smri_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(ct_tensor, smri_tensor)
        pred_mask = (output > 0.5).float().squeeze().cpu().numpy()

    mask_binary = (mask_resized > 127).astype(np.float32)

    plt.figure(figsize=(16, 4), dpi=300)

    plt.subplot(1, 4, 1)
    plt.imshow(ct_resized, cmap='gray')
    plt.title('CT Input', fontsize=14, pad=10)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(smri_resized, cmap='gray')
    plt.title('sMRI Input', fontsize=14, pad=10)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask_binary, cmap='gray')
    plt.title('Ground Truth Mask', fontsize=14, pad=10)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask', fontsize=14, pad=10)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('test_visualization.png', bbox_inches='tight')
    print("Test complete. Visualization saved as test_visualization.png")
    print(f"Tested on: {ct_path}")

if __name__ == "__main__":
    test_model_and_visualize()