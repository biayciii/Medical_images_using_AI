import os
import numpy as np
import cv2

root = 'content/HaN-Seg_2D_Split'
ct_dir = os.path.join(root, 'CT')
mask_dir = os.path.join(root, 'Mask')
out_mask_dir = os.path.join(root, 'Mask_2D')

if not os.path.exists(out_mask_dir):
    os.makedirs(out_mask_dir)

ct_cases = [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d))]

for case in ct_cases:
    case_ct_dir = os.path.join(ct_dir, case)
    case_mask_dir = os.path.join(mask_dir, case)
    case_out_dir = os.path.join(out_mask_dir, case)
    
    if not os.path.exists(case_out_dir):
        os.makedirs(case_out_dir)
        
    if not os.path.exists(case_mask_dir):
        continue
        
    ct_slices = [f for f in os.listdir(case_ct_dir) if f.lower().endswith(('.png', '.jpg'))]
    organ_folders = [d for d in os.listdir(case_mask_dir) if os.path.isdir(os.path.join(case_mask_dir, d))]
    
    for slice_name in ct_slices:
        ct_path = os.path.join(case_ct_dir, slice_name)
        ct_img = cv2.imread(ct_path, 0)
        
        if ct_img is None:
            continue
            
        combined_mask = np.zeros_like(ct_img)
        
        for organ in organ_folders:
            organ_slice_path = os.path.join(case_mask_dir, organ, slice_name)
            if os.path.exists(organ_slice_path):
                organ_img = cv2.imread(organ_slice_path, 0)
                if organ_img is not None:
                    combined_mask = np.maximum(combined_mask, organ_img)
        
        
        if np.max(combined_mask) > 0:
            combined_mask[combined_mask > 0] = 255
            
        out_path = os.path.join(case_out_dir, slice_name)
        cv2.imwrite(out_path, combined_mask)

print(f"Process complete. Masks saved to: {out_mask_dir}")