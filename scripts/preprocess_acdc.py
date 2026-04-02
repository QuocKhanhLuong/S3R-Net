"""
Preprocess ACDC dataset: Cardiac MRI → .npy volumes
Classes: 0=BG, 1=RV, 2=MYO, 3=LV

Usage:
    python scripts/preprocess_acdc.py --input data/ACDC/training --output preprocessed_data/ACDC/training
"""

import os
import argparse
import configparser
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
from skimage.transform import resize


def normalize_zscore(image):
    """Z-score normalization with outlier clipping."""
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    image = np.clip(image, p05, p995)
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / std if std > 0 else image - mean


def preprocess_patient(patient_path, target_size=(224, 224)):
    """Process one ACDC patient: load ED and ES frames with spacing info."""
    patient_folder = os.path.basename(patient_path)
    info_cfg_path = os.path.join(patient_path, 'Info.cfg')
    
    if not os.path.exists(info_cfg_path):
        return []
    
    try:
        parser = configparser.ConfigParser()
        with open(info_cfg_path, 'r') as f:
            parser.read_string('[DEFAULT]\n' + f.read())
        ed_frame = int(parser['DEFAULT']['ED'])
        es_frame = int(parser['DEFAULT']['ES'])
    except Exception as e:
        print(f"  Error: {patient_folder}: {e}")
        return []
    
    results = []
    
    for frame_num, frame_name in [(ed_frame, 'ED'), (es_frame, 'ES')]:
        img_filename = f'{patient_folder}_frame{frame_num:02d}.nii.gz'
        mask_filename = f'{patient_folder}_frame{frame_num:02d}_gt.nii.gz'
        
        img_path = os.path.join(patient_path, img_filename)
        mask_path = os.path.join(patient_path, mask_filename)
        
        if not os.path.exists(img_path):
            img_path = img_path.replace('.nii.gz', '.nii')
            mask_path = mask_path.replace('.nii.gz', '.nii')
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        
        try:
            img_nii = nib.load(img_path)
            img_data = normalize_zscore(img_nii.get_fdata())
            mask_data = nib.load(mask_path).get_fdata()
            
            orig_shape = img_data.shape
            orig_spacing = img_nii.header.get_zooms()  # (sx, sy, sz) in mm
            num_slices = img_data.shape[2]
            
            eff_spacing_y = float(orig_spacing[0]) * orig_shape[0] / target_size[0]
            eff_spacing_x = float(orig_spacing[1]) * orig_shape[1] / target_size[1]
            eff_spacing_z = float(orig_spacing[2])
            
            resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
            resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
            
            for i in range(num_slices):
                resized_img[:, :, i] = resize(img_data[:, :, i], target_size, order=1, preserve_range=True, anti_aliasing=True, mode='reflect')
                resized_mask[:, :, i] = resize(mask_data[:, :, i], target_size, order=0, preserve_range=True, anti_aliasing=False, mode='reflect').astype(np.uint8)
            
            spacing_info = {
                'orig_shape': [int(s) for s in orig_shape],
                'orig_spacing': [float(s) for s in orig_spacing],
                'effective_spacing': [eff_spacing_z, eff_spacing_y, eff_spacing_x],
            }
            
            results.append((resized_img, resized_mask, f"{patient_folder}_{frame_name}", spacing_info))
        except Exception as e:
            print(f"  Error: {patient_folder} frame {frame_num}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Preprocess ACDC dataset')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--no-skip', action='store_true')
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    os.makedirs(args.output, exist_ok=True)
    volumes_dir = os.path.join(args.output, 'volumes')
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    patient_folders = sorted([
        os.path.join(args.input, d) for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d)) and d.startswith('patient')
    ])
    
    print(f"ACDC Preprocessing: {len(patient_folders)} patients")
    
    volume_info = {}
    processed, skipped = 0, 0
    
    for patient_path in tqdm(patient_folders, desc="ACDC"):
        for volume, mask, volume_id, spacing_info in preprocess_patient(patient_path, target_size):
            vol_path = os.path.join(volumes_dir, f'{volume_id}.npy')
            mask_path = os.path.join(masks_dir, f'{volume_id}.npy')
            
            if not args.no_skip and os.path.exists(vol_path):
                skipped += 1
                volume_info[volume_id] = {
                    'num_slices': int(mask.shape[2]),
                    **spacing_info,
                }
                continue
            
            np.save(vol_path, volume)
            np.save(mask_path, mask)
            volume_info[volume_id] = {
                'num_slices': int(mask.shape[2]),
                **spacing_info,
            }
            processed += 1
    
    metadata = {
        'dataset': 'ACDC', 'num_classes': 4,
        'class_names': ['Background', 'RV', 'MYO', 'LV'],
        'target_size': list(target_size),
        'total_volumes': len(volume_info), 'volume_info': volume_info
    }
    with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
