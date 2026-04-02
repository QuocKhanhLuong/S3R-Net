"""
ACDC Dataset with Data Augmentation support
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import json

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Augmentation disabled.")


def get_train_augmentations():
    """Get training augmentations for medical image segmentation.
    
    Input is assumed to be Z-score normalized (mean~0, std~1) from preprocessing.
    Only spatial and intensity augmentations are applied; no re-normalization
    to avoid train/val distribution mismatch.
    """
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), p=0.5),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_val_augmentations():
    """No-op for validation: data is already Z-score normalized."""
    return None


class ACDCDataset2D(Dataset):
    """ACDC 2D Dataset without augmentation (original)."""
    
    def __init__(self, npy_dir, use_memmap=True, in_channels=3, max_cache=10):
        self.use_memmap = use_memmap
        self.in_channels = in_channels
        self.max_cache = max_cache
        self._cache = OrderedDict()
        
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        self.vol_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            volume_info = meta.get('volume_info', {})
        else:
            volume_info = None
        
        self.index_map = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                vol = np.load(vp, mmap_mode='r')
                n_slices = vol.shape[2]
            for s in range(n_slices):
                self.index_map.append((i, s))
        
        print(f"ACDCDataset2D: {len(self.index_map)} slices from {len(self.vol_paths)} volumes")
    
    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        
        mode = 'r' if self.use_memmap else None
        vol = np.load(self.vol_paths[idx], mmap_mode=mode)
        mask = np.load(self.mask_paths[idx], mmap_mode=mode)
        self._cache[idx] = (vol, mask)
        
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return vol, mask
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        
        img = vol[:, :, slice_idx].copy().astype(np.float32)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        
        img = torch.from_numpy(img).unsqueeze(0)
        if self.in_channels == 3:
            img = img.repeat(3, 1, 1)
        
        return img, torch.from_numpy(gt)


class ACDCDataset2DAugmented(Dataset):
    """ACDC 2D Dataset with albumentations augmentation support."""
    
    def __init__(self, npy_dir, use_memmap=True, in_channels=3, max_cache=10, 
                 augment=False, transform=None):
        self.use_memmap = use_memmap
        self.in_channels = in_channels
        self.max_cache = max_cache
        self._cache = OrderedDict()
        self.augment = augment
        
        # Set transform
        if transform is not None:
            self.transform = transform
        elif augment and HAS_ALBUMENTATIONS:
            self.transform = get_train_augmentations()
        else:
            self.transform = get_val_augmentations() if HAS_ALBUMENTATIONS else None
        
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        self.vol_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            volume_info = meta.get('volume_info', {})
        else:
            volume_info = None
        
        self.index_map = []
        for i, vp in enumerate(self.vol_paths):
            vid = os.path.basename(vp).replace('.npy', '')
            if volume_info and vid in volume_info:
                n_slices = volume_info[vid]['num_slices']
            else:
                vol = np.load(vp, mmap_mode='r')
                n_slices = vol.shape[2]
            for s in range(n_slices):
                self.index_map.append((i, s))
        
        aug_str = "with augmentation" if augment else "no augmentation"
        print(f"ACDCDataset2DAugmented: {len(self.index_map)} slices from {len(self.vol_paths)} volumes ({aug_str})")
    
    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        
        mode = 'r' if self.use_memmap else None
        vol = np.load(self.vol_paths[idx], mmap_mode=mode)
        mask = np.load(self.mask_paths[idx], mmap_mode=mode)
        self._cache[idx] = (vol, mask)
        
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return vol, mask
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.index_map[idx]
        vol, mask = self._load(vol_idx)
        
        img = vol[:, :, slice_idx].copy().astype(np.float32)
        gt = mask[:, :, slice_idx].copy().astype(np.int64)
        
        if self.transform is not None:
            transformed = self.transform(image=img, mask=gt)
            img = transformed['image']
            gt = transformed['mask']
            
            img = torch.from_numpy(img).unsqueeze(0).float()
            gt = torch.from_numpy(gt).long()
        else:
            img = torch.from_numpy(img).unsqueeze(0)
            gt = torch.from_numpy(gt)
        
        if self.in_channels == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        return img, gt
