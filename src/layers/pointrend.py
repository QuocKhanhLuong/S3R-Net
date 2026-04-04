"""
[DEPRECATED] PointRend Module for Fine-grained Boundary Refinement
Reference: https://arxiv.org/abs/1912.08193

This module is deprecated. SpecMambaNet does not use PointRend.
Kept for backward-compatible checkpoint loading only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointRend(nn.Module):
    """
    PointRend: Post-processing module to refine segmentation boundaries.
    
    Instead of upsampling the entire mask, it:
    1. Predicts a coarse mask
    2. Selects N most uncertain points (typically at boundaries)
    3. Samples features at those points from high-res feature maps
    4. Classifies each point with a small MLP
    """
    
    def __init__(self, in_channels, num_classes, num_points=2048, oversample=3.0, 
                 importance_sample_ratio=0.75, hidden_dim=256):
        super().__init__()
        self.num_points = num_points
        self.oversample = oversample
        self.importance_sample_ratio = importance_sample_ratio
        self.num_classes = num_classes
        
        # Point-wise MLP to classify each sampled point
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + num_classes, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, num_classes, 1)
        )
        
    def get_uncertain_points(self, coarse_logits, num_points):
        """
        Sample points with highest uncertainty (entropy or close to 0.5 probability).
        During training: use random + importance sampling
        During inference: use uniform grid or top-k uncertain
        """
        B, C, H, W = coarse_logits.shape
        
        # Compute uncertainty as negative of max probability
        probs = torch.softmax(coarse_logits, dim=1)
        uncertainty = -probs.max(dim=1)[0]  # [B, H, W]
        
        # Flatten and get top-k uncertain points
        uncertainty_flat = uncertainty.view(B, -1)  # [B, H*W]
        
        if self.training:
            # Importance sampling: sample more from uncertain regions
            num_uncertain = int(num_points * self.importance_sample_ratio)
            num_random = num_points - num_uncertain
            
            # Top-k uncertain points
            _, idx_uncertain = torch.topk(uncertainty_flat, num_uncertain, dim=1)
            
            # Random points
            idx_random = torch.randint(0, H * W, (B, num_random), device=coarse_logits.device)
            
            point_indices = torch.cat([idx_uncertain, idx_random], dim=1)
        else:
            # Inference: use top-k uncertain
            _, point_indices = torch.topk(uncertainty_flat, num_points, dim=1)
        
        # Convert to normalized coordinates [-1, 1]
        y_coords = (point_indices // W).float() / (H - 1) * 2 - 1
        x_coords = (point_indices % W).float() / (W - 1) * 2 - 1
        point_coords = torch.stack([x_coords, y_coords], dim=-1)  # [B, N, 2]
        
        return point_coords, point_indices
    
    def sample_features(self, features, point_coords):
        """Sample features at given point coordinates using bilinear interpolation."""
        # point_coords: [B, N, 2] in range [-1, 1]
        # features: [B, C, H, W]
        
        point_coords = point_coords.unsqueeze(2)  # [B, N, 1, 2]
        sampled = F.grid_sample(features, point_coords, align_corners=True)  # [B, C, N, 1]
        return sampled.squeeze(-1)  # [B, C, N]
    
    def forward(self, coarse_logits, fine_features, target_size=None):
        """
        Args:
            coarse_logits: [B, num_classes, H_coarse, W_coarse] - initial prediction
            fine_features: [B, C, H_fine, W_fine] - high-resolution features
            target_size: (H, W) - final output size
            
        Returns:
            refined_logits: [B, num_classes, H, W]
        """
        if target_size is None:
            target_size = fine_features.shape[2:]
        
        B = coarse_logits.shape[0]
        
        # Upsample coarse logits to fine resolution
        coarse_up = F.interpolate(coarse_logits, size=target_size, mode='bilinear', align_corners=True)
        
        if not self.training and self.num_points == 0:
            return coarse_up
        
        # Get uncertain points
        num_points = self.num_points if self.training else min(self.num_points, target_size[0] * target_size[1] // 4)
        point_coords, point_indices = self.get_uncertain_points(coarse_up, num_points)
        
        # Sample features at uncertain points
        fine_point_features = self.sample_features(fine_features, point_coords)  # [B, C, N]
        coarse_point_logits = self.sample_features(coarse_up, point_coords)  # [B, num_classes, N]
        
        # Concatenate and predict
        point_features = torch.cat([fine_point_features, coarse_point_logits], dim=1)  # [B, C+num_classes, N]
        point_logits = self.mlp(point_features)  # [B, num_classes, N]
        
        # Scatter refined predictions back
        refined_logits = coarse_up.clone()
        B, C, H, W = refined_logits.shape
        
        # Ensure dtype match for AMP compatibility
        point_logits = point_logits.to(refined_logits.dtype)
        
        for b in range(B):
            for c in range(C):
                refined_logits[b, c].view(-1).scatter_(
                    0, point_indices[b], point_logits[b, c]
                )
        
        return refined_logits


class PointRendHead(nn.Module):
    """Wrapper that combines coarse head with PointRend refinement."""
    
    def __init__(self, in_channels, num_classes, num_points=2048, hidden_dim=256):
        super().__init__()
        self.coarse_head = nn.Conv2d(in_channels, num_classes, 1)
        self.pointrend = PointRend(
            in_channels=in_channels, 
            num_classes=num_classes,
            num_points=num_points,
            hidden_dim=hidden_dim
        )
        
    def forward(self, features, fine_features=None, target_size=None):
        """
        Args:
            features: Main features for coarse prediction
            fine_features: High-res features for refinement (optional, uses features if None)
            target_size: Final output size
        """
        coarse_logits = self.coarse_head(features)
        
        if fine_features is None:
            fine_features = features
            
        refined_logits = self.pointrend(coarse_logits, fine_features, target_size)
        
        return refined_logits
