"""
Time Kernel Module implementing Bochner's theorem for temporal alignment
"""
import torch
import torch.nn as nn
import numpy as np
import math


class BochnerTimeKernel(nn.Module):
    """
    Bochner time kernel using random Fourier features
    K_time(t,t') ≈ φ_time(t)^T φ_time(t') 
    where φ_time(t) = sqrt(2/k) * [cos(ω_r * t), sin(ω_r * t)]_r
    """
    
    def __init__(self, rff_dim=16, sigma=1.0, device='cpu'):
        super(BochnerTimeKernel, self).__init__()
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.device = device
        
        # Pre-sample random frequencies ω ~ N(0, σ²)
        self.register_buffer('omega', torch.randn(rff_dim) * sigma)
        self.feature_dim = 2 * rff_dim  # cos + sin features
        
    def compute_time_features(self, timestamps):
        """
        Enhanced to compute random Fourier features φ_time(t) for arbitrary input shapes
        
        Args:
            timestamps: tensor of any shape (...,) containing timestamps
                       e.g., (n,) for traditional usage or (n, k) for neighbor timestamps
            
        Returns:
            time_features: tensor of shape (..., 2*rff_dim) containing RFF features
        """
        # Store original shape for reshaping output
        original_shape = timestamps.shape
        timestamps = timestamps.to(self.omega.device).float()
        
        # Flatten timestamps for batch RFF computation
        t_flat = timestamps.reshape(-1)  # [total_elements]
        
        # Compute ω * t for all frequencies and timestamps
        omega_t = torch.outer(t_flat, self.omega)  # (total_elements, rff_dim)
        
        # Compute cos and sin features
        cos_features = torch.cos(omega_t)
        sin_features = torch.sin(omega_t)
        
        # Concatenate and normalize
        time_features = torch.cat([cos_features, sin_features], dim=1)
        time_features = time_features * math.sqrt(2.0 / self.rff_dim)
        
        # Reshape back to original shape + feature dimension
        return time_features.reshape(*original_shape, -1)
    
    def compute_neighbor_time_features(self, timestamp_matrix, normalize_per_sample=True):
        """
        Compute time features for (n, k) neighbor timestamp matrix
        
        Args:
            timestamp_matrix: tensor of shape (n, k) where n=samples, k=neighbors
            normalize_per_sample: if True, apply softmax normalization per sample
            
        Returns:
            Phi: tensor of shape (n, k, 2*rff_dim) - raw time features for each neighbor
        """
        n, k = timestamp_matrix.shape
        
        # Compute RFF features for all timestamps
        Phi = self.compute_time_features(timestamp_matrix)  # [n, k, 2*rff_dim]
        
        if normalize_per_sample:
            # Apply softmax normalization across neighbors (dim=1) for each feature
            # This helps with numerical stability and emphasizes relative differences
            Phi = torch.softmax(Phi, dim=1)
        
        return Phi
    
    def forward(self, timestamps):
        """Forward pass returns time features"""
        return self.compute_time_features(timestamps)


class AdaptiveTimeKernel(BochnerTimeKernel):
    """
    Adaptive version with learnable bandwidth parameter
    """
    
    def __init__(self, rff_dim=16, initial_sigma=1.0, device='cpu'):
        super().__init__(rff_dim, initial_sigma, device)
        
        # Make sigma learnable
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(initial_sigma)))
        
    def compute_time_features(self, timestamps):
        """Enhanced adaptive version with learnable sigma"""
        # Store original shape for reshaping output
        original_shape = timestamps.shape
        timestamps = timestamps.to(self.omega.device).float()
        
        # Use learnable sigma
        sigma = torch.exp(self.log_sigma)
        
        # Flatten timestamps for batch computation
        t_flat = timestamps.reshape(-1)
        omega_t = torch.outer(t_flat, self.omega * sigma)
        
        cos_features = torch.cos(omega_t)
        sin_features = torch.sin(omega_t)
        
        time_features = torch.cat([cos_features, sin_features], dim=1)
        time_features = time_features * math.sqrt(2.0 / self.rff_dim)
        
        # Reshape back to original shape + feature dimension
        return time_features.reshape(*original_shape, -1)
