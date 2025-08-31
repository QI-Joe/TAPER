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
        Compute random Fourier features φ_time(t) for given timestamps
        
        Args:
            timestamps: tensor of shape (n,) containing timestamps
            
        Returns:
            time_features: tensor of shape (n, 2*rff_dim) containing RFF features
        """
        timestamps = timestamps.to(self.device).float()
        
        # Compute ω * t for all frequencies and timestamps
        omega_t = torch.outer(timestamps, self.omega)  # (n, rff_dim)
        
        # Compute cos and sin features
        cos_features = torch.cos(omega_t)
        sin_features = torch.sin(omega_t)
        
        # Concatenate and normalize
        time_features = torch.cat([cos_features, sin_features], dim=1)
        time_features = time_features * math.sqrt(2.0 / self.rff_dim)
        
        return time_features
    
    def compute_kernel_matrix(self, timestamps1, timestamps2=None):
        """
        Compute temporal kernel matrix K_time(t_i, t_j)
        
        Args:
            timestamps1: tensor of shape (n1,)
            timestamps2: tensor of shape (n2,) or None (defaults to timestamps1)
            
        Returns:
            kernel_matrix: tensor of shape (n1, n2)
        """
        if timestamps2 is None:
            timestamps2 = timestamps1
            
        phi_t1 = self.compute_time_features(timestamps1)
        phi_t2 = self.compute_time_features(timestamps2)
        
        return torch.mm(phi_t1, phi_t2.t())
    
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
        timestamps = timestamps.to(self.device).float()
        
        # Use learnable sigma
        sigma = torch.exp(self.log_sigma)
        omega_t = torch.outer(timestamps, self.omega * sigma)
        
        cos_features = torch.cos(omega_t)
        sin_features = torch.sin(omega_t)
        
        time_features = torch.cat([cos_features, sin_features], dim=1)
        time_features = time_features * math.sqrt(2.0 / self.rff_dim)
        
        return time_features
