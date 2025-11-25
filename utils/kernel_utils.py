"""
Utility functions for kernel fusion in TDSL-TPPR implementation
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
import numpy as np
import warnings
from model.kernels.joint_kernel import TDSLTPPRJointKernel


class KernelFusion(TDSLTPPRJointKernel):
    def __init__(self, fuse_mode, combine_mode,time_rff_dim=16, time_sigma=1.0, device='cpu'):
        """
        Initialize KernelFusion with proper parameter handling
        
        Args:
            fuse_mode: fusion strategy ("product", "cosine", "harmonic", etc.)
            time_rff_dim: dimension for random Fourier features
            time_sigma: sigma parameter for time kernel
            device: computation device
            graph_mu: graph diffusion parameter (unused in current implementation)
            max_nodes: maximum number of nodes (unused in current implementation)
            **kwargs: additional keyword arguments (ignored)
        """
        # 只传递父类实际需要的参数
        super(KernelFusion, self).__init__(
            time_rff_dim=time_rff_dim, 
            time_sigma=time_sigma, 
            device=device
        )
        self.fuse_mode = fuse_mode
        self.combine_mode=combine_mode
        

    def kernel_fuse(self, tppr_weights: Tensor, tdsl_weights: Tensor, timestamp, ablation: str, eps=1e-8):
        """
        Fuse TPPR and TDSL weights according to different strategies
        
        Args:
            tppr_weights: tensor of TPPR-derived weights
            tdsl_weights: tensor of TDSL-derived weights  
            mode: fusion strategy - "product", "sum", "cosine", or "learned"
            eps: small constant for numerical stability
            
        Returns:
            fused_weights: normalized fusion result
        """
        if self.fuse_mode == "product":
            # Theoretical joint kernel: w_ij = π^TPPR_ij × h^TDSL_ij
            fused = tppr_weights * tdsl_weights
            
        elif self.fuse_mode == "cosine":
            # Legacy cosine fusion (for ablation comparison)
            fused = (torch.cos(tppr_weights) * torch.cos(tdsl_weights) + 
                    torch.sin(tppr_weights) * torch.sin(tdsl_weights))
            
        elif self.fuse_mode == "harmonic":
            # Harmonic mean (for positive weights)
            fused = 2 * (tppr_weights * tdsl_weights) / (tppr_weights + tdsl_weights + eps)
            
        elif self.fuse_mode == "geometric":
            # Geometric mean 
            fused = torch.sqrt(tppr_weights * tdsl_weights + eps)
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fuse_mode}")
        
        if ablation=="tdsl":
            fused = tdsl_weights
        elif ablation=="tppr":
            fused=tppr_weights
        
        if self.combine_mode == "full":
            fused = self.compute_joint_features(timestamp, fused)
        
        return torch.relu(fused)



class LearnedKernelFusion(nn.Module):
    """
    Learnable fusion of TPPR and TDSL weights
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super(LearnedKernelFusion, self).__init__()
        
        self.fusion_net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Softmax(dim=-1)  # Ensure output is normalized
        )
        
    def forward(self, tppr_weights, tdsl_weights):
        """
        Learn to fuse TPPR and TDSL weights
        """
        # Concatenate inputs
        combined = torch.cat([tppr_weights, tdsl_weights], dim=-1)
        
        # Apply fusion network
        fused_weights = self.fusion_net(combined)
        
        return fused_weights


def validate_kernel_inputs(tppr_weights, tdsl_weights):
    """
    Validate input tensors for kernel fusion
    """
    assert tppr_weights.shape == tdsl_weights.shape, \
        f"Shape mismatch: TPPR {tppr_weights.shape} vs TDSL {tdsl_weights.shape}"
    
    assert torch.all(tppr_weights >= 0), "TPPR weights must be non-negative"
    assert torch.all(tdsl_weights >= 0), "TDSL weights must be non-negative"
    
    # Check for NaN or Inf
    assert torch.isfinite(tppr_weights).all(), "TPPR weights contain NaN/Inf"
    assert torch.isfinite(tdsl_weights).all(), "TDSL weights contain NaN/Inf"
    
    return True
