"""
Graph Diffusion Kernel Module implementing TDSL (Temporal Decay Spectral Laplacian)
"""
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor
import torch.nn.functional as F


class TDSLGraphKernel(nn.Module):
    """
    TDSL Graph kernel using resolvent filter on temporally decayed Laplacian
    φ_graph,t(v) = (I + μL_sym,t)^(-1) s_t
    K_graph(v,v') = φ_graph,t(v)^T φ_graph,t'(v')
    """
    
    def __init__(self, mu=1.0, max_nodes=10000, device='cpu'):
        super(TDSLGraphKernel, self).__init__()
        self.mu = mu
        self.max_nodes = max_nodes
        self.device = device
        
        # Cache for storing computed diffusion features
        self.cached_features = None
        self.cached_snapshot_id = None
        
    def compute_resolvent_filter(self, edge_index, edge_weight, num_nodes, temporal_decay_factors=None):
        """
        Compute resolvent filter (I + μL_sym)^(-1) for TDSL
        
        Args:
            edge_index: tensor of shape (2, num_edges)
            edge_weight: tensor of shape (num_edges,) 
            num_nodes: number of nodes
            temporal_decay_factors: optional decay factors for edges
            
        Returns:
            resolvent_matrix: sparse tensor representing (I + μL_sym)^(-1)
        """
        # Apply temporal decay if provided
        if temporal_decay_factors is not None:
            edge_weight = edge_weight * temporal_decay_factors
        
        # Compute normalized Laplacian L_sym = I - D^(-1/2) A D^(-1/2)
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=num_nodes
        )
        
        # Convert to dense for inversion (for now, can optimize later)
        lap_dense = torch.sparse_coo_tensor(
            edge_index_lap, edge_weight_lap, (num_nodes, num_nodes)
        ).to_dense()
        
        # Compute I + μL_sym
        identity = torch.eye(num_nodes, device=self.device)
        resolvent_dense = identity + self.mu * lap_dense
        
        # Invert: (I + μL_sym)^(-1)
        try:
            inv_resolvent = torch.inverse(resolvent_dense)
        except:
            # Use pseudo-inverse if singular
            inv_resolvent = torch.pinverse(resolvent_dense)
            
        return inv_resolvent
    
    def compute_chebyshev_approximation(self, edge_index, edge_weight, num_nodes, 
                                      degree=10, temporal_decay_factors=None):
        """
        Approximate resolvent using Chebyshev polynomials for large graphs
        """
        # Apply temporal decay
        if temporal_decay_factors is not None:
            edge_weight = edge_weight * temporal_decay_factors
            
        # Get normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=num_nodes
        )
        
        # Scale spectrum to [-1, 1] for Chebyshev stability
        # Estimate λ_max (for now use simple heuristic)
        lambda_max = 2.0  # Conservative estimate for normalized Laplacian
        
        # Scale L: L_scaled = (2/λ_max) * L - I
        scaled_edge_weight = (2.0 / lambda_max) * edge_weight_lap
        
        # Add -I to diagonal
        identity_indices = torch.stack([torch.arange(num_nodes)] * 2).to(self.device)
        identity_values = -torch.ones(num_nodes).to(self.device)
        
        full_edge_index = torch.cat([edge_index_lap, identity_indices], dim=1)
        full_edge_weight = torch.cat([scaled_edge_weight, identity_values])
        
        # Implement Chebyshev recursion for (I + μL)^(-1)
        # For simplicity, return identity for now (can be extended)
        return torch.eye(num_nodes, device=self.device)
    
    def compute_graph_features(self, edge_index, edge_weight, num_nodes, 
                             source_vector=None, temporal_decay_factors=None,
                             use_chebyshev=False, degree=10):
        """
        Compute graph diffusion features φ_graph,t(v) = h(L_sym,t) s_t
        
        Args:
            edge_index: edge indices
            edge_weight: edge weights  
            num_nodes: number of nodes
            source_vector: source signal s_t (default: identity matrix)
            temporal_decay_factors: decay factors for temporal weighting
            use_chebyshev: whether to use Chebyshev approximation
            
        Returns:
            graph_features: tensor of shape (num_nodes, feature_dim)
        """
        if use_chebyshev and num_nodes > 1000:
            resolvent = self.compute_chebyshev_approximation(
                edge_index, edge_weight, num_nodes, degree, temporal_decay_factors
            )
        else:
            resolvent = self.compute_resolvent_filter(
                edge_index, edge_weight, num_nodes, temporal_decay_factors
            )
        
        # Default source: identity matrix (impulse response)
        if source_vector is None:
            source_vector = torch.eye(num_nodes, device=self.device)
        elif source_vector.dim() == 1:
            # Convert to matrix form
            source_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
            source_matrix[torch.arange(num_nodes), :] = source_vector.unsqueeze(0)
            source_vector = source_matrix
            
        # Apply diffusion: φ = (I + μL)^(-1) @ s
        graph_features = torch.mm(resolvent, source_vector)
        
        return graph_features
    
    def compute_kernel_matrix(self, features1, features2=None):
        """
        Compute graph kernel matrix K_graph(v_i, v_j) = φ_i^T φ_j
        """
        if features2 is None:
            features2 = features1
            
        return torch.mm(features1, features2.t())
    
    def forward(self, edge_index, edge_weight, num_nodes, source_vector=None, 
                temporal_decay_factors=None):
        """Forward pass returns graph features"""
        return self.compute_graph_features(
            edge_index, edge_weight, num_nodes, source_vector, temporal_decay_factors
        )


class MultiScaleTDSLKernel(TDSLGraphKernel):
    """
    Multi-scale TDSL kernel with multiple diffusion strengths
    """
    
    def __init__(self, mu_list=[0.1, 0.5, 1.0], max_nodes=10000, device='cpu'):
        super().__init__(mu=1.0, max_nodes=max_nodes, device=device)
        self.mu_list = mu_list
        self.kernels = nn.ModuleList([
            TDSLGraphKernel(mu=mu, max_nodes=max_nodes, device=device) 
            for mu in mu_list
        ])
        
    def compute_graph_features(self, edge_index, edge_weight, num_nodes, 
                             source_vector=None, temporal_decay_factors=None):
        """
        Compute multi-scale graph features by concatenating different μ values
        """
        features_list = []
        
        for kernel in self.kernels:
            features = kernel.compute_graph_features(
                edge_index, edge_weight, num_nodes, source_vector, temporal_decay_factors
            )
            features_list.append(features)
            
        # Concatenate features from different scales
        multi_scale_features = torch.cat(features_list, dim=1)
        return multi_scale_features
