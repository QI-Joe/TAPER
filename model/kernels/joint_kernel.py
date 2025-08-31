"""
Joint Kernel Module combining time and graph kernels for TDSL-TPPR
"""
import torch
import torch.nn as nn
import numpy as np
from .time_kernel import BochnerTimeKernel
from .graph_kernel import TDSLGraphKernel


class TDSLTPPRJointKernel(nn.Module):
    """
    Joint kernel implementing factorized TDSL × TPPR kernel
    K_joint((v,t),(v',t')) = K_time(t,t') × K_graph(v,v')
    """
    
    def __init__(self, time_rff_dim=16, time_sigma=1.0, graph_mu=1.0, 
                 max_nodes=10000, device='cpu'):
        super(TDSLTPPRJointKernel, self).__init__()
        
        self.time_kernel = BochnerTimeKernel(
            rff_dim=time_rff_dim, sigma=time_sigma, device=device
        )
        self.graph_kernel = TDSLGraphKernel(
            mu=graph_mu, max_nodes=max_nodes, device=device
        )
        self.device = device
        
    def compute_joint_features(self, timestamps, edge_index, edge_weight, num_nodes,
                             source_vector=None, temporal_decay_factors=None):
        """
        Compute joint features ψ(v,t) = φ_time(t) ⊗ φ_graph,t(v)
        
        Args:
            timestamps: tensor of shape (n,) 
            edge_index: tensor of shape (2, num_edges)
            edge_weight: tensor of shape (num_edges,)
            num_nodes: number of nodes
            source_vector: optional source signal
            temporal_decay_factors: optional temporal decay
            
        Returns:
            joint_features: tensor of shape (n, time_dim * graph_dim)
        """
        # Compute time features φ_time(t)
        time_features = self.time_kernel.compute_time_features(timestamps)  # (n, time_dim)
        
        # Compute graph features φ_graph,t(v)  
        graph_features = self.graph_kernel.compute_graph_features(
            edge_index, edge_weight, num_nodes, source_vector, temporal_decay_factors
        )  # (num_nodes, graph_dim)
        
        # Select relevant node features for the timestamps
        # Assuming timestamps correspond to node indices (may need adjustment)
        n_samples = len(timestamps)
        if n_samples <= num_nodes:
            selected_graph_features = graph_features[:n_samples]  # (n, graph_dim)
        else:
            # If more timestamps than nodes, repeat graph features
            selected_graph_features = graph_features[timestamps.long() % num_nodes]
        
        # Compute tensor product: ψ(v,t) = φ_time(t) ⊗ φ_graph,t(v)
        joint_features = torch.kron(time_features, selected_graph_features)
        
        return joint_features
    
    def compute_joint_kernel_matrix(self, timestamps1, edge_index1, edge_weight1, num_nodes1,
                                   timestamps2=None, edge_index2=None, edge_weight2=None, 
                                   num_nodes2=None, source_vector1=None, source_vector2=None):
        """
        Compute joint kernel matrix K_joint((v_i,t_i),(v_j,t_j))
        """
        # Handle default case
        if timestamps2 is None:
            timestamps2 = timestamps1
            edge_index2 = edge_index1  
            edge_weight2 = edge_weight1
            num_nodes2 = num_nodes1
            source_vector2 = source_vector1
        
        # Compute time kernel matrix
        time_kernel_matrix = self.time_kernel.compute_kernel_matrix(timestamps1, timestamps2)
        
        # Compute graph features for both snapshots
        graph_features1 = self.graph_kernel.compute_graph_features(
            edge_index1, edge_weight1, num_nodes1, source_vector1
        )
        graph_features2 = self.graph_kernel.compute_graph_features(
            edge_index2, edge_weight2, num_nodes2, source_vector2
        )
        
        # Compute graph kernel matrix
        graph_kernel_matrix = self.graph_kernel.compute_kernel_matrix(graph_features1, graph_features2)
        
        # Joint kernel: element-wise product (Hadamard product)
        joint_kernel_matrix = time_kernel_matrix * graph_kernel_matrix
        
        return joint_kernel_matrix, time_kernel_matrix, graph_kernel_matrix
    
    def forward(self, timestamps, edge_index, edge_weight, num_nodes, 
                mode='features', **kwargs):
        """
        Forward pass with different modes
        
        Args:
            mode: 'features' returns joint features, 'kernel' returns kernel matrix
        """
        if mode == 'features':
            return self.compute_joint_features(
                timestamps, edge_index, edge_weight, num_nodes, **kwargs
            )
        elif mode == 'kernel':
            return self.compute_joint_kernel_matrix(
                timestamps, edge_index, edge_weight, num_nodes, **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


class LinearJointKernelHead(nn.Module):
    """
    Linear classification/regression head using joint kernel features
    """
    
    def __init__(self, joint_kernel, feature_dim, num_classes, dropout=0.1):
        super(LinearJointKernelHead, self).__init__()
        
        self.joint_kernel = joint_kernel
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, timestamps, edge_index, edge_weight, num_nodes, **kwargs):
        # Get joint features
        joint_features = self.joint_kernel(
            timestamps, edge_index, edge_weight, num_nodes, mode='features', **kwargs
        )
        
        # Apply classifier
        logits = self.classifier(joint_features)
        return logits


class HybridEmbeddingKernelModel(nn.Module):
    """
    Hybrid model combining learned embeddings with kernel features
    """
    
    def __init__(self, embedding_dim, joint_kernel, kernel_weight=0.5, dropout=0.1):
        super(HybridEmbeddingKernelModel, self).__init__()
        
        self.joint_kernel = joint_kernel
        self.kernel_weight = kernel_weight
        self.embedding_weight = 1.0 - kernel_weight
        
        # Fusion layers
        self.kernel_proj = nn.Linear(joint_kernel.time_kernel.feature_dim * embedding_dim, embedding_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
    def forward(self, learned_embeddings, timestamps, edge_index, edge_weight, num_nodes, **kwargs):
        """
        Fuse learned embeddings with kernel features
        
        Args:
            learned_embeddings: tensor from TGN/GNN model
            timestamps, edge_index, etc.: kernel computation inputs
            
        Returns:
            fused_embeddings: combined representation
        """
        # Get kernel features
        kernel_features = self.joint_kernel(
            timestamps, edge_index, edge_weight, num_nodes, mode='features', **kwargs
        )
        
        # Project kernel features to embedding dimension
        kernel_projected = self.kernel_proj(kernel_features)
        
        # Weighted fusion
        fused = (self.embedding_weight * learned_embeddings + 
                self.kernel_weight * kernel_projected)
        
        # Apply fusion layer
        fused_embeddings = self.fusion(fused)
        
        return fused_embeddings
