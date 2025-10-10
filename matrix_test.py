import argparse
import torch
import numpy as np
from pathlib import Path
from model.tgn_model import TGN
from utils.util import get_neighbor_finder
from utils.data_processing import get_data_TPPR, get_Temporal_data_TPPR_Node_Justification, get_data
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import get_laplacian
from scipy.sparse import save_npz, load_npz, csr_matrix

import warnings
from datetime import datetime
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

def str2bool(order: str)->bool:
  if order in ["True", "1"]:
    return True
  return False

parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='cora')
parser.add_argument('--bs', type=int, default=10000, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=7, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--snapshot', type=int, default=4, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.3, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="diffusion", help='Type of embedding module')

parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--save_best',action='store_true', help='store the largest model')
parser.add_argument('--tppr_strategy', type=str, help='[streaming|pruning]', default='streaming')
parser.add_argument('--topk', type=int, default=40, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')
parser.add_argument('--dynamic', type=str2bool, default=False)
parser.add_argument('--task', type=str, default="None", help='robustness task') # edge_disturb
parser.add_argument('--ratio', type=float, default=50, help='imbalance, few shot learning and edge distrubution ratio')
parser.add_argument('--cora_inductive', type=str2bool, default=False, help='whether to use inductive training')
parser.add_argument('--matrix_value', type=str, default="laplacian", choices=["tppr", "laplacian", "ppr"], help='Type of matrix value to compute')


parser.add_argument('--ignore_edge_feats', action='store_true')
parser.add_argument('--ignore_node_feats', action='store_true')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')

# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy streaming --gpu 0 --alpha_list 0.1 --beta_list 0.9

args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = True
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
dynamic: bool = args.dynamic
ROBUST_TASK = args.task
CORA_INDUCTIVE: bool = args.cora_inductive
EPOCH_INTERVAL = 25
RATIO = int(args.ratio) if args.ratio>10 else args.ratio
SNAPSHOT = args.snapshot

round_list, graph_num, graph_feature, edge_num = get_Temporal_data_TPPR_Node_Justification(DATA, snapshot=SNAPSHOT, dynamic=dynamic, task=ROBUST_TASK, ratio=RATIO)
VIEWS = len(round_list)
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
training_strategy = "node"
NODE_DIM = round_list[0][0].node_feat.shape[1]
test_record = []

def tppr_to_sparse_pandas(ptr2tppr_dict, n):
    # collect into lists
    recs = []
    for lvl1 in ptr2tppr_dict:
        for src, val in enumerate(lvl1):
            for (eids, dst, ts), tval in val.items():
                recs.append((int(src), int(dst), float(ts), float(tval)))

    if not recs:
        return sp.csr_matrix((n,n), dtype=np.float64)

    df = pd.DataFrame(recs, columns=['i','j','ts','v'])
    # keep only the row with the max ts for each (i,j)
    idx = df.groupby(['i','j'])['ts'].idxmax()
    df2 = df.loc[idx]

    return sp.csr_matrix((df2.v.values, (df2.i.values, df2.j.values)),
                         shape=(n,n))

def torch_sparse_to_scipy_csr(sparse_tensor):
    """
    Convert a PyTorch sparse COO tensor to SciPy CSR matrix.
    
    Args:
        sparse_tensor (torch.sparse_coo_tensor): PyTorch sparse tensor on CPU
    
    Returns:
        scipy.sparse.csr_matrix: equivalent sparse matrix
    """
    if sparse_tensor.device.type != 'cpu':
        sparse_tensor = sparse_tensor.cpu()
    sparse_tensor = sparse_tensor.coalesce()  # make sure indices are unique and sorted
    
    indices = sparse_tensor.indices().numpy()
    values = sparse_tensor.values().numpy()
    shape = sparse_tensor.size()
    
    # COO matrix in scipy is created with (data, (row, col)) format
    coo = csr_matrix((values, (indices[0], indices[1])), shape=shape)
    
    return coo

def compute_ppr_torch(num_nodes, edge_index, alpha=0.15, num_iters=20, device='cpu'):
    """
    Compute approximate Personalized PageRank matrix P using power iteration in PyTorch.
    
    Args:
        num_nodes (int): number of nodes
        edge_index (LongTensor): shape [2, num_edges], COO format edges
        alpha (float): teleport probability, e.g. 0.15
        num_iters (int): number of power iterations
        device (str): 'cpu' or 'cuda'
    
    Returns:
        ppr_matrix (torch.sparse.FloatTensor): shape (num_nodes, num_nodes)
    """
    # Move edge_index to device
    edge_index = edge_index.to(device)

    # Build adjacency matrix A (sparse)
    values = torch.ones(edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    
    # Compute degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # shape [num_nodes]
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0.0  # avoid inf
    
    # Normalize adjacency to get random walk matrix W = D^{-1} A
    # For sparse: multiply values by deg_inv[src]
    row, col = edge_index
    norm_values = deg_inv[row]
    adj_norm = torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes)).coalesce()

    # Initialize P = I (identity)
    indices = torch.stack([torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)])
    values = torch.ones(num_nodes, device=device)
    P = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    # Power iteration: P = alpha * I + (1 - alpha) * W * P
    for _ in range(num_iters):
        # Sparse matrix multiplication: W @ P
        WP = torch.sparse.mm(adj_norm, P.to_dense())
        P_dense = alpha * torch.eye(num_nodes, device=device) + (1 - alpha) * WP
        # Convert back to sparse for next iteration
        P = P_dense.to_sparse().coalesce()

    return P

if __name__ == "__main__":
    get_data("dblp")

    matrix_test = False
    if matrix_test:
        for i in range(1):
            full_data, train_data, val_data, test_data, train_learn, nn_val_data, nn_test_data, n_nodes, n_edges = round_list[i]
            
            args.n_nodes = graph_num +1
            args.n_edges = edge_num +1
            node_num = graph_num + 1
            
            edge_feats = None
            node_feats = graph_feature
            node_feat_dims = full_data.node_feat.shape[1]
            print(f"Current snapshot full nodes size {full_data.n_unique_nodes}, test_data node size {test_data.n_unique_nodes}, total graph nodes {graph_num}")

            if edge_feats is None or args.ignore_edge_feats: 
                print('>>> Ignore edge features')
                edge_feats = np.zeros((args.n_edges, 1))
                edge_feat_dims = 1

            train_ngh_finder = get_neighbor_finder(train_learn)
            val_tppr_backup, test_tppr_backup = float(0), float(0)
            def get_tppr_value():
                tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_feats, edge_features=edge_feats, device=device,
                            n_layers=NUM_LAYER,n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                            node_dimension = NODE_DIM, time_dimension = TIME_DIM, memory_dimension=NODE_DIM,
                            embedding_module_type=args.embedding_module, 
                            message_function=args.message_function, 
                            aggregator_type=args.aggregator,
                            memory_updater_type=args.memory_updater,
                            n_neighbors=NUM_NEIGHBORS,
                            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                            use_source_embedding_in_message=args.use_source_embedding_in_message,
                            args=args)
            
                train_src = np.concatenate([test_data.sources, test_data.destinations])
                timestamps_train = np.concatenate([test_data.timestamps, test_data.timestamps])

                tgn.embedding_module.streaming_topk_node(source_nodes=train_src, timestamps=timestamps_train, edge_idxs=test_data.edge_idxs)
                ptr2tppr_dict = tgn.embedding_module.tppr_finder.PPR_list
                np_adj = tppr_to_sparse_pandas(ptr2tppr_dict, graph_num+1)
                return np_adj
            edge_index = np.vstack((test_data.sources, test_data.destinations))
            edge_index = torch.from_numpy(edge_index).long().to(device)
            if args.matrix_value == "tppr":
                np_adj = get_tppr_value()
            elif args.matrix_value == "laplacian":
                # Get the Laplacian matrix
                np_index, np_value = get_laplacian(edge_index, normalization='sym', num_nodes=node_num, dtype=torch.float32)
                np_adj = csr_matrix((np_value.cpu().numpy(), (np_index[0].cpu().numpy(), np_index[1].cpu().numpy())), shape=(node_num, node_num))
                np_adj = np_adj.tocsr()
            elif args.matrix_value == "ppr":
                torch_adj = compute_ppr_torch(num_nodes = graph_num+1, edge_index = edge_index, alpha=0.1, num_iters=50, device=device)
                np_adj = torch_sparse_to_scipy_csr(torch_adj)
            # Ensure the output directory exists
            output_dir = Path(f"data/matrix/{DATA}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the matrix with the appropriate name
            output_path = output_dir / f"{args.matrix_value}_value_{VIEWS}_{i}.npz"
            save_npz(output_path, np_adj, compressed=True)
            print(f"Saved TPPR matrix to {output_path}")
    
