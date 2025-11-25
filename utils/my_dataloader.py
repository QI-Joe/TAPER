import pandas as pd
import numpy as np
import torch
import random
import math
from torch_geometric.data import Data
import os
from typing import Any, Union, Optional, Tuple
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T


MOOC, Mooc_extra = "Temporal_Dataset/act-mooc/act-mooc/", ["mooc_action_features", "mooc_action_labels", "mooc_actions"]
OVERFLOW = r"../TestProejct/Temporal_Dataset/"
STATIC = ["mathoverflow", "dblp", "askubuntu", "stackoverflow", "mooc", "dgraph"]
DYNAMIC = ["mathoverflow", "askubuntu", "stackoverflow"]
TGB = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
LARGE_SCALE = ["tmall", "tax51"]


class Temporal_Dataloader(Data):
    """
    an overrided class of Data, to store splitted temporal data and reset their index 
    which support for fast local idx and global idx mapping/fetching
    """
    def __init__(self, nodes: Union[list, np.ndarray], edge_index: np.ndarray,\
                  edge_attr: Union[list|np.ndarray], y: list,\
                    pos: torch.Tensor) -> None:
        
        super(Temporal_Dataloader, self).__init__(x = nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        self.x = nodes
        self.edge_index = edge_index
        self.ori_edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.kept_train_mask = None
        self.kept_val_mask = None
        self.train_mask = None
        self.test_mask = None
        self.val_mask = None

        self.layer2_n_id: pd.DataFrame = None
        self.edge_pos: np.ndarray = None
        
        self.robustness_task: str = None
        self.side_initial()
    
    def side_initial(self):
        self.nn_val_mask, self.nn_test_mask = None, None
    
    def mask_(self, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def test_fast_sparse_build(self, key: np.ndarray, value: np.ndarray) -> torch.Tensor:
        r"""
        Without considering idx matching, assume that tppr_node idx is consistent with current node idx
        """
        return [[src, dst, value[src, idx]] \
                    for src in range(key.shape[0]) \
                    for idx, dst in enumerate(key[src]) \
                        if dst>0 or value[src, idx] > 0]

    def train_val_mask_injection(self, train_mask: np.ndarray, val_mask: np.ndarray, nn_val_mask: np.ndarray):
        """
        train_mask: param; a boolean array, True means the node is selected as training node
        val_mask: param; a boolean array, True means the node is selected as validation node
        nn_val_mask: param; a boolean array, True means the node is selected as validation node
        """
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.nn_val_mask = nn_val_mask
        
    def test_mask_injection(self, nn_test_mask: np.ndarray):
        self.nn_test_mask = nn_test_mask


class Temporal_Splitting(object):
    def __init__(self, graph: Data, dynamic: bool) -> None:
        
        super(Temporal_Splitting, self).__init__()
        self.graph = graph 

        if not dynamic and isinstance(self.graph.x, torch.Tensor) and self.graph.x.size(1) > 1:
            self.graph.pos = self.graph.x
            self.graph.x = np.arange(self.graph.x.size(0))

        if isinstance(self.graph.edge_attr, type(None)):
            self.graph.edge_attr = np.arange(self.graph.edge_index.size(1))

        self.dynamic = dynamic
        self.temporal_list: list[Temporal_Dataloader] = []
        self.set_mapping: dict = None

    def sampling_layer(self, snapshots: int, views: int, span: float, strategy: str):
        T = []
        if strategy == 'low_overlap':
            if (0.75 * views + 0.25) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
            T = [start + (0.75 * i * span) / snapshots for i in range(views)]
        elif strategy == 'high_overlap':
            if (0.25 * views + 0.75) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
            T = [start + (0.25 * i * span) / snapshots for i in range(views)]
        elif strategy == 'sequential':
            T = [span * i / (snapshots-1) for i in range(1, snapshots)]
            if views > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
        random.seed(2025)
        np.random.seed(2025)
        T = random.sample(T, views)
        T= sorted(T)
        if span not in T: T[-1] = span
        if T[0] == float(0):
            T.pop(0)
        return T

    def time_sequential_select(self, snapshot: int, views: int, edge_attr):
        unique_time=np.unique(edge_attr)
        if views+1 > len(unique_time):
            raise ValueError("The number of sampled views exceeds the maximum value of the current policy.")
        batch = len(unique_time) // snapshot
        T_list = unique_time[batch::batch]
        random.seed(2025)
        np.random.seed(2025)
        T = random.sample(T_list.tolist(), views)
        T= sorted(T)
        if unique_time[-1] not in T: 
            T.pop(6)
            T.append(unique_time[-1].item())
        return T

    def temporal_splitting(self, snapshot, **kwargs) -> list[Data]:
        """
        currently only suitable for CLDG dataset, to present flexibilty of function Data\n
        log 12.3:\n
        Temporal and Any Dynamic data loader will no longer compatable with static graph
        """
        edge_index = self.graph.edge_index
        edge_attr = self.graph.edge_attr
        pos = self.graph.pos

        max_time = max(edge_attr)
        temporal_subgraphs = []

        T: list = []

        span = (max(edge_attr) - min(edge_attr)).item()
        views, strategy = snapshot-2, "sequential"
        T = self.sampling_layer(snapshot, views, span, strategy)
        if self.graph.edge_index.shape[1] > 1_000_000:
            T = self.time_sequential_select(snapshot, views, edge_attr)
        
        for idx, start in enumerate(T):
            if start<0.01: continue

            sample_time = start

            end = min(T[idx], max_time)
            sample_time = (edge_attr<=end)
            sampled_edges = edge_index[:, sample_time]
            sampled_nodes = torch.unique(sampled_edges) if isinstance(sampled_edges, torch.Tensor) else np.unique(sampled_edges)
        
            y = self.get_label_by_node(sampled_nodes)
            subpos = pos[self.sample_idx(sampled_nodes)]

            # Zebra node classification will not refresh node idx...To maintain global node structure
            temporal_subgraph = Temporal_Dataloader(nodes=sampled_nodes, edge_index=sampled_edges, \
                                                edge_attr=edge_attr[sample_time], y=y, pos=subpos)
            temporal_subgraphs.append(temporal_subgraph)

        return temporal_subgraphs


def position_encoding(max_len, emb_size)->torch.Tensor:
    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def load_dblp_interact(path: str = None, dataset: str = "dblp", *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join("/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/", dataset, '{}.txt'.format(dataset)), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join('/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/', dataset, 'node2label.txt'), sep=' ', names=['node', 'label'])

    return edges, label

def load_static_overflow(prefix: str, path: str=None, *wargs) -> tuple[Data, pd.DataFrame]:
    dataset = "sx-"+prefix
    path = OVERFLOW + prefix + r"/static"
    edges = pd.read_csv(os.path.join(path, dataset+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label.txt"), sep=' ', names=['node', 'label'])
    return edges, label

def load_mooc(path:str=None) -> Tuple[pd.DataFrame]:
    feat = pd.read_csv(os.path.join(path, "mooc_action_features.tsv"), sep = '\t')
    general = pd.read_csv(os.path.join(path, "mooc_actions.tsv"), sep = '\t')
    edge_label = pd.read_csv(os.path.join(path, "mooc_action_labels.tsv"), sep = '\t')
    return general, feat, edge_label

def edge_load_mooc(dataset:str):
    auto_path = r"../Standard_Dataset/lp/act-mooc/act-mooc"
    edge, feat, label = load_mooc(auto_path)
    # for edge, its column idx is listed as ["ACTIONID", "USERID", "TARGETID", "TIMESTAMP"]
    edge = edge.values
    edge_idx, src2dst, timestamp = edge[:, 0], edge[:, 1:3].T, edge[:, 3]
    
    print(src2dst.dtype, src2dst.shape)
    src2dst = src2dst.astype(np.int64)
    
    edge_pos = feat.iloc[:, 1:].values
    y = label.iloc[:, 1].values
    
    node = np.unique(src2dst).astype(np.int64)
    max_node = int(np.max(node)) + 1
    if dataset == "mooc":
        node = np.unique(src2dst[0])
    node_pos = position_encoding(max_node, 64)# .numpy()

    graph = Data(x = node, edge_index=src2dst, edge_attr=timestamp, y = y, pos = node_pos)
    return graph

def dynamic_label(edges: pd.DataFrame, combination_dict: dict) -> pd.DataFrame:
    """
    Very slow when facing large dataset. Recommend to use function matrix_dynamic_label
    """
    unique_node = edges[["src", "dst"]].stack().unique()
    node_label: list[tuple[int, int]] = []
    for node in unique_node:
        appearance = edges[(edges.src == node) | (edges.dst == node)].apprarance.values
        appearance = tuple(set(appearance))
        node_label.append((node, combination_dict[appearance]))
    return pd.DataFrame(node_label, columns=["node", "label"])

def get_dataset(path, name: str):
    assert name.lower() in [val.lower() for val in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']]
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS'.lower():
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy'.lower():
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS'.lower():
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers'.lower():
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo'.lower():
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

def load_standard(dataset: str, **wargs) -> Data:
    
    path = osp.expanduser('~/datasets')
    path = osp.join(path, dataset)
    dataset = get_dataset(path, dataset)
    return dataset

def load_static_dataset(path: str = None, dataset: str = "mathoverflow", fea_dim: int = 64, **wargs) -> Temporal_Dataloader:
    """
    Now this txt file only limited to loading data in from mathoverflow datasets
    path: (path, last three words of dataset) -> (str, str) e.g. ('data/mathoverflow/sx-mathoverflow-a2q.txt', 'a2q')
    node Idx of mathoverflow is not consistent!
    """
    if dataset == "dblp":
        edges, label = load_dblp_interact() if not path else load_dblp_interact(path)
    elif dataset == "mooc":
        return edge_load_mooc(dataset)

    x = label.node.to_numpy()
    nodes = position_encoding(x.max()+1, fea_dim)[x]
    labels = torch.tensor(label.label.to_numpy())

    edge_index = torch.tensor(edges.loc[:, ["src", "dst"]].values.T)
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = torch.tensor(edges.time)

    graph = Data(x=x, edge_index=edge_index, edge_attr=time, y=labels, pos = nodes)
    # neighborloader = NeighborLoader(graph, num_neighbors=[10, 10], batch_size =2048, shuffle = False
    return graph

def load_padded_dataset(dataset: str = "tmall", fea_dim: int = 64, **wargs) -> Data:
    """
    tmall and tax51 are large scale datasets, but its node label is not completed, so we use
    -1 to pad the node label, and use position encoding to generate node features
    """
    if dataset in LARGE_SCALE:
        edges, label = load_dblp_interact(dataset=dataset)
    else:
        raise ValueError("Dataset not found or not supported")
    
    x = label.node.to_numpy()
    labels = np.array(label.label)
    unique_labels = np.unique(labels)
    new_labels = np.arange(unique_labels.shape[0])
    new_old_label_match = np.vectorize({old : new for old, new in zip(unique_labels, new_labels)}.get)
    consistent_labels = new_old_label_match(labels)

    edge_index = np.array(edges.loc[:, ["src", "dst"]].values.T)
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = np.array(edges.time)

    max_node, min_node = edge_index.max(), edge_index.min()
    
    if min_node != x.min():
        print(f"Attention! The minimum node index in edge_index is {min_node}, but the minimum node index in label is {x.min()}.")
        print(f"The max node index in edge_index is {max_node}, but the max node index in label is {x.max()}.")
        print("The node index in edge_index will be padded to match the label node index.")
    
    full_x = np.arange(min_node, max_node+1)
    full_labels = np.full_like(full_x, -1, dtype=int)
    full_labels[x] = consistent_labels
    full_labels = torch.from_numpy(full_labels)
    
    node_feature = position_encoding(max_node+1, fea_dim)

    graph = Data(x=full_x, edge_index=edge_index, edge_attr=time, y=full_labels, pos = node_feature)
    return graph

def load_tsv(path: list[tuple[str]], *wargs) -> tuple[pd.DataFrame]:
    """
    Note this function only for loading data in act-mooc dataset
    """
    dfs:dict = {p[1]: pd.read_csv(p[0], sep='\t') for p in path}

    label = dfs["mooc_action_labels"]
    action_features = dfs["mooc_action_features"]
    actions = dfs["mooc_actions"]
    return label, action_features, actions

def load_example():
    return "node_feat", "node_label", "edge_index", "train_indices", "val_indices", "test_indices"

def data_load(dataset: str, **wargs) -> Temporal_Dataloader:
    dataset = dataset.lower()
    
    if dataset in STATIC:
        graph = load_static_dataset(dataset=dataset, **wargs)
    elif dataset in ["cora", "citeseer", "wikics"] :
        graph = load_standard(dataset, **wargs)[0]

        nodes = [i for i in range(graph.x.shape[0])]
        graph.pos = graph.x
        graph.x = np.array(nodes)
        graph.edge_index = graph.edge_index.numpy()
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph.y = graph.y.numpy()
    elif dataset in LARGE_SCALE:
        return load_padded_dataset(dataset=dataset, **wargs)
    else:
        raise ValueError("Dataset not found")
    
    task: Optional[str|None] = wargs["rb_task"]
    if task != None: task = task.lower()
    return graph

    