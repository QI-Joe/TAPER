import numpy as np
import random
from utils.my_dataloader import data_load, Temporal_Splitting, Temporal_Dataloader
import torch
import copy
from typing import Union, Optional

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, hash_table: dict[int, int], node_feat: np.ndarray = None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0
        self.node_feat = node_feat
        self.hash_table = hash_table

        self.target_node: Optional[set|None] = None
  
  def set_up_features(self, node_feat, edge_feat):
    self.node_feat = node_feat
    self.edge_feat = edge_feat

  def call_for_inductive_nodes(self, val_data: 'Data', test_data: 'Data', single_graph: bool):
    validation_node: set = val_data.unique_nodes
    test_node: set = test_data.unique_nodes
    train_node = self.unique_nodes

    common_share = validation_node & test_node & train_node
    train_val_share = validation_node & train_node
    train_test_share = train_node & test_node
    val_test_share = validation_node & test_node

    expected_val = list(validation_node - (common_share | train_val_share))

    if single_graph:
      expected_test = list(test_node - (train_test_share | common_share | val_test_share))
      test_data.propogator_back(expected_test, single_graph = single_graph)
    else:
      t_times_common_data = list(train_test_share | common_share | val_test_share)
      t_times_hash_table = val_data.hash_table
      test_data.propogator_back(t_times_common_data, single_graph, t_times_hash_table)

    assert len(set(expected_val) & train_node) == 0, "train_node data is exposed to validation set"
    if single_graph:
      assert len(set(expected_test) & train_node & set(expected_val)) == 0, "train node and val data has interacted with test data"

    val_data.propogator_back(expected_val, single_graph=True)
    
    self.propogator_back(train_node, single_graph=True)
    return 

  def edge_mask(self, data: 'Data', test_element: set):
    test_element = sorted(test_element)
    src_mask = ~np.isin(data.sources, test_element)
    dst_mask = ~np.isin(data.destinations, test_element)
    return src_mask & dst_mask

  def cover_the_edges(self, val_data: 'Data', test_data: 'Data', single_graph: bool = True):
    """
    delete both edges and nodes appeared in train_data to make pure inductive val_data \n
    also, delete both edges and nodes appeared in train_data and val_data to make pure inductive test_data
    """
    valid_node = val_data.unique_nodes
    test_node = test_data.unique_nodes
    train_node = self.unique_nodes

    # common_share = valid_node & test_node & train_node
    train_val_share = valid_node & train_node
    train_test_share = train_node & test_node
    val_test_share = valid_node & test_node

    node_2be_removed_val = train_val_share
    node_2be_removed_test = val_test_share | train_test_share

    val_data.edge_propagate_back(self.edge_mask(val_data, node_2be_removed_val))
    test_data.edge_propagate_back(self.edge_mask(test_data, node_2be_removed_test))

    return

  def edge_propagate_back(self, edge_mask: np.ndarray):
    """
    keep the edge mask as permanent variable and modify edge \n
    maintain edges to inductive edge mask
    """
    self.inductive_edge_mask = edge_mask
    self.sources = self.sources[self.inductive_edge_mask]
    self.destinations = self.destinations[self.inductive_edge_mask]
    self.timestamps = self.timestamps[self.inductive_edge_mask]

  def propogator_back(self, node_idx: list, single_graph: bool, t_hash_table: Union[dict | None] = None):
    """
    Expected to clear the node index get establish the mask mainly for non-visible data node \n

    :Attention: meaning of node_idx is different(reversed) when single_graph is different!!!

    :param node_idx when single_graph is True -- is the node that uniquness to the given data object,
    :param node_idx when single_graph is False -- it represent node that should be removed in given node set!!!

    :return self.target_node -- whatever how node_idx and single_graph changed, it always present node to be Uniquness
    to the given Data object
    """
    batch_nodes = np.array(sorted(self.unique_nodes))
    if single_graph:
      # self.target_node_mask = np.isin(batch_nodes, sorted(node_idx))
      self.target_node = self.unique_nodes & set(node_idx)
    else:
      t_transfer_map = np.vectorize(t_hash_table.get)
      t1_transfer_map = np.vectorize(self.hash_table.get)

      seen_nodes = t_transfer_map(node_idx)
      test_seen_nodes = t1_transfer_map(batch_nodes)

      test_node = set(test_seen_nodes) - set(seen_nodes)
      reverse_test_hashtable = {v:k for k, v in self.hash_table.items()}
      t1_back_transfer = np.vectorize(reverse_test_hashtable.get)
      t_test_node = t1_back_transfer(test_node)

      # self.target_node_mask = np.isin(batch_nodes, sorted(t_test_node))
      self.target_node = self.unique_nodes - t_test_node

    return

  def sample(self,ratio):
    data_size=self.n_interactions
    sample_size=int(ratio*data_size)
    sample_inds=random.sample(range(data_size),sample_size)
    sample_inds=np.sort(sample_inds)
    sources=self.sources[sample_inds]
    destination=self.destinations[sample_inds]
    timestamps=self.timestamps[sample_inds]
    edge_idxs=self.edge_idxs[sample_inds]
    labels=self.labels[sample_inds]
    return Data(sources,destination,timestamps,edge_idxs,labels)


def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr.numpy() if isinstance(graph.edge_attr, torch.Tensor) else graph.edge_attr
    if isinstance(graph.edge_index, torch.Tensor):
        graph.edge_index = graph.edge_index.numpy()
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
    
    if np.any(graph.edge_attr != None):
        edge_attr = graph.edge_attr
    if np.any(graph.pos != None):
        pos = graph.pos
        pos = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    else:
        pos = graph.x

    TPPR_data = Data(sources= src, destinations=dest, timestamps=timestamp, edge_idxs = edge_idx, labels=labels, hash_table=hash_table, node_feat=pos)

    return TPPR_data

def quantile_(threshold: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(threshold*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:] = True

  return train_mask, val_mask

def quantile_static(val: float, test: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(val*full_length)
  test_idx = int(test*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:test_idx] = True

  test_mask = torch.zeros_like(timestamps, dtype=bool)
  test_mask[test_idx:] = True

  return train_mask, val_mask, test_mask

def get_simplified_temporal_data(dataset_name, snapshot: int, dynamic: bool, task: str="None", ratio: float = 0.0)-> list[list[Data], int, np.ndarray, int]:
    """
    Simplified function for inductive validation without robustness/edge_disturb/fsl settings
    
    Input:
    - dataset_name: name of the dataset 
    - snapshot: number of temporal snapshots
    - dynamic: whether to use dynamic temporal splitting
    
    Output:
    - TPPR_list: list of [full_data, train_data, val_data, test_data, node_num, edge_num] for each snapshot
    - graph_num_node: max node number
    - graph_feat: node features
    - edge_number: total number of edges
    """
    # Load data without robustness settings
    wargs = {"rb_task": task, "ratio": ratio}
    graph = data_load(dataset_name, **wargs)
    
    if snapshot <= 3:
        # For small snapshots, treat as single graph
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph_list = [Temporal_Dataloader(nodes=graph.x, edge_index=graph.edge_index, 
                                          edge_attr=graph.edge_attr, y=graph.y, pos=graph.pos)]
    else:
        # Use temporal splitting
        graph_list = Temporal_Splitting(graph, dynamic=dynamic).temporal_splitting(snapshot=snapshot)
    
    graph_num_node = max(graph.x) if hasattr(graph, 'x') else len(graph.x)
    graph_feat = copy.deepcopy(graph.pos)
    edge_number = graph.edge_index.shape[1]
    
    TPPR_list, full_label = [], graph.y
    length = len(graph_list) - 1 if len(graph_list) > 1 else 1
    single_graph = length < 2
    
    for idx in range(length):
        items = graph_list[idx]
        temporal_node_num = items.x.shape[0]
        
        # Convert to Data object
        full_data = to_TPPR_Data(items)
        timestamp = full_data.timestamps
        
        # Node-based splitting following original pattern
        all_nodes = items.my_n_id.node["index"].values
        flipped_nodes = items.my_n_id.node["node"].values
        
        # Simple temporal split: 80% train, 20% val for current snapshot
        train_node_boundary = int(temporal_node_num * 0.8)
        train_nodes = all_nodes[:train_node_boundary]
        train_nodes_original = flipped_nodes[:train_node_boundary]
        val_nodes = all_nodes[train_node_boundary:]
        val_nodes_original = flipped_nodes[train_node_boundary:]
        
        # Create edge masks based on node participation (transductive approach)
        src_edge = items.edge_index[0, :]
        dst_edge = items.edge_index[1, :]
        
        # Train mask: both src and dst in train nodes
        train_src_mask = np.isin(src_edge, train_nodes_original)
        train_dst_mask = np.isin(dst_edge, train_nodes_original)
        train_mask = train_src_mask & train_dst_mask
        
        # Val mask: edges involving validation nodes (transductive: can see train nodes)
        val_src_mask = np.isin(src_edge, val_nodes_original)
        val_dst_mask = np.isin(dst_edge, val_nodes_original)
        val_mask = val_src_mask | val_dst_mask  # Include edges with at least one val node
        
        # NN Val mask: both src and dst in validation nodes (for inductive evaluation)
        nn_val_mask = val_src_mask & val_dst_mask
        
        # Create hash table for node mapping
        hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
        hash_table = {node: idx for idx, node in zip(*hash_dataframe)}
        
        # Create train data
        train_data = Data(
            full_data.sources[train_mask], 
            full_data.destinations[train_mask], 
            full_data.timestamps[train_mask],
            full_data.edge_idxs[train_mask], 
            full_label, 
            hash_table=hash_table, 
            node_feat=full_data.node_feat
        )
        
        # Create validation data (transductive: can see train nodes)
        val_data = Data(
            full_data.sources[val_mask], 
            full_data.destinations[val_mask], 
            full_data.timestamps[val_mask],
            full_data.edge_idxs[val_mask], 
            full_label, 
            hash_table=hash_table, 
            node_feat=full_data.node_feat
        )
        
        # Create nn_val_data (inductive validation: only val-val edges)
        if nn_val_mask.sum() == 0:
            nn_val_data = copy.deepcopy(val_data)
        else:
            nn_val_data = Data(
                full_data.sources[nn_val_mask],
                full_data.destinations[nn_val_mask], 
                full_data.timestamps[nn_val_mask],
                full_data.edge_idxs[nn_val_mask],
                full_label,
                hash_table=hash_table,
                node_feat=full_data.node_feat
            )
        
        # Create test data
        if single_graph:
            # For single graph, use temporal splitting
            train_mask_time, val_mask_time, test_mask_time = quantile_static(val=0.70, test=0.85, timestamps=timestamp)
            test_data = Data(
                full_data.sources[test_mask_time], 
                full_data.destinations[test_mask_time], 
                full_data.timestamps[test_mask_time],
                full_data.edge_idxs[test_mask_time], 
                full_label, 
                hash_table=hash_table, 
                node_feat=full_data.node_feat
            )
            # For single graph, nn_test is same as test
            nn_test_data = copy.deepcopy(test_data)
        else:
            # Use next temporal graph as test
            test_graph = graph_list[idx + 1]
            test_data = to_TPPR_Data(test_graph)
            test_data.labels = full_label  # Use full labels
            
            # Create nn_test_data: test nodes not seen in training (inductive testing)
            test_nodes_original = np.array(sorted(set(test_graph.my_n_id.node["node"].values) - set(flipped_nodes)))
            test_src_mask = np.isin(test_data.sources, test_nodes_original)
            test_dst_mask = np.isin(test_data.destinations, test_nodes_original)
            
            test_mask = test_src_mask | test_dst_mask
            nn_test_mask = test_src_mask & test_dst_mask
            
            transduc_test_data = Data(
                test_data.sources[test_mask],
                test_data.destinations[test_mask],
                test_data.timestamps[test_mask],
                test_data.edge_idxs[test_mask],
                full_label,
                hash_table=test_data.hash_table,
                node_feat=test_data.node_feat
            )
            
            nn_test_data = Data(
                test_data.sources[nn_test_mask],
                test_data.destinations[nn_test_mask],
                test_data.timestamps[nn_test_mask],
                test_data.edge_idxs[nn_test_mask],
                full_label,
                hash_table=test_data.hash_table,
                node_feat=test_data.node_feat
            )
            if nn_test_data.sources.shape[0] == 0:
                nn_test_data = copy.deepcopy(transduc_test_data)
        
        node_num = items.num_nodes
        node_edges = items.num_edges
        
        # Follow original pattern: [full_data, train_data, val_data, test_data, train_data_edge_learn, nn_val_data, nn_test_data, node_num, node_edges]
        # For simplicity, we'll use train_data as train_data_edge_learn (can be refined if needed)
        train_data_edge_learn = copy.deepcopy(train_data)
        
        TPPR_list.append([full_data, train_data, val_data, test_data, transduc_test_data, train_data_edge_learn, nn_val_data, nn_test_data, node_num, node_edges])
    
    return TPPR_list, graph_num_node, graph_feat, edge_number

