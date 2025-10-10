import numpy as np
import torch
from torch import nn, Tensor
import os
import sys
import datetime
import json
import argparse
from model.tgn_model import TGN
from typing import Dict, List, Tuple, Optional, Union

class EarlyStopMonitor(object):
    def __init__(self, max_round, dataset, snapshot, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        
        self.snapshot = snapshot
        self.preset_path = rf"./log/{dataset}"
        self.get_start_time = datetime.datetime.now()
        self.folder_day_time = self.get_start_time.strftime('%m-%d')
        self.file_exact_time = self.get_start_time.strftime('%H-%M-%S')
        
    def model_store(self, model: TGN, current_snapshot: int):
        self.folder_path = os.path.join(self.preset_path, self.folder_day_time, self.file_exact_time)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        model_path = os.path.join(self.folder_path, f'best_model_{self.snapshot}_currentSnapshot_{current_snapshot}.pth')
        # store the best model
        torch.save(model.state_dict(), model_path)
        return

    def data_store(self, data: Dict):
        file_name = os.path.join(self.folder_path, f'val_test_result_Snapshot_{self.snapshot}.txt')
        with open(file_name, "w") as file:
            for idx, recs in enumerate(data):
                val, tests = recs
                print(f"At snapshot {idx}, we get:")
                file.write(f"At snapshot {idx}, we get:\n")
                
                for metric, value in val.items():
                    print(f"avg {metric}: {value}")
                    file.write(f"avg {metric}: {value}\n")
                
                print("\n")
                file.write("\n")
                
                for metric, value in tests.items():
                    print(f"test {metric}: {value}")
                    file.write(f"test {metric}: {value}\n")
                
                print("\n")
                file.write("\n")

    def store_args(self, args: Union[argparse.Namespace, Dict], format_type: str = 'both'):
        """
        Store parser arguments in both TXT and JSON formats
        
        Args:
            args: Either argparse.Namespace object or dictionary containing arguments
            format_type: 'txt', 'json', or 'both' to specify output format(s)
        """
        # Ensure folder exists
        if not hasattr(self, 'folder_path') or not os.path.exists(self.folder_path):
            self.folder_path = os.path.join(self.preset_path, self.folder_day_time, self.file_exact_time)
            os.makedirs(self.folder_path, exist_ok=True)
        
        # Convert argparse.Namespace to dictionary if needed
        if isinstance(args, argparse.Namespace):
            args_dict = vars(args)
        else:
            args_dict = args
            
        # Add metadata
        metadata = {
            "timestamp": self.get_start_time.isoformat(),
            "execution_date": self.get_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "snapshot_count": self.snapshot
        }
        
        # Store in TXT format
        if format_type in ['txt', 'both']:
            txt_file_path = os.path.join(self.folder_path, f'training_args_snapshot_{self.snapshot}.txt')
            with open(txt_file_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("TRAINING CONFIGURATION PARAMETERS\n")
                f.write("="*60 + "\n")
                f.write(f"Execution Time: {metadata['execution_date']}\n")
                f.write(f"Snapshot Count: {metadata['snapshot_count']}\n")
                f.write("-"*60 + "\n\n")
                
                # Group arguments by category for better readability
                model_args = {}
                training_args = {}
                data_args = {}
                kernel_args = {}
                other_args = {}
                
                for key, value in args_dict.items():
                    if key in ['n_layer', 'n_head', 'drop_out', 'node_dim', 'time_dim', 'memory_dim', 
                              'embedding_module', 'message_function', 'aggregator', 'memory_updater']:
                        model_args[key] = value
                    elif key in ['n_epoch', 'bs', 'lr', 'patience', 'enable_random', 'gpu']:
                        training_args[key] = value
                    elif key in ['data', 'n_degree', 'topk', 'ignore_edge_feats', 'ignore_node_feats']:
                        data_args[key] = value
                    elif key in ['time_rff_dim', 'time_rff_sigma', 'graph_mu', 'fusion_mode', 'alpha_list', 'beta_list']:
                        kernel_args[key] = value
                    else:
                        other_args[key] = value
                
                # Write categorized arguments
                categories = [
                    ("MODEL ARCHITECTURE", model_args),
                    ("TRAINING PARAMETERS", training_args), 
                    ("DATA PARAMETERS", data_args),
                    ("KERNEL PARAMETERS", kernel_args),
                    ("OTHER PARAMETERS", other_args)
                ]
                
                for cat_name, cat_args in categories:
                    if cat_args:
                        f.write(f"{cat_name}:\n")
                        f.write("-" * len(cat_name) + "\n")
                        for key, value in sorted(cat_args.items()):
                            f.write(f"  {key:25} = {value}\n")
                        f.write("\n")
        
        # Store in JSON format
        if format_type in ['json', 'both']:
            json_file_path = os.path.join(self.folder_path, f'training_args_snapshot_{self.snapshot}.json')
            
            # Create comprehensive JSON structure
            json_data = {
                "metadata": metadata,
                "arguments": args_dict,
                "argument_summary": {
                    "total_parameters": len(args_dict),
                    "model_type": args_dict.get('embedding_module', 'unknown'),
                    "dataset": args_dict.get('data', 'unknown'),
                    "fusion_strategy": args_dict.get('fusion_mode', 'unknown')
                }
            }
            
            with open(json_file_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        print(f"Arguments stored in {self.folder_path}")
        return self.folder_path

    def early_stop_check(self, curr_val: float, model: TGN, current_snapshot: int) -> bool:
        # should be regarded as the core function, model store, data store should all be called after checking
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            self.model_store(model, current_snapshot)
        else:
            self.num_round += 1
            self.epoch_count += 1
        return self.num_round >= self.max_round
    
    