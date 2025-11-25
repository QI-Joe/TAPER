# Zebra-Laplacian

## TAPER: Temporal Attenuated PageRank with Spectral Regularization

TAPER integrates temporal Personalized PageRank (TPPR) with time-aware spectral diffusion on the normalized Laplacian to improve sensitivity for newly emergent nodes and edges in dynamic graphs, addressing TPPR's limitation in capturing rapidly changing topology structures.

---

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn

## 🚀 Quick Start

### Basic Training

To train the model with default settings:

```bash
python train.py -d mathoverflow --n_epoch 100 --n_layer 2 --bs 200 --gpu 0
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-d, --data` | Dataset name (mathoverflow, dblp, cora, etc.) | tmall |
| `--n_epoch` | Number of training epochs | 600 |
| `--n_layer` | Number of network layers | 2 |
| `--bs` | Batch size | 10000 |
| `--lr` | Learning rate | 5e-4 |
| `--snapshot` | Number of temporal snapshots | 15 |
| `--gpu` | GPU device index | 0 |

### TAPER-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fusion_mode` | Kernel fusion strategy (product, cosine, harmonic, etc.) | cosine |
| `--time_rff_dim` | Random frequencies for Bochner temporal kernel | 16 |
| `--time_rff_sigma` | Bandwidth parameter for temporal kernel | 1.0 |
| `--graph_mu` | Diffusion parameter for spectral graph kernel | 1.0 |
| `--ablation_tppr` | Ablation test mode (merge, tdsl, tppr) | merge |

### Advanced Usage

**Training with specific kernel fusion:**
```bash
python train.py -d dblp --fusion_mode harmonic --time_rff_dim 32 --graph_mu 0.5
```

**Training with temporal dynamics:**
```bash
python train.py -d mathoverflow --dynamic True --snapshot 10 --n_epoch 50
```

**Training with different embedding strategies:**
```bash
python train.py -d cora --embedding_module diffusion --tppr_strategy streaming --topk 40
```

## 📊 Supported Datasets

- **Citation Networks**: cora, citeseer, dblp
- **Social Networks**: mathoverflow, askubuntu, stackoverflow  
- **E-commerce**: tmall
- **Academic**: mooc
- **Others**: dgraph

## 🔧 Model Architecture

The training script supports:
- **Temporal Graph Networks (TGN)** with memory modules
- **TPPR streaming** for efficient neighbor sampling
- **Kernel fusion** combining temporal and spectral information
- **Multi-head attention** mechanisms
- **Early stopping** with patience-based monitoring

## 📈 Output

The model automatically:
- Saves training logs with timestamps
- Monitors validation performance
- Implements early stopping
- Records timing information across snapshots
- Stores best model checkpoints (with `--save_best`)

## 🎯 Example Commands

**Basic node classification:**
```bash
python train.py -d mathoverflow --n_epoch 50 --bs 200 --lr 0.001
```

**Large-scale training:**
```bash
python train.py -d tmall --bs 10000 --n_epoch 200 --snapshot 20
```

**Ablation study:**
```bash
python train.py -d dblp --ablation_tppr tppr --fusion_mode product
```
 
