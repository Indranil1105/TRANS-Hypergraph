Requirements and Setup for HGNN MIMIC Project
Project Overview
This project implements Hypergraph Neural Networks (HGNN) as a replacement for heterographs in the TRANS model for Electronic Health Record (EHR) prediction tasks using MIMIC-IV data.

Key Features
Hypergraph Construction: Replace heterogeneous graphs with hypergraphs to capture high-order relationships

Multi-modal Medical Data: Handle diagnoses, procedures, and medications

Temporal Awareness: Incorporate time information in hypergraph convolutions

Scalable Architecture: Support for different HGNN variants

Comprehensive Evaluation: Multiple metrics and visualization tools

File Structure
text
project/
├── hypergraph_utils.py          # Hypergraph construction utilities
├── hgnn_layers.py              # Neural network layers for HGNN
├── ehr_data_processor.py       # MIMIC data preprocessing
├── hgnn_models.py              # Different HGNN model architectures
├── train_hgnn.py               # Training script
├── analyze_results.py          # Result analysis and visualization
├── demo.py                     # Complete demo pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/
    ├── ADMISSIONS.csv          # MIMIC admission data
    ├── PATIENTS.csv            # MIMIC patient data
    ├── DIAGNOSES_ICD.csv       # MIMIC diagnosis data
    ├── PROCEDURES_ICD.csv      # MIMIC procedure data
    └── processed_mimic_data.pkl # Preprocessed data
Requirements
Python Version
Python 3.8+

Core Dependencies
text
torch>=1.12.0
torch-geometric>=2.3.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.6.0
tensorboard>=2.8.0
tqdm>=4.62.0
Optional Dependencies
text
jupyter>=1.0.0              # For notebook examples
plotly>=5.0.0               # Interactive visualizations
Installation
Clone or download the project files

Install dependencies:

bash
pip install -r requirements.txt
Install PyTorch Geometric:

bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
(Adjust URL based on your PyTorch version and CUDA support)

Data Setup
Download MIMIC-IV data (requires credentialed access):

Visit https://physionet.org/content/mimiciv/

Download the required CSV files

Place them in the project directory

Preprocess the data:

python
from ehr_data_processor import EHRDataProcessor

processor = EHRDataProcessor()
# This will create processed_mimic_data.pkl
Usage
Quick Start - Run Demo
bash
python demo.py
This will run a complete demo showing:

Data preprocessing

Hypergraph construction

Model training (few epochs)

Evaluation and comparison

Full Training
bash
# Basic HGNN model
python train_hgnn.py --model_type basic --epochs 100 --batch_size 32

# Advanced HGNN with multi-head attention
python train_hgnn.py --model_type advanced --epochs 100 --num_heads 8

# Temporal HGNN
python train_hgnn.py --model_type temporal --epochs 100 --use_temporal True
Model Types Available
basic: Standard HGNN with hypergraph convolution layers

advanced: Multi-head attention hypergraph convolution

hierarchical: Models patient-visit-code hierarchy

multimodal: Separate processing for different medical modalities

temporal: Time-aware hypergraph convolution

Training Parameters
bash
python train_hgnn.py \
    --model_type basic \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --dropout 0.5 \
    --hidden_dim 256 \
    --k_neig 10 \
    --feature_type statistical
Analyze Results
bash
python analyze_results.py
This will generate:

Training curves

Hypergraph structure analysis

Model performance comparisons

Prediction pattern analysis

Key Components
1. Hypergraph Construction
The HypergraphUtils class provides methods for:

K-nearest neighbor hypergraph construction

Medical co-occurrence pattern hypergraphs

Multi-modal hypergraph fusion

Hypergraph Laplacian computation

2. HGNN Layers
Implemented layers include:

HGNN_conv: Basic hypergraph convolution

MultiHeadHGNN_conv: Multi-head attention version

TemporalHGNN_conv: Time-aware convolution

HypergraphAttentionLayer: Attention over hyperedges

3. Model Architectures
Five different HGNN architectures:

Basic: Simple stacked hypergraph convolutions

Advanced: Multi-head attention and residual connections

Hierarchical: Three-level patient-visit-code modeling

Multimodal: Separate processing for different medical data types

Temporal: Incorporates temporal information explicitly

4. Evaluation Metrics
Comprehensive evaluation including:

Micro/Macro averaged Precision, Recall, F1

AUC scores

Top-K accuracy for multi-label classification

Per-class performance analysis

Comparison with Original TRANS
Aspect	TRANS (Original)	HGNN (This Implementation)
Graph Type	Heterogeneous	Hypergraph
Relationships	Pairwise only	High-order (3+ nodes)
Scalability	Limited by graph size	Better for large datasets
Medical Patterns	Visit-level	Population-level
Complexity	Higher	More straightforward
Key Advantages of Hypergraph Approach
High-order Relationships: Capture complex medical co-occurrence patterns

Simplified Architecture: More direct mathematical formulation

Better Generalization: Population-level pattern learning

Flexible Integration: Easy multi-modal data handling

Interpretability: Clear hyperedge structure meaning

Troubleshooting
Common Issues
CUDA out of memory: Reduce batch_size or hidden_dim

Import errors: Check PyTorch Geometric installation

Data not found: Ensure MIMIC data files are in correct location

Low performance: Try different feature types or increase k_neig

Performance Tips
Use GPU if available: --device cuda

Experiment with different k_neig values (5-20)

Try different feature types: statistical, embedding, cooccurrence

Adjust hypergraph construction parameters

Expected Results
With the demo dataset, you should expect:

Training converges within 50-100 epochs

Micro F1 score: 0.3-0.6 (depends on data quality)

Top-5 accuracy: 0.4-0.7

Better performance than random baseline

Citation
If you use this code, please cite the original TRANS paper and HGNN paper:

text
@article{chen2024trans,
  title={Predictive Modeling with Temporal Graphical Representation on Electronic Health Records},
  author={Chen, Jiayuan and Yin, Changchang and Wang, Yuanlong and Zhang, Ping},
  year={2024}
}

@article{feng2019hypergraph,
  title={Hypergraph Neural Networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
  journal={AAAI},
  year={2019}
}
Support
For questions or issues:

Check the troubleshooting section above

Review the demo.py for usage examples

Examine the analyze_results.py for result interpretation

License
This project is for educational and research purposes. Please respect the MIMIC-IV data usage requirements and licensing terms.