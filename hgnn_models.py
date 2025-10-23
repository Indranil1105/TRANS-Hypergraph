# HGNN Model for EHR Prediction
# Hypergraph Neural Network adapted for Electronic Health Records

import torch
import torch.nn as nn
import torch.nn.functional as F
from hgnn_layers import HGNN_conv, HGNN_embedding, HGNN_classifier, MultiHeadHGNN_conv, TemporalHGNN_conv

class HGNN_EHR(nn.Module):
    """
    Hypergraph Neural Network for EHR prediction
    Based on HGNN architecture adapted for medical prediction tasks
    """
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.5, num_layers=2):
        super(HGNN_EHR, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.num_layers = num_layers
        
        # Hypergraph convolution layers
        self.hgc_layers = nn.ModuleList()
        
        # First layer
        self.hgc_layers.append(HGNN_conv(in_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.hgc_layers.append(HGNN_conv(hidden_dim, hidden_dim))
        
        # Output layer
        self.hgc_layers.append(HGNN_conv(hidden_dim, n_classes))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, G):
        """
        Forward pass
        :param x: Node features (batch_size, n_nodes, in_features)
        :param G: Hypergraph Laplacian (n_nodes, n_nodes)
        :return: Output predictions (batch_size, n_nodes, n_classes)
        """
        # Handle batch dimension
        if x.dim() == 3:
            batch_size = x.size(0)
            x = x.view(-1, x.size(-1))  # Flatten batch dimension
            
            # Apply hypergraph convolutions
            for i, hgc in enumerate(self.hgc_layers[:-1]):
                x = hgc(x, G)
                x = F.relu(x)
                x = self.dropout(x)
            
            # Final layer without activation for multi-label classification
            x = self.hgc_layers[-1](x, G)
            
            # Reshape back to batch format
            x = x.view(batch_size, -1, self.n_classes)
        else:
            # No batch dimension
            for i, hgc in enumerate(self.hgc_layers[:-1]):
                x = hgc(x, G)
                x = F.relu(x)
                x = self.dropout(x)
            
            x = self.hgc_layers[-1](x, G)
        
        return x

class HGNN_EHR_Advanced(nn.Module):
    """
    Advanced HGNN model with multi-head attention and temporal awareness
    """
    def __init__(self, in_features, hidden_dim, n_classes, num_heads=4, num_layers=3, 
                 dropout=0.5, use_temporal=False, time_dim=32):
        super(HGNN_EHR_Advanced, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        
        # Input projection
        self.input_projection = nn.Linear(in_features, hidden_dim)
        
        # Hypergraph layers
        self.hg_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_temporal:
                layer = TemporalHGNN_conv(hidden_dim, hidden_dim, time_dim=time_dim)
            else:
                layer = MultiHeadHGNN_conv(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            self.hg_layers.append(layer)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, G, time_stamps=None):
        """
        Forward pass with optional temporal information
        """
        # Input projection
        x = self.input_projection(x)
        
        # Apply hypergraph layers
        for i, (hg_layer, layer_norm) in enumerate(zip(self.hg_layers, self.layer_norms)):
            residual = x
            
            if self.use_temporal and time_stamps is not None:
                x = hg_layer(x, G, time_stamps)
            else:
                x = hg_layer(x, G)
            
            # Residual connection and layer norm
            x = layer_norm(x + residual)
            x = self.dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class HGNN_EHR_Hierarchical(nn.Module):
    """
    Hierarchical HGNN that models patient-visit-code relationships
    """
    def __init__(self, code_vocab_size, visit_dim, patient_dim, n_classes, 
                 embedding_dim=128, hidden_dim=256, dropout=0.5):
        super(HGNN_EHR_Hierarchical, self).__init__()
        
        # Code embeddings
        self.code_embedding = nn.Embedding(code_vocab_size, embedding_dim, padding_idx=0)
        
        # Visit-level hypergraph processing
        self.visit_hgnn = HGNN_embedding(embedding_dim, visit_dim, dropout=dropout)
        
        # Patient-level hypergraph processing  
        self.patient_hgnn = HGNN_embedding(visit_dim, patient_dim, dropout=dropout)
        
        # Classification head
        self.classifier = HGNN_classifier(patient_dim, n_classes)
        
    def forward(self, visit_codes, visit_hypergraph, patient_hypergraph):
        """
        Hierarchical forward pass
        :param visit_codes: Code indices for each visit (batch_size, max_visits, max_codes)
        :param visit_hypergraph: Visit-level hypergraph Laplacian
        :param patient_hypergraph: Patient-level hypergraph Laplacian
        """
        batch_size, max_visits, max_codes = visit_codes.size()
        
        # Code embeddings
        code_emb = self.code_embedding(visit_codes)  # (batch_size, max_visits, max_codes, embedding_dim)
        
        # Aggregate codes to visit level (simple mean pooling)
        visit_emb = torch.mean(code_emb, dim=2)  # (batch_size, max_visits, embedding_dim)
        
        # Reshape for hypergraph processing
        visit_emb = visit_emb.view(-1, visit_emb.size(-1))
        
        # Visit-level hypergraph processing
        visit_repr = self.visit_hgnn(visit_emb, visit_hypergraph)
        
        # Reshape back and aggregate to patient level
        visit_repr = visit_repr.view(batch_size, max_visits, -1)
        patient_emb = torch.mean(visit_repr, dim=1)  # Simple aggregation
        
        # Patient-level hypergraph processing
        patient_repr = self.patient_hgnn(patient_emb, patient_hypergraph)
        
        # Final classification
        logits = self.classifier(patient_repr)
        
        return logits

class HGNN_EHR_MultiModal(nn.Module):
    """
    Multi-modal HGNN that handles different types of medical data
    """
    def __init__(self, diag_vocab_size, proc_vocab_size, n_classes,
                 embedding_dim=128, hidden_dim=256, dropout=0.5):
        super(HGNN_EHR_MultiModal, self).__init__()
        
        # Separate embeddings for different modalities
        self.diag_embedding = nn.Embedding(diag_vocab_size, embedding_dim, padding_idx=0)
        self.proc_embedding = nn.Embedding(proc_vocab_size, embedding_dim, padding_idx=0)
        
        # Modality-specific hypergraph processing
        self.diag_hgnn = HGNN_embedding(embedding_dim, hidden_dim, dropout=dropout)
        self.proc_hgnn = HGNN_embedding(embedding_dim, hidden_dim, dropout=dropout)
        
        # Cross-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self, diag_codes, proc_codes, diag_hypergraph, proc_hypergraph):
        """
        Multi-modal forward pass
        :param diag_codes: Diagnosis code indices
        :param proc_codes: Procedure code indices  
        :param diag_hypergraph: Diagnosis hypergraph Laplacian
        :param proc_hypergraph: Procedure hypergraph Laplacian
        """
        # Embed codes
        diag_emb = self.diag_embedding(diag_codes)
        proc_emb = self.proc_embedding(proc_codes)
        
        # Average pooling to get patient-level embeddings
        diag_patient_emb = torch.mean(diag_emb, dim=1)
        proc_patient_emb = torch.mean(proc_emb, dim=1)
        
        # Hypergraph processing for each modality
        diag_repr = self.diag_hgnn(diag_patient_emb, diag_hypergraph)
        proc_repr = self.proc_hgnn(proc_patient_emb, proc_hypergraph)
        
        # Fusion
        fused_repr = torch.cat([diag_repr, proc_repr], dim=-1)
        fused_repr = self.fusion_layer(fused_repr)
        
        # Classification
        logits = self.classifier(fused_repr)
        
        return logits

class HGNN_EHR_Temporal(nn.Module):
    """
    Temporal HGNN that incorporates time-aware hypergraph convolution
    """
    def __init__(self, in_features, hidden_dim, n_classes, max_seq_len=20,
                 time_dim=32, dropout=0.5):
        super(HGNN_EHR_Temporal, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Temporal hypergraph layers
        self.temporal_hgnn1 = TemporalHGNN_conv(hidden_dim, hidden_dim, time_dim)
        self.temporal_hgnn2 = TemporalHGNN_conv(hidden_dim, hidden_dim, time_dim)
        
        # Temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, G, time_stamps):
        """
        Temporal-aware forward pass
        :param x: Sequential features (batch_size, seq_len, in_features)
        :param G: Hypergraph Laplacian
        :param time_stamps: Temporal stamps (batch_size, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for hypergraph processing
        x = x.view(-1, x.size(-1))  # (batch_size * seq_len, hidden_dim)
        time_stamps = time_stamps.view(-1, 1)  # (batch_size * seq_len, 1)
        
        # Temporal hypergraph convolutions
        x = self.temporal_hgnn1(x, G, time_stamps)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.temporal_hgnn2(x, G, time_stamps)
        x = F.relu(x)
        
        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # Temporal attention aggregation
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        attended_x, _ = self.temporal_attention(x, x, x)
        
        # Take the last time step or use global pooling
        final_repr = attended_x[-1]  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(final_repr)
        
        return logits

def create_model(model_type, model_config):
    """
    Factory function to create different HGNN models
    """
    if model_type == 'basic':
        return HGNN_EHR(**model_config)
    elif model_type == 'advanced':
        return HGNN_EHR_Advanced(**model_config)
    elif model_type == 'hierarchical':
        return HGNN_EHR_Hierarchical(**model_config)
    elif model_type == 'multimodal':
        return HGNN_EHR_MultiModal(**model_config)
    elif model_type == 'temporal':
        return HGNN_EHR_Temporal(**model_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Model configurations
MODEL_CONFIGS = {
    'basic': {
        'hidden_dim': 256,
        'dropout': 0.5,
        'num_layers': 3
    },
    'advanced': {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.3,
        'use_temporal': False
    },
    'hierarchical': {
        'embedding_dim': 128,
        'visit_dim': 256,
        'patient_dim': 512,
        'hidden_dim': 256,
        'dropout': 0.5
    },
    'multimodal': {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'dropout': 0.5
    },
    'temporal': {
        'hidden_dim': 256,
        'time_dim': 64,
        'max_seq_len': 20,
        'dropout': 0.4
    }
}