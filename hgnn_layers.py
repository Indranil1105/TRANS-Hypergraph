# HGNN Layers for EHR Prediction
# Hypergraph Neural Network layers adapted from original HGNN implementation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    """
    Hypergraph Convolution Layer
    Implements the core hypergraph convolution operation
    """
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        """
        Forward pass of hypergraph convolution
        :param x: Input node features (N x in_ft)
        :param G: Normalized hypergraph Laplacian (N x N)
        :return: Output node features (N x out_ft)
        """
        # Apply linear transformation
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        
        # Apply hypergraph convolution
        x = torch.matmul(G, x)
        return x

class HGNN_fc(nn.Module):
    """
    Simple fully connected layer for HGNN
    """
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        
    def forward(self, x):
        return self.fc(x)

class HGNN_embedding(nn.Module):
    """
    HGNN embedding layer with two hypergraph convolutions
    """
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        
    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))
        return x

class HGNN_classifier(nn.Module):
    """
    Classification head for HGNN
    """
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)
        
    def forward(self, x):
        x = self.fc1(x)
        return x

class MultiHeadHGNN_conv(nn.Module):
    """
    Multi-head hypergraph convolution layer
    """
    def __init__(self, in_ft, out_ft, num_heads=4, bias=True, dropout=0.1):
        super(MultiHeadHGNN_conv, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.num_heads = num_heads
        self.head_dim = out_ft // num_heads
        
        assert out_ft % num_heads == 0, "out_ft must be divisible by num_heads"
        
        self.W_q = nn.Linear(in_ft, out_ft, bias=False)
        self.W_k = nn.Linear(in_ft, out_ft, bias=False)  
        self.W_v = nn.Linear(in_ft, out_ft, bias=False)
        self.W_o = nn.Linear(out_ft, out_ft)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
            
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Multi-head attention computation
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention with hypergraph structure
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply hypergraph structure
        if G.dim() == 2:
            G = G.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        attention_scores = attention_scores * G
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.out_ft)
        
        # Final linear transformation
        output = self.W_o(context)
        if self.bias is not None:
            output = output + self.bias
            
        return output

class TemporalHGNN_conv(nn.Module):
    """
    Temporal-aware hypergraph convolution for sequential EHR data
    """
    def __init__(self, in_ft, out_ft, time_dim=32, bias=True):
        super(TemporalHGNN_conv, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.time_dim = time_dim
        
        # Spatial transformation
        self.weight_spatial = Parameter(torch.Tensor(in_ft, out_ft))
        
        # Temporal transformation
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, out_ft)
        )
        
        # Combination layer
        self.combine = nn.Linear(out_ft * 2, out_ft)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_spatial.size(1))
        self.weight_spatial.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, G: torch.Tensor, time_stamps=None):
        """
        Forward pass with temporal information
        :param x: Input features
        :param G: Hypergraph Laplacian
        :param time_stamps: Temporal information
        """
        # Spatial convolution
        x_spatial = torch.matmul(x, self.weight_spatial)
        x_spatial = torch.matmul(G, x_spatial)
        
        # Temporal encoding
        if time_stamps is not None:
            if time_stamps.dim() == 1:
                time_stamps = time_stamps.unsqueeze(-1)
            x_temporal = self.time_encoder(time_stamps.float())
            
            # Combine spatial and temporal features
            x_combined = torch.cat([x_spatial, x_temporal], dim=-1)
            x = self.combine(x_combined)
        else:
            x = x_spatial
            
        if self.bias is not None:
            x = x + self.bias
            
        return x

class HypergraphAttentionLayer(nn.Module):
    """
    Hypergraph attention layer for learning edge importance
    """
    def __init__(self, in_features, out_features, num_hyperedges, dropout=0.1):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hyperedges = num_hyperedges
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Linear(out_features * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hyperedge_index):
        """
        :param x: Node features (N, in_features)
        :param hyperedge_index: Hyperedge connectivity (2, num_edges)
        """
        h = self.W(x)  # (N, out_features)
        
        # Compute attention scores for each hyperedge
        edge_h = []
        for i in range(self.num_hyperedges):
            # Get nodes in this hyperedge
            nodes_in_edge = hyperedge_index[1] == i
            if nodes_in_edge.sum() > 0:
                node_indices = hyperedge_index[0][nodes_in_edge]
                edge_features = h[node_indices]  # (num_nodes_in_edge, out_features)
                
                # Compute pairwise attention
                attention_scores = []
                for j in range(len(node_indices)):
                    for k in range(j+1, len(node_indices)):
                        pair_features = torch.cat([edge_features[j], edge_features[k]], dim=0)
                        score = self.attention(pair_features)
                        attention_scores.append(score)
                
                if attention_scores:
                    edge_attention = torch.stack(attention_scores).mean()
                    edge_h.append(edge_attention)
        
        return h, torch.stack(edge_h) if edge_h else torch.zeros(1)