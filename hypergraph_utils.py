# Hypergraph Utilities for MIMIC Data Processing
# Based on HGNN implementation with modifications for EHR data

import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class HypergraphUtils:
    """Utility functions for hypergraph construction and operations"""
    
    @staticmethod
    def euclidean_distance(x):
        """
        Calculate the distance among each row of x
        :param x: N X D tensor/array
        :return: N X N distance matrix
        """
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        x = np.array(x)
        
        # Compute pairwise euclidean distances
        distances = euclidean_distances(x, x)
        return distances
    
    @staticmethod
    def cosine_distance(x):
        """
        Calculate cosine distance matrix
        :param x: N X D tensor/array
        :return: N X N distance matrix
        """
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # Compute cosine similarity then convert to distance
        similarity = cosine_similarity(x, x)
        distance = 1 - similarity
        return distance
    
    @staticmethod
    def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
        """
        Construct hypergraph incidence matrix from distance matrix
        :param dis_mat: node distance matrix
        :param k_neig: K nearest neighbor
        :param is_probH: probabilistic hyperedge weights or binary
        :param m_prob: probability parameter
        :return: N_object X N_hyperedge incidence matrix
        """
        n_obj = dis_mat.shape[0]
        n_edge = n_obj  # Each node creates one hyperedge
        H = np.zeros((n_obj, n_edge))
        
        for center_idx in range(n_obj):
            # Set self-distance to 0
            dis_vec = dis_mat[center_idx].copy()
            dis_vec[center_idx] = 0
            
            # Find k nearest neighbors
            nearest_idx = np.argsort(dis_vec)[:k_neig]
            avg_dis = np.mean(dis_vec[dis_vec > 0])
            
            if avg_dis == 0 or np.isnan(avg_dis):
                avg_dis = 1.0
                
            # Include center node if not in k-nearest
            if center_idx not in nearest_idx:
                nearest_idx = np.append(nearest_idx[:-1], center_idx)
                
            # Set hyperedge weights
            for node_idx in nearest_idx:
                if is_probH and avg_dis > 0:
                    weight = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
                    H[node_idx, center_idx] = max(weight, 1e-8)  # Avoid zero weights
                else:
                    H[node_idx, center_idx] = 1.0
                    
        return H
    
    @staticmethod
    def construct_H_with_KNN(X, k_neig=10, is_probH=True, m_prob=1, distance_type='euclidean'):
        """
        Initialize hypergraph incidence matrix from feature matrix
        :param X: N_object x feature_number
        :param k_neig: number of nearest neighbors
        :param is_probH: use probabilistic weights
        :param m_prob: probability parameter
        :param distance_type: 'euclidean' or 'cosine'
        :return: N_object x N_hyperedge incidence matrix
        """
        if len(X.shape) != 2:
            X = X.reshape(-1, X.shape[-1])
        
        # Choose distance metric
        if distance_type == 'euclidean':
            dis_mat = HypergraphUtils.euclidean_distance(X)
        elif distance_type == 'cosine':
            dis_mat = HypergraphUtils.cosine_distance(X)
        else:
            raise ValueError("distance_type must be 'euclidean' or 'cosine'")
            
        H = HypergraphUtils.construct_H_with_KNN_from_distance(
            dis_mat, k_neig, is_probH, m_prob)
        return H
    
    @staticmethod
    def construct_H_medical_cooccurrence(visit_codes_list, vocab_size, min_cooccur=2):
        """
        Construct hypergraph based on medical code co-occurrence patterns
        :param visit_codes_list: List of lists, each sublist contains codes for a visit
        :param vocab_size: Total vocabulary size
        :param min_cooccur: Minimum co-occurrence threshold
        :return: vocab_size x N_hyperedge incidence matrix
        """
        # Count co-occurrences
        cooccurrence_count = {}
        hyperedge_id = 0
        
        for visit_codes in visit_codes_list:
            if len(visit_codes) >= 2:  # Need at least 2 codes for co-occurrence
                # Create hyperedge from all codes in this visit
                for i in range(len(visit_codes)):
                    for j in range(i + 1, len(visit_codes)):
                        code_pair = tuple(sorted([visit_codes[i], visit_codes[j]]))
                        if code_pair not in cooccurrence_count:
                            cooccurrence_count[code_pair] = 0
                        cooccurrence_count[code_pair] += 1
        
        # Create hyperedges from frequent co-occurrences
        hyperedges = []
        for code_pair, count in cooccurrence_count.items():
            if count >= min_cooccur:
                hyperedges.append(code_pair)
        
        # Build incidence matrix
        n_edges = len(hyperedges)
        H = np.zeros((vocab_size, n_edges))
        
        for edge_idx, code_pair in enumerate(hyperedges):
            for code in code_pair:
                if code < vocab_size:  # Safety check
                    H[code, edge_idx] = 1.0
                    
        return H
    
    @staticmethod
    def generate_G_from_H(H, variable_weight=False):
        """
        Calculate normalized hypergraph Laplacian from incidence matrix
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether to use variable edge weights
        :return: Normalized hypergraph Laplacian G
        """
        if torch.is_tensor(H):
            H = H.cpu().numpy()
        
        H = np.array(H, dtype=np.float32)
        n_edge = H.shape[1]
        
        if n_edge == 0:
            return np.eye(H.shape[0], dtype=np.float32)
        
        # Edge weights
        W = np.ones(n_edge, dtype=np.float32)
        
        # Node degrees
        DV = np.sum(H * W, axis=1)
        # Edge degrees  
        DE = np.sum(H, axis=0)
        
        # Handle zero degrees
        DV = np.maximum(DV, 1e-8)
        DE = np.maximum(DE, 1e-8)
        
        # Compute normalized Laplacian
        invDE = np.diag(1.0 / DE)
        DV_sqrt = np.diag(1.0 / np.sqrt(DV))
        W_diag = np.diag(W)
        
        # G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        G = DV_sqrt @ H @ W_diag @ invDE @ H.T @ DV_sqrt
        
        return G.astype(np.float32)
    
    @staticmethod
    def hyperedge_concat(*H_list):
        """
        Concatenate multiple hypergraph incidence matrices
        :param H_list: List of hypergraph incidence matrices
        :return: Concatenated hypergraph incidence matrix
        """
        H_combined = None
        
        for H in H_list:
            if H is not None and H.size > 0:
                if H_combined is None:
                    H_combined = H
                else:
                    H_combined = np.hstack([H_combined, H])
                    
        return H_combined if H_combined is not None else np.array([])

    @staticmethod  
    def multi_modal_hypergraph(features_list, k_neig_list=None, distance_types=None):
        """
        Construct multi-modal hypergraph from multiple feature modalities
        :param features_list: List of feature matrices for different modalities
        :param k_neig_list: List of k values for each modality
        :param distance_types: List of distance types for each modality
        :return: Combined hypergraph incidence matrix
        """
        if k_neig_list is None:
            k_neig_list = [10] * len(features_list)
        if distance_types is None:
            distance_types = ['euclidean'] * len(features_list)
            
        H_list = []
        
        for i, (features, k_neig, dist_type) in enumerate(zip(features_list, k_neig_list, distance_types)):
            if features is not None and features.size > 0:
                H_i = HypergraphUtils.construct_H_with_KNN(
                    features, k_neig=k_neig, distance_type=dist_type)
                H_list.append(H_i)
        
        # Concatenate all hypergraph matrices
        if H_list:
            return HypergraphUtils.hyperedge_concat(*H_list)
        else:
            return np.array([])