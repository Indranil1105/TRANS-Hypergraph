# ehr_data_processor.py - UPDATED VERSION WITH BETTER DATA HANDLING

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from hypergraph_utils import HypergraphUtils
from icd9_ccs_mapping import icd9_to_ccs


class EHRDataProcessor:
    """Process EHR data from MIMIC into hypergraph-compatible format"""
    
    def __init__(self, min_visits=2, max_visits=20, min_codes_per_visit=1):
        self.min_visits = min_visits
        self.max_visits = max_visits
        self.min_codes_per_visit = min_codes_per_visit  # NEW: Ensure each visit has codes
        self.diag_encoder = LabelEncoder()
        self.proc_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.diag_vocab = []
        self.proc_vocab = []
    
    def load_mimic_data(self, data_path=None):
        """Load MIMIC data from pickle or CSV files"""
        if data_path and data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Load from CSV files
            admissions = pd.read_csv('ADMISSIONS.csv')
            patients = pd.read_csv('PATIENTS.csv')
            diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
            procedures = pd.read_csv('PROCEDURES_ICD.csv')
            return self.preprocess_raw_data(admissions, patients, diagnoses, procedures)
    
    def preprocess_raw_data(self, admissions, patients, diagnoses, procedures):
        """Preprocess raw MIMIC data"""
        # Clean admissions
        admissions = admissions.dropna(subset=['subject_id', 'hadm_id'])
        admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
        admissions = admissions.sort_values(['subject_id', 'admittime'])
        
        # Clean diagnoses and procedures
        diagnoses = diagnoses.dropna(subset=['icd9_code'])
        procedures = procedures.dropna(subset=['icd9_code'])
        
        # Build patient visits
        patient_visits = {}
        
        for _, adm in admissions.iterrows():
            pid, hid = adm['subject_id'], adm['hadm_id']
            patient_visits.setdefault(pid, [])
            
            # Get diagnosis and procedure codes for this admission
            diag_codes = diagnoses[diagnoses['hadm_id'] == hid]['icd9_code'].astype(str).tolist()
            proc_codes = procedures[procedures['hadm_id'] == hid]['icd9_code'].astype(str).tolist()
            
            if diag_codes or proc_codes:
                patient_visits[pid].append({
                    'hadm_id': hid,
                    'admittime': adm['admittime'],
                    'diagnosis_codes': diag_codes,
                    'procedure_codes': proc_codes
                })
        
        # Filter: Keep patients with min_visits to max_visits
        filtered_visits = {
            pid: v for pid, v in patient_visits.items()
            if self.min_visits <= len(v) <= self.max_visits
        }
        
        print(f"[DATA] Loaded {len(filtered_visits)} patients with {self.min_visits}-{self.max_visits} visits")
        
        return filtered_visits
    
    def create_vocabularies(self, patient_visits):
        """Create CCS vocabulary from diagnosis codes"""
        # Convert all diagnoses to CCS categories
        all_diags = [code for visits in patient_visits.values() 
                     for visit in visits for code in visit['diagnosis_codes']]
        
        all_ccs = [icd9_to_ccs(code) for code in all_diags]
        all_ccs = [c for c in all_ccs if c != 999]  # Remove unmapped
        
        self.diag_vocab = sorted(list(set(all_ccs)))
        
        print(f"\n[CCS] Converted {len(all_diags)} diagnosis codes")
        print(f"[CCS] Got {len(all_ccs)} valid CCS codes")
        print(f"[CCS] Vocabulary size: {len(self.diag_vocab)} unique CCS categories")
    
    def extract_features(self, patient_visits, feature_type='statistical'):
        """Extract features and labels from patient visits"""
        if not self.diag_vocab:
            self.create_vocabularies(patient_visits)
        
        features = []
        labels = []
        patient_ids = []
        
        for pid, visits in patient_visits.items():
            if len(visits) < 2:
                continue  # Need at least 2 visits
            
            # Historical visits (all but last)
            history_visits = visits[:-1]
            # Target visit (last visit)
            target_visit = visits[-1]
            
            # Extract statistical features from history
            num_visits = len(history_visits)
            total_diags = sum(len(v['diagnosis_codes']) for v in history_visits)
            total_procs = sum(len(v['procedure_codes']) for v in history_visits)
            avg_diags = total_diags / num_visits if num_visits > 0 else 0
            avg_procs = total_procs / num_visits if num_visits > 0 else 0
            unique_diags = len({code for v in history_visits for code in v['diagnosis_codes']})
            unique_procs = len({code for v in history_visits for code in v['procedure_codes']})
            
            # Feature vector
            feature_vector = np.array([
                num_visits,
                total_diags,
                total_procs,
                avg_diags,
                avg_procs,
                unique_diags,
                unique_procs
            ], dtype=np.float32)
            
            # Create label vector (binary multi-label)
            label_vector = np.zeros(len(self.diag_vocab), dtype=np.float32)
            
            # Get target diagnoses and convert to CCS
            target_diags = target_visit['diagnosis_codes']
            target_ccs = [icd9_to_ccs(code) for code in target_diags]
            target_ccs = [c for c in target_ccs if c != 999]  # Remove unmapped
            
            # Set labels to 1 for present codes
            for ccs_code in target_ccs:
                if ccs_code in self.diag_vocab:
                    idx = self.diag_vocab.index(ccs_code)
                    label_vector[idx] = 1.0
            
            features.append(feature_vector)
            labels.append(label_vector)
            patient_ids.append(pid)
        
        return np.array(features), np.array(labels), patient_ids
    
    def construct_patient_hypergraphs(self, X, k_neig=10, hypergraph_type='euclidean'):
        """Construct hypergraph for feature space"""
        if hypergraph_type == 'euclidean':
            H = HypergraphUtils.construct_H_with_KNN(X, k_neig=k_neig, distance_type='euclidean')
        else:
            H = HypergraphUtils.construct_H_with_KNN(X, k_neig=k_neig, distance_type='cosine')
        
        # Generate normalized Laplacian
        G = HypergraphUtils.generate_G_from_H(H)
        
        return H, G
    
    def create_datasets(self, patient_visits, test_size=0.15, val_size=0.15, feature_type='statistical'):
        """Create train/val/test datasets"""
        print("\n[DATASET CREATION]")
        
        # Extract features and labels
        X, y, pids = self.extract_features(patient_visits, feature_type)
        
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Label dimension: {y.shape[1]}")
        print(f"Total positive labels: {np.sum(y):.0f}")
        print(f"Avg labels per sample: {np.mean(np.sum(y, axis=1)):.2f}")
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Split into train/test
        X_temp, X_test, y_temp, y_test, pids_temp, pids_test = train_test_split(
            X, y, pids, test_size=test_size, random_state=42
        )
        
        # Split train into train/val
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, pids_train, pids_val = train_test_split(
            X_temp, y_temp, pids_temp, test_size=val_ratio, random_state=42
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Construct hypergraphs for each split
        H_train, G_train = self.construct_patient_hypergraphs(X_train, k_neig=10)
        H_val, G_val = self.construct_patient_hypergraphs(X_val, k_neig=10)
        H_test, G_test = self.construct_patient_hypergraphs(X_test, k_neig=10)
        
        return {
            'train': {
                'features': X_train,
                'labels': y_train,
                'hypergraph_H': H_train,
                'hypergraph_G': G_train,
                'patient_ids': pids_train
            },
            'val': {
                'features': X_val,
                'labels': y_val,
                'hypergraph_H': H_val,
                'hypergraph_G': G_val,
                'patient_ids': pids_val
            },
            'test': {
                'features': X_test,
                'labels': y_test,
                'hypergraph_H': H_test,
                'hypergraph_G': G_test,
                'patient_ids': pids_test
            },
            'vocab': {
                'diag': self.diag_vocab,
                'proc': self.proc_vocab
            }
        }


class EHRHypergraphDataset(Dataset):
    """PyTorch Dataset for EHR hypergraph data"""
    
    def __init__(self, features, labels, hypergraph_G):
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(labels)
        self.G = torch.FloatTensor(hypergraph_G)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'labels': self.y[idx],
            'hypergraph_G': self.G
        }


def create_data_loaders(datasets, batch_size=32):
    """Create DataLoaders for train/val/test"""
    train_dataset = EHRHypergraphDataset(
        datasets['train']['features'],
        datasets['train']['labels'],
        datasets['train']['hypergraph_G']
    )
    
    val_dataset = EHRHypergraphDataset(
        datasets['val']['features'],
        datasets['val']['labels'],
        datasets['val']['hypergraph_G']
    )
    
    test_dataset = EHRHypergraphDataset(
        datasets['test']['features'],
        datasets['test']['labels'],
        datasets['test']['hypergraph_G']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader