# mimic4_data_loader.py - MIMIC-IV CSV Data Loader & Processor

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from hypergraph_utils import HypergraphUtils
from icd9_ccs_mapping import icd9_to_ccs

class MIMIC4DataLoader:
    """
    Load and process MIMIC-IV data from CSV files
    Works with:
    - ADMISSIONS.csv
    - DIAGNOSES_ICD.csv  
    - PROCEDURES_ICD.csv
    - PATIENTS.csv
    - PRESCRIPTIONS.csv (optional)
    - ICU_STAY.csv (optional)
    """
    
    def __init__(self, min_visits=2, max_visits=20):
        self.min_visits = min_visits
        self.max_visits = max_visits
        self.scaler = StandardScaler()
        self.diag_vocab = []
        self.proc_vocab = []
    
    def load_from_csv(self, admissions_path, diagnoses_path, procedures_path, patients_path):
        """Load MIMIC-IV data from CSV files"""
        print("\n[LOADING MIMIC-IV CSV FILES]")
        
        # Load CSV files
        print(f"Loading ADMISSIONS from {admissions_path}...")
        admissions = pd.read_csv(admissions_path)
        
        print(f"Loading DIAGNOSES_ICD from {diagnoses_path}...")
        diagnoses = pd.read_csv(diagnoses_path)
        
        print(f"Loading PROCEDURES_ICD from {procedures_path}...")
        procedures = pd.read_csv(procedures_path)
        
        print(f"Loading PATIENTS from {patients_path}...")
        patients = pd.read_csv(patients_path)
        
        print(f"\nDataset sizes:")
        print(f"  Admissions: {len(admissions)} records")
        print(f"  Diagnoses: {len(diagnoses)} records")
        print(f"  Procedures: {len(procedures)} records")
        print(f"  Patients: {len(patients)} records")
        
        return admissions, diagnoses, procedures, patients
    
    def process_mimic4(self, admissions, diagnoses, procedures, patients):
        """
        Process MIMIC-IV data into patient visits structure
        
        Returns: {patient_id: [{visit_1}, {visit_2}, ...]}
        """
        print("\n[PROCESSING MIMIC-IV DATA]")
        
        # Clean data
        admissions = admissions.dropna(subset=['subject_id', 'hadm_id'])
        admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
        admissions = admissions.sort_values(['subject_id', 'admittime'])
        
        diagnoses = diagnoses.dropna(subset=['icd_code'])
        procedures = procedures.dropna(subset=['icd_code'])
        
        print(f"✓ Data cleaned and sorted")
        print(f"  Unique patients: {admissions['subject_id'].nunique()}")
        print(f"  Unique admissions: {len(admissions)}")
        
        # Build patient visits
        patient_visits = {}
        diag_summary = defaultdict(int)
        proc_summary = defaultdict(int)
        
        for _, adm in admissions.iterrows():
            pid = int(adm['subject_id'])
            hadm_id = int(adm['hadm_id'])
            admit_time = adm['admittime']
            
            if pid not in patient_visits:
                patient_visits[pid] = []
            
            # Get diagnosis codes for this admission
            diag_codes = diagnoses[diagnoses['hadm_id'] == hadm_id]['icd_code'].astype(str).tolist()
            diag_codes = [c.strip() for c in diag_codes if pd.notna(c)]
            
            # Get procedure codes for this admission
            proc_codes = procedures[procedures['hadm_id'] == hadm_id]['icd_code'].astype(str).tolist()
            proc_codes = [c.strip() for c in proc_codes if pd.notna(c)]
            
            # Count codes
            for dc in diag_codes:
                diag_summary[dc] += 1
            for pc in proc_codes:
                proc_summary[pc] += 1
            
            # Store visit (only if has codes)
            if diag_codes or proc_codes:
                patient_visits[pid].append({
                    'hadm_id': hadm_id,
                    'admittime': admit_time,
                    'diagnosis_codes': diag_codes,
                    'procedure_codes': proc_codes,
                    'num_diagnoses': len(diag_codes),
                    'num_procedures': len(proc_codes)
                })
        
        # Filter: Keep only patients with min_visits to max_visits
        filtered_visits = {
            pid: visits for pid, visits in patient_visits.items()
            if self.min_visits <= len(visits) <= self.max_visits
        }
        
        print(f"\n✓ Patient visits constructed")
        print(f"  Total patients before filtering: {len(patient_visits)}")
        print(f"  Patients with {self.min_visits}-{self.max_visits} visits: {len(filtered_visits)}")
        print(f"  Total visits: {sum(len(v) for v in filtered_visits.values())}")
        print(f"  Avg visits per patient: {np.mean([len(v) for v in filtered_visits.values()]):.2f}")
        
        # Statistics
        total_diags = sum(len(v.get('diagnosis_codes', [])) for visits in filtered_visits.values() for v in visits)
        total_procs = sum(len(v.get('procedure_codes', [])) for visits in filtered_visits.values() for v in visits)
        avg_diags_per_visit = total_diags / sum(len(v) for v in filtered_visits.values())
        avg_procs_per_visit = total_procs / sum(len(v) for v in filtered_visits.values())
        
        print(f"\n[VISIT STATISTICS]")
        print(f"  Total diagnoses across all visits: {total_diags}")
        print(f"  Total procedures across all visits: {total_procs}")
        print(f"  Avg diagnoses per visit: {avg_diags_per_visit:.2f}")
        print(f"  Avg procedures per visit: {avg_procs_per_visit:.2f}")
        print(f"  Unique diagnosis codes: {len(diag_summary)}")
        print(f"  Unique procedure codes: {len(proc_summary)}")
        
        return filtered_visits
    
    def create_vocabularies(self, patient_visits):
        """Create CCS vocabulary from diagnosis codes"""
        print("\n[VOCABULARY CREATION]")
        
        # Collect all diagnoses
        all_diags = []
        for pid, visits in patient_visits.items():
            for visit in visits:
                codes = visit.get('diagnosis_codes', [])
                all_diags.extend(codes)
        
        print(f"Total diagnosis code mentions: {len(all_diags)}")
        
        # Convert to CCS categories
        all_ccs = []
        unmapped = 0
        
        for code in all_diags:
            code_str = str(code).strip() if isinstance(code, str) else str(code)
            ccs = icd9_to_ccs(code_str)
            
            if ccs != 999:  # 999 = unmapped
                all_ccs.append(ccs)
            else:
                unmapped += 1
        
        self.diag_vocab = sorted(list(set(all_ccs)))
        
        print(f"✓ Converted diagnosis codes to CCS")
        print(f"  Valid CCS codes: {len(all_ccs)}")
        print(f"  Unmapped codes: {unmapped}")
        print(f"  Vocabulary size: {len(self.diag_vocab)} unique CCS categories")
        print(f"  CCS categories: {self.diag_vocab[:10]}... (first 10)")
    
    def extract_features(self, patient_visits, feature_type='statistical'):
        """Extract features and labels from patient visits"""
        print("\n[FEATURE EXTRACTION]")
        
        if not self.diag_vocab:
            self.create_vocabularies(patient_visits)
        
        features = []
        labels = []
        patient_ids = []
        valid_patients = 0
        skipped_patients = 0
        
        for patient_id, visits in patient_visits.items():
            if len(visits) < 2:
                skipped_patients += 1
                continue
            
            # Historical visits (all but last)
            history_visits = visits[:-1]
            # Target visit (last visit)
            target_visit = visits[-1]
            
            # Extract statistical features from history
            num_visits = len(history_visits)
            total_diags = sum(v.get('num_diagnoses', 0) for v in history_visits)
            total_procs = sum(v.get('num_procedures', 0) for v in history_visits)
            
            unique_diags = len(set(code for v in history_visits for code in v.get('diagnosis_codes', [])))
            unique_procs = len(set(code for v in history_visits for code in v.get('procedure_codes', [])))
            
            avg_diags = total_diags / num_visits if num_visits > 0 else 0
            avg_procs = total_procs / num_visits if num_visits > 0 else 0
            
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
            
            # Get target diagnoses
            target_diags = target_visit.get('diagnosis_codes', [])
            
            # Convert to CCS
            target_ccs = []
            for code in target_diags:
                code_str = str(code).strip()
                ccs = icd9_to_ccs(code_str)
                if ccs != 999:
                    target_ccs.append(ccs)
            
            # Set labels
            for ccs_code in target_ccs:
                if ccs_code in self.diag_vocab:
                    idx = self.diag_vocab.index(ccs_code)
                    label_vector[idx] = 1.0
            
            features.append(feature_vector)
            labels.append(label_vector)
            patient_ids.append(patient_id)
            valid_patients += 1
        
        print(f"✓ Extracted features")
        print(f"  Valid patients: {valid_patients}")
        print(f"  Skipped patients: {skipped_patients}")
        print(f"  Feature dimension: {features[0].shape if features else 0}")
        print(f"  Label dimension: {len(self.diag_vocab)}")
        
        return np.array(features), np.array(labels), patient_ids
    
    def construct_hypergraphs(self, X, k_neig=10):
        """Construct hypergraph for feature space"""
        print(f"Constructing hypergraph with k_neig={k_neig}...")
        
        if len(X) < k_neig:
            k_neig = max(1, len(X) // 2)
            print(f"⚠ Adjusted k_neig to {k_neig}")
        
        H = HypergraphUtils.construct_H_with_KNN(X, k_neig=k_neig, distance_type='euclidean')
        G = HypergraphUtils.generate_G_from_H(H)
        
        return H, G
    
    def create_datasets(self, admissions_path, diagnoses_path, procedures_path, patients_path,
                       test_size=0.15, val_size=0.15):
        """Create train/val/test datasets from MIMIC-IV CSV files"""
        
        # Step 1: Load CSV files
        admissions, diagnoses, procedures, patients = self.load_from_csv(
            admissions_path, diagnoses_path, procedures_path, patients_path
        )
        
        # Step 2: Process into patient visits
        patient_visits = self.process_mimic4(admissions, diagnoses, procedures, patients)
        
        # Step 3: Extract features and labels
        X, y, pids = self.extract_features(patient_visits)
        
        if len(X) == 0:
            raise ValueError("❌ No valid features extracted!")
        
        print(f"\n[DATASET STATISTICS]")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Label dimension: {y.shape[1]}")
        print(f"Total positive labels: {np.sum(y):.0f}")
        print(f"Avg labels per sample: {np.mean(np.sum(y, axis=1)):.2f}")
        print(f"Label sparsity: {(y.size - np.count_nonzero(y)) / y.size:.2%}")
        
        # Step 4: Standardize features
        print(f"\n[STANDARDIZATION]")
        X = self.scaler.fit_transform(X)
        print(f"✓ Features standardized")
        
        # Step 5: Split data
        print(f"\n[DATA SPLITTING]")
        X_temp, X_test, y_temp, y_test, pids_temp, pids_test = train_test_split(
            X, y, pids, test_size=test_size, random_state=42
        )
        
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, pids_train, pids_val = train_test_split(
            X_temp, y_temp, pids_temp, test_size=val_ratio, random_state=42
        )
        
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Val set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Step 6: Construct hypergraphs
        print(f"\n[HYPERGRAPH CONSTRUCTION]")
        H_train, G_train = self.construct_hypergraphs(X_train, k_neig=min(10, len(X_train)//2))
        H_val, G_val = self.construct_hypergraphs(X_val, k_neig=min(10, max(1, len(X_val)//2)))
        H_test, G_test = self.construct_hypergraphs(X_test, k_neig=min(10, max(1, len(X_test)//2)))
        print(f"✓ Hypergraphs constructed")
        
        print("\n" + "="*80 + "\n")
        
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
                'diag': self.diag_vocab
            }
        }
    
    def save_processed_data(self, datasets, save_path='mimic4_processed.pkl'):
        """Save processed datasets to pickle"""
        with open(save_path, 'wb') as f:
            pickle.dump(datasets, f)
        print(f"✓ Saved processed data to {save_path}")


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


def create_mimic4_datasets(admissions_path, diagnoses_path, procedures_path, patients_path,
                          min_visits=2, max_visits=20, test_size=0.15, val_size=0.15):
    """
    One-function loader for MIMIC-IV data
    
    Usage:
    datasets = create_mimic4_datasets(
        'ADMISSIONS.csv',
        'DIAGNOSES_ICD.csv',
        'PROCEDURES_ICD.csv',
        'PATIENTS.csv'
    )
    """
    loader = MIMIC4DataLoader(min_visits=min_visits, max_visits=max_visits)
    
    datasets = loader.create_datasets(
        admissions_path, diagnoses_path, procedures_path, patients_path,
        test_size=test_size, val_size=val_size
    )
    
    return datasets


if __name__ == '__main__':
    # Example usage
    print("MIMIC-IV Data Loader Demo")
    print("="*80)
    
    # Load data
    datasets = create_mimic4_datasets(
        'ADMISSIONS.csv',
        'DIAGNOSES_ICD.csv',
        'PROCEDURES_ICD.csv',
        'PATIENTS.csv',
        min_visits=2,
        max_visits=20
    )
    
    print("Datasets created successfully!")
    print(f"Train samples: {len(datasets['train']['features'])}")
    print(f"Val samples: {len(datasets['val']['features'])}")
    print(f"Test samples: {len(datasets['test']['features'])}")