# ehr_data_processor.py

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from hypergraph_utils import HypergraphUtils

class EHRDataProcessor:
    def __init__(self, min_visits=2, max_visits=20):
        self.min_visits = min_visits
        self.max_visits = max_visits
        self.diag_encoder = LabelEncoder()
        self.proc_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.diag_vocab = []
        self.proc_vocab = []

    def load_mimic_data(self, data_path=None):
        if data_path and data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        else:
            admissions = pd.read_csv('ADMISSIONS.csv')
            patients   = pd.read_csv('PATIENTS.csv')
            diagnoses  = pd.read_csv('DIAGNOSES_ICD.csv')
            procedures = pd.read_csv('PROCEDURES_ICD.csv')
            return self.preprocess_raw_data(admissions, patients, diagnoses, procedures)

    def preprocess_raw_data(self, admissions, patients, diagnoses, procedures):
        admissions = admissions.dropna(subset=['subject_id','hadm_id'])
        admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
        admissions = admissions.sort_values(['subject_id','admittime'])

        diagnoses = diagnoses.dropna(subset=['icd9_code'])
        procedures= procedures.dropna(subset=['icd9_code'])

        patient_visits = {}
        for _, adm in admissions.iterrows():
            pid, hid = adm['subject_id'], adm['hadm_id']
            patient_visits.setdefault(pid, [])
            diag_codes = diagnoses[diagnoses['hadm_id']==hid]['icd9_code'].astype(str).tolist()
            proc_codes = procedures[procedures['hadm_id']==hid]['icd9_code'].astype(str).tolist()
            if diag_codes or proc_codes:
                patient_visits[pid].append({
                    'hadm_id': hid,
                    'admittime': adm['admittime'],
                    'diagnoses': diag_codes,
                    'procedures': proc_codes
                })
        return {pid: v for pid, v in patient_visits.items()
                if self.min_visits <= len(v) <= self.max_visits}

    def create_vocabularies(self, patient_visits):
        all_diag, all_proc = set(), set()
        for visits in patient_visits.values():
            for visit in visits:
                all_diag.update(visit['diagnoses'])
                all_proc.update(visit['procedures'])
        self.diag_vocab = ['<PAD>','<UNK>'] + sorted(all_diag)
        self.proc_vocab = ['<PAD>','<UNK>'] + sorted(all_proc)
        self.diag_encoder.fit(self.diag_vocab)
        self.proc_encoder.fit(self.proc_vocab)
        return self.diag_vocab, self.proc_vocab

    def extract_features(self, patient_visits, feature_type='statistical'):
        if not self.diag_vocab:
            self.create_vocabularies(patient_visits)
        features, labels, pids = [], [], []
        for pid, visits in patient_visits.items():
            if len(visits)<2: continue
            fv, tv = visits[:-1], visits[-1]
            num_v = len(fv)
            tot_d = sum(len(v['diagnoses']) for v in fv)
            tot_p = sum(len(v['procedures']) for v in fv)
            avg_d = tot_d/num_v if num_v else 0
            avg_p = tot_p/num_v if num_v else 0
            uniq_d= len({d for v in fv for d in v['diagnoses']})
            uniq_p= len({p for v in fv for p in v['procedures']})
            feat = [num_v,tot_d,tot_p,avg_d,avg_p,uniq_d,uniq_p]
            tgt = np.zeros(len(self.diag_vocab))
            for d in tv['diagnoses']:
                if d in self.diag_vocab:
                    tgt[self.diag_vocab.index(d)] = 1
            features.append(feat); labels.append(tgt); pids.append(pid)
        return np.array(features), np.array(labels), pids

    def construct_patient_hypergraphs(self, X, k_neig=10, hypergraph_type='knn'):
        if hypergraph_type=='knn':
            H = HypergraphUtils.construct_H_with_KNN(X, k_neig=k_neig)
        else:
            H = HypergraphUtils.construct_H_with_KNN(X, k_neig=k_neig, distance_type='cosine')
        G = HypergraphUtils.generate_G_from_H(H)
        return H, G

    def create_datasets(self, patient_visits, test_size=0.2, val_size=0.1, feature_type='statistical'):
        X,y,_ = self.extract_features(patient_visits, feature_type)
        X = self.scaler.fit_transform(X)
        X_t, X_test, y_t, y_test = train_test_split(X,y,test_size=test_size,random_state=42)
        vs = val_size/(1-test_size)
        X_train,X_val,y_train,y_val = train_test_split(X_t,y_t,test_size=vs,random_state=42)
        H_tr,G_tr = self.construct_patient_hypergraphs(X_train)
        H_va,G_va = self.construct_patient_hypergraphs(X_val)
        H_te,G_te = self.construct_patient_hypergraphs(X_test)
        return {
            'train':{'features':X_train,'labels':y_train,'hypergraph_H':H_tr,'hypergraph_G':G_tr},
            'val':  {'features':X_val,  'labels':y_val,  'hypergraph_H':H_va,'hypergraph_G':G_va},
            'test': {'features':X_test, 'labels':y_test,'hypergraph_H':H_te,'hypergraph_G':G_te},
            'vocab':{'diag':self.diag_vocab,'proc':self.proc_vocab}
        }

class EHRHypergraphDataset(Dataset):
    def __init__(self, features, labels, hypergraph_H, hypergraph_G):
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(labels)
        self.G = torch.FloatTensor(hypergraph_G)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {'features':self.X[i],'labels':self.y[i],'hypergraph_G':self.G}

def create_data_loaders(datasets, batch_size=32):
    train_ds = EHRHypergraphDataset(**datasets['train'])
    val_ds   = EHRHypergraphDataset(**datasets['val'])
    test_ds  = EHRHypergraphDataset(**datasets['test'])
    return (DataLoader(train_ds,batch_size=batch_size,shuffle=True),
            DataLoader(val_ds,  batch_size=batch_size,shuffle=False),
            DataLoader(test_ds, batch_size=batch_size,shuffle=False))
