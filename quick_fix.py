# Quick Fix Script for MIMIC Data Processing
# This script bypasses the sklearn version issue by recreating the data from scratch

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def quick_fix_data():
    """
    Quick fix to recreate the processed data without sklearn version issues
    """
    print("Quick Fix: Processing MIMIC data from scratch...")
    
    # Load CSV files
    try:
        admissions = pd.read_csv('ADMISSIONS.csv')
        patients = pd.read_csv('PATIENTS.csv')
        diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
        procedures = pd.read_csv('PROCEDURES_ICD.csv')
        print("✓ Loaded CSV files successfully")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # Clean data
    admissions = admissions.dropna(subset=['subject_id', 'hadm_id'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
    admissions = admissions.sort_values(['subject_id', 'admittime'])
    
    diagnoses = diagnoses.dropna(subset=['icd9_code'])
    procedures = procedures.dropna(subset=['icd9_code'])
    
    print("✓ Cleaned data")
    
    # Create patient visits
    patient_visits = {}
    
    for _, adm in admissions.iterrows():
        subject_id = adm['subject_id']
        hadm_id = adm['hadm_id']
        
        if subject_id not in patient_visits:
            patient_visits[subject_id] = []
        
        # Get codes for this visit
        visit_diagnoses = diagnoses[diagnoses['hadm_id'] == hadm_id]['icd9_code'].astype(str).tolist()
        visit_procedures = procedures[procedures['hadm_id'] == hadm_id]['icd9_code'].astype(str).tolist()
        
        visit_info = {
            'hadm_id': hadm_id,
            'admittime': adm['admittime'],
            'diagnoses': visit_diagnoses,
            'procedures': visit_procedures,
            'subject_id': subject_id
        }
        
        patient_visits[subject_id].append(visit_info)
    
    # Filter patients with multiple visits
    filtered_patients = {}
    for patient_id, visits in patient_visits.items():
        if len(visits) >= 2:
            filtered_patients[patient_id] = visits
    
    print(f"✓ Created patient visits for {len(filtered_patients)} patients")
    
    # Create vocabularies
    all_diagnoses = set()
    all_procedures = set()
    
    for visits in filtered_patients.values():
        for visit in visits:
            all_diagnoses.update(visit['diagnoses'])
            all_procedures.update(visit['procedures'])
    
    diag_vocab = ['<PAD>', '<UNK>'] + sorted(list(all_diagnoses))
    proc_vocab = ['<PAD>', '<UNK>'] + sorted(list(all_procedures))
    
    print(f"✓ Created vocabularies - Diagnoses: {len(diag_vocab)}, Procedures: {len(proc_vocab)}")
    
    # Save the processed data
    processed_data = {
        'patient_visits': filtered_patients,
        'diag_vocab': diag_vocab,
        'proc_vocab': proc_vocab,
        'data_stats': {
            'n_patients': len(filtered_patients),
            'n_diagnoses': len(diag_vocab),
            'n_procedures': len(proc_vocab),
            'total_visits': sum(len(visits) for visits in filtered_patients.values())
        }
    }
    
    with open('processed_mimic_data_fixed.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("✓ Saved processed data to 'processed_mimic_data_fixed.pkl'")
    print(f"✓ Statistics: {processed_data['data_stats']}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = quick_fix_data()
    print("\nQuick fix completed! Now you can run the demo with:")
    print("python demo_simple.py")