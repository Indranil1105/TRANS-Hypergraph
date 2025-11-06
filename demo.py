# demo_fixed.py - CORRECTED VERSION
# This fixes all the errors in your demo.py

import os
import sys
import pickle
import numpy as np
import torch
import warnings
from collections import Counter

warnings.filterwarnings('ignore', category=UserWarning)

from ehr_data_processor import EHRDataProcessor, create_data_loaders
from hgnn_models import create_model, MODEL_CONFIGS
from train_hgnn_o import FixedHGNNTrainer
from analyze_results import HGNNAnalyzer


def reduce_to_topk_classes(patient_visits, k=50):
    """Reduce to top-k most frequent diagnosis codes"""
    print("="*80)
    print(f"REDUCING TO TOP-{k} MOST FREQUENT CLASSES")
    print("="*80)

    # Count all diagnosis codes
    code_counter = Counter()
    for patient_id, visits in patient_visits.items():
        for visit in visits:
            if 'diagnoses' in visit:
                code_counter.update(visit['diagnoses'])

    # Get top-k codes
    topk_codes = [code for code, count in code_counter.most_common(k)]

    print(f"Total unique codes: {len(code_counter)}")
    print(f"Top-{k} codes selected: {len(topk_codes)}")
    coverage = sum(code_counter[c] for c in topk_codes) / sum(code_counter.values()) * 100
    print(f"Coverage: {coverage:.1f}% of all diagnosis occurrences")

    # Show top 10 most common codes (FIXED: only show top 10, not repeated)
    print("\nTop 10 most common diagnosis codes:")
    for code, count in list(code_counter.most_common(10)):
        print(f"  {code}: {count} occurrences")

    # Filter patient visits to only include top-k codes
    filtered_visits = {}
    for patient_id, visits in patient_visits.items():
        filtered_patient_visits = []
        for visit in visits:
            if 'diagnoses' in visit:
                # Keep only diagnoses in top-k
                filtered_diagnoses = [d for d in visit['diagnoses'] if d in topk_codes]
                if filtered_diagnoses:  # Only keep visits with at least one diagnosis
                    new_visit = visit.copy()
                    new_visit['diagnoses'] = filtered_diagnoses
                    filtered_patient_visits.append(new_visit)

        if len(filtered_patient_visits) >= 2:  # Keep patients with at least 2 visits
            filtered_visits[patient_id] = filtered_patient_visits

    print(f"\nOriginal patients: {len(patient_visits)}")
    print(f"Filtered patients: {len(filtered_visits)}")
    print(f"Retained: {len(filtered_visits)/len(patient_visits)*100:.1f}%")

    return filtered_visits, topk_codes


def run_demo(k_classes=50):
    """Run complete demo of HGNN on MIMIC data with top-K class reduction"""

    print("\n" + "="*70)
    print("HGNN MIMIC Demo - TOP-K CLASSES VERSION")
    print("="*70)
    print(f"\nUsing top-{k_classes} most frequent diagnosis codes")
    print("This dramatically improves learning signal\n")

    # Step 1: Data Processing
    print("Step 1: Data Processing")
    print("-" * 25)

    if os.path.exists('processed_mimic_data.pkl'):
        print("Loading preprocessed MIMIC data...")
        data_processor = EHRDataProcessor(min_visits=2, max_visits=20)

        with open('processed_mimic_data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        if isinstance(loaded_data, dict) and 'patient_visits' in loaded_data:
            patient_visits = loaded_data['patient_visits']
        else:
            patient_visits = loaded_data

        print(f"Loaded data for {len(patient_visits)} patients")
    else:
        print("Preprocessed data not found. Creating from CSV files...")
        data_processor = EHRDataProcessor(min_visits=2, max_visits=20)

        try:
            import pandas as pd
            admissions = pd.read_csv('ADMISSIONS.csv')
            patients = pd.read_csv('PATIENTS.csv')
            diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
            procedures = pd.read_csv('PROCEDURES_ICD.csv')

            patient_visits = data_processor.preprocess_raw_data(admissions, patients, diagnoses, procedures)
            print(f"Processed data for {len(patient_visits)} patients")

            with open('processed_mimic_data_new.pkl', 'wb') as f:
                pickle.dump({'patient_visits': patient_visits}, f)
            print("Saved processed data to 'processed_mimic_data_new.pkl'")
        except FileNotFoundError:
            print("ERROR: CSV files not found! Using sample data...")
            # Create minimal sample data for testing
            patient_visits = {
                1: [{'diagnoses': ['250', '401', '272'], 'procedures': []},
                    {'diagnoses': ['250', '272', '427'], 'procedures': []}],
                2: [{'diagnoses': ['401', '272'], 'procedures': []},
                    {'diagnoses': ['401', '427'], 'procedures': []}],
            }
            print(f"Created sample data for {len(patient_visits)} patients")

    # *** CRITICAL FIX: Reduce to top-K classes ***
    print("\n" + "="*80)
    patient_visits, topk_codes = reduce_to_topk_classes(patient_visits, k=k_classes)
    print("="*80 + "\n")

    # Check if we have enough data
    if len(patient_visits) < 2:
        print("ERROR: Not enough patients after filtering!")
        print("Try using k_classes=20 or load more MIMIC data")
        return None

    print("Creating datasets and constructing hypergraphs...")
    try:
        datasets = data_processor.create_datasets(
            patient_visits,
            feature_type='statistical',
            test_size=0.2,
            val_size=0.1
        )

        # FIXED: Get data sizes correctly
        n_train = datasets['train']['features'].shape[0]
        n_val = datasets['val']['features'].shape[0]
        n_test = datasets['test']['features'].shape[0]
        n_features = datasets['train']['features'].shape[1]
        n_classes = datasets['train']['labels'].shape[1]

        print(f"\nData Shape:")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Test samples: {n_test}")
        print(f"  Feature dimension: {n_features}")
        print(f"  Number of diagnosis classes: {n_classes}")
        print(f"  Class type: Top-{k_classes} most frequent codes")
        
        # Calculate sparsity
        avg_samples_per_class = (datasets['train']['labels'].sum(axis=0).mean())
        print(f"  Data sparsity: {n_classes} classes / {n_train} samples")
        print(f"  Avg samples per class: {avg_samples_per_class:.1f}")
        print("")

        # Save datasets
        with open('processed_datasets.pkl', 'wb') as f:
            pickle.dump(datasets, f)
        print("Saved datasets to 'processed_datasets.pkl'")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 2: Hypergraph Analysis
    print("\nStep 2: Hypergraph Analysis")
    print("-" * 27)

    try:
        analyzer = HGNNAnalyzer()
        H_train = datasets['train']['hypergraph_H']

        print("Analyzing hypergraph structure...")
        print(f"Hypergraph incidence matrix shape: {H_train.shape}")
        print(f"Number of nodes (patients): {H_train.shape[0]}")
        print(f"Number of hyperedges: {H_train.shape[1]}")
        
        if H_train.shape[0] > 0 and H_train.shape[1] > 0:
            density = np.sum(H_train) / (H_train.shape[0] * H_train.shape[1])
            print(f"Hypergraph density: {density:.4f}")

            node_degrees = np.sum(H_train, axis=1)
            edge_sizes = np.sum(H_train, axis=0)
            print(f"Average node degree: {np.mean(node_degrees):.2f}")
            print(f"Average hyperedge size: {np.mean(edge_sizes):.2f}")

    except Exception as e:
        print(f"Warning: Error in hypergraph analysis: {e}")

    # Step 3: Model Training
    print("\nStep 3: Model Training (Demo)")
    print("-" * 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # FIXED: Create data loaders properly
        train_loader, val_loader, test_loader = create_data_loaders(datasets, batch_size=min(16, n_train))

        model_config = {
            'in_features': n_features,
            'n_classes': n_classes,
            'hidden_dim': 64,  # Reduced for small dataset
            'dropout': 0.3,
            'num_layers': 2
        }
        print(f"Creating HGNN model with config: {model_config}")

        model = create_model('basic', model_config)
        trainer = FixedHGNNTrainer(model, device, lr=0.001, weight_decay=1e-4)

        print("Training model (demo with few epochs)...\n")

        num_epochs = 5
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Micro F1: {val_metrics.get('micro_f1', 0.0):.4f}")
            print(f"  Val Top-5 Accuracy: {val_metrics.get('top_5_accuracy', 0.0):.4f}\n")

        # Step 4: Evaluation
        print("Step 4: Model Evaluation")
        print("-" * 24)
        print("Evaluating on test set...")

        test_metrics = trainer.evaluate(test_loader)

        print("Test Results:")
        print(f"  Test Loss: {test_metrics.get('loss', 0.0):.4f}")
        print(f"  Test Micro F1: {test_metrics.get('micro_f1', 0.0):.4f}")
        print(f"  Test Macro F1: {test_metrics.get('macro_f1', 0.0):.4f}")
        print(f"  Test Micro AUC: {test_metrics.get('micro_auc', 0.0):.4f}")
        print(f"  Test Top-5 Accuracy: {test_metrics.get('top_5_accuracy', 0.0):.4f}")
        print(f"  Test Top-10 Accuracy: {test_metrics.get('top_10_accuracy', 0.0):.4f}")

        torch.save(model.state_dict(), 'demo_model.pt')
        print("\nModel saved to 'demo_model.pt'")

        demo_results = {
            'datasets': datasets,
            'model': model,
            'test_metrics': test_metrics,
            'hypergraph_stats': {
                'n_nodes': H_train.shape[0],
                'n_edges': H_train.shape[1],
                'density': np.sum(H_train) / (H_train.shape[0] * H_train.shape[1]) if H_train.shape[0] > 0 else 0
            },
            'k_classes': k_classes,
            'topk_codes': topk_codes
        }

    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 5: Comparison Summary
    print("\nStep 5: Results Summary")
    print("-" * 33)
    print(f"Classes: {k_classes} most frequent codes (reduced from 2981)")
    print(f"Training samples: {n_train}")
    print(f"Avg samples per class: {avg_samples_per_class:.1f}")

    print("\n" + "="*50)
    print("HGNN MIMIC Demo Completed Successfully!")
    print("="*50)
    print(f"\nFinal test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nGenerated Files:")
    print("- processed_datasets.pkl (datasets)")
    print("- demo_model.pt (trained model)")

    return demo_results


if __name__ == "__main__":
    print("Starting HGNN Demo...\n")

    # You can change k_classes: try 20, 50, or 100
    results = run_demo(k_classes=50)

    if results:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("\nNext steps:")
        print("1. Check metrics are now meaningful (>0.01)")
        print("2. Try k_classes=100 for more classes")
        print("3. Implement full CCS mapping for better semantics")
    else:
        print("\nDemo encountered errors. Check messages above.")