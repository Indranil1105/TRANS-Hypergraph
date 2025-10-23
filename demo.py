# Fixed Demo Script for HGNN on MIMIC Data
# Demonstrates the complete pipeline from data processing to model training

import os
import sys
import pickle
import numpy as np
import torch
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ehr_data_processor import EHRDataProcessor, create_data_loaders
from hgnn_models import create_model, MODEL_CONFIGS
from train_hgnn_o import HGNNTrainer
from analyze_results import HGNNAnalyzer

def run_demo():
    """
    Run complete demo of HGNN on MIMIC data
    """
    print("HGNN MIMIC Demo - FIXED VERSION")
    print("===============================")
    print("This demo shows how to replace heterographs with hypergraphs in the TRANS model.")
    print()
    
    # Step 1: Data Processing
    print("Step 1: Data Processing")
    print("-" * 25)
    
    # Load processed MIMIC data or create from CSV files
    if os.path.exists('processed_mimic_data.pkl'):
        print("Loading preprocessed MIMIC data...")
        data_processor = EHRDataProcessor(min_visits=2, max_visits=20)
        
        # Load the pickle file
        with open('processed_mimic_data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)
            
        # Extract patient visits data
        if isinstance(loaded_data, dict) and 'patient_visits' in loaded_data:
            patient_visits = loaded_data['patient_visits']
        else:
            patient_visits = loaded_data
        
        print(f"Loaded data for {len(patient_visits)} patients")
        
    else:
        print("Preprocessed data not found. Creating from CSV files...")
        data_processor = EHRDataProcessor(min_visits=2, max_visits=20)
        
        # Load CSV files directly
        import pandas as pd
        admissions = pd.read_csv('ADMISSIONS.csv')
        patients = pd.read_csv('PATIENTS.csv')
        diagnoses = pd.read_csv('DIAGNOSES_ICD.csv')
        procedures = pd.read_csv('PROCEDURES_ICD.csv')
        
        # Process raw data  
        patient_visits = data_processor.preprocess_raw_data(admissions, patients, diagnoses, procedures)
        print(f"Processed data for {len(patient_visits)} patients")
        
        # Save processed data
        with open('processed_mimic_data_new.pkl', 'wb') as f:
            pickle.dump({'patient_visits': patient_visits}, f)
        print("Saved processed data to 'processed_mimic_data_new.pkl'")
    
    # Create datasets with hypergraph construction
    print("Creating datasets and constructing hypergraphs...")
    try:
        datasets = data_processor.create_datasets(
            patient_visits, 
            feature_type='statistical',  # Use statistical features for demo
            test_size=0.2,
            val_size=0.1
        )
        
        print(f"Training samples: {len(datasets['train']['features'])}")
        print(f"Validation samples: {len(datasets['val']['features'])}")
        print(f"Test samples: {len(datasets['test']['features'])}")
        print(f"Feature dimension: {datasets['train']['features'].shape[1]}")
        print(f"Number of diagnosis classes: {datasets['train']['labels'].shape[1]}")
        
        # Save processed datasets
        with open('processed_datasets.pkl', 'wb') as f:
            pickle.dump(datasets, f)
        print("Saved datasets to 'processed_datasets.pkl'")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None
    
    # Step 2: Hypergraph Analysis
    print("\nStep 2: Hypergraph Analysis")
    print("-" * 27)
    
    # Analyze hypergraph structure
    try:
        analyzer = HGNNAnalyzer()
        H_train = datasets['train']['hypergraph_H']
        
        print("Analyzing hypergraph structure...")
        print(f"Hypergraph incidence matrix shape: {H_train.shape}")
        print(f"Number of nodes (patients): {H_train.shape[0]}")
        print(f"Number of hyperedges: {H_train.shape[1]}")
        print(f"Hypergraph density: {np.sum(H_train) / (H_train.shape[0] * H_train.shape[1]):.4f}")
        
        # Basic statistics
        node_degrees = np.sum(H_train, axis=1)
        edge_sizes = np.sum(H_train, axis=0)
        print(f"Average node degree: {np.mean(node_degrees):.2f}")
        print(f"Average hyperedge size: {np.mean(edge_sizes):.2f}")
        
    except Exception as e:
        print(f"Error in hypergraph analysis: {e}")
    
    # Step 3: Model Training (Quick Demo)
    print("\nStep 3: Model Training (Demo)")
    print("-" * 30)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(datasets, batch_size=16)
        
        # Model configuration for demo (smaller model for faster training)
        model_config = {
            'in_features': datasets['train']['features'].shape[1],
            'n_classes': datasets['train']['labels'].shape[1],
            'hidden_dim': 128,  # Smaller for demo
            'dropout': 0.5,
            'num_layers': 2
        }
        
        print(f"Creating HGNN model with config: {model_config}")
        
        # Create and train model
        model = create_model('basic', model_config)
        trainer = HGNNTrainer(model, device, lr=0.001, weight_decay=1e-4)
        
        print("Training model (demo with few epochs)...")
        num_epochs = 5  # Few epochs for demo
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train one epoch
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Val Top-5 Accuracy: {val_metrics['top_5_accuracy']:.4f}")
        
        # Step 4: Evaluation
        print("\nStep 4: Model Evaluation")
        print("-" * 24)
        
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        print("Test Results:")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Micro F1: {test_metrics['micro_f1']:.4f}")
        print(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Test Micro AUC: {test_metrics['micro_auc']:.4f}")
        print(f"  Test Top-5 Accuracy: {test_metrics['top_5_accuracy']:.4f}")
        print(f"  Test Top-10 Accuracy: {test_metrics['top_10_accuracy']:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'demo_model.pt')
        print("Model saved to 'demo_model.pt'")
        
        demo_results = {
            'datasets': datasets,
            'model': model,
            'test_metrics': test_metrics,
            'hypergraph_stats': {
                'n_nodes': H_train.shape[0],
                'n_edges': H_train.shape[1],
                'density': np.sum(H_train) / (H_train.shape[0] * H_train.shape[1])
            }
        }
        
    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 5: Comparison with Original TRANS
    print("\nStep 5: TRANS vs HGNN Comparison")
    print("-" * 33)
    
    print("Key Differences:")
    print("1. Graph Structure:")
    print("   - TRANS: Uses heterogeneous graphs with visit and medical event nodes")
    print("   - HGNN: Uses hypergraphs where hyperedges connect multiple patients with similar patterns")
    
    print("\n2. Convolution Operation:")
    print("   - TRANS: Heterogeneous graph convolution with temporal edge features")
    print("   - HGNN: Hypergraph convolution that captures high-order relationships")
    
    print("\n3. Information Propagation:")
    print("   - TRANS: Node-edge-node message passing in bipartite graph")
    print("   - HGNN: Node-hyperedge-node aggregation capturing complex correlations")
    
    print("\n4. Advantages of Hypergraphs:")
    print("   - Can model relationships beyond pairwise connections")
    print("   - Better capture of complex medical co-occurrence patterns")
    print("   - More flexible representation of patient similarities")
    print("   - Natural handling of multi-modal medical data")
    
    # Step 6: Implementation Steps Summary
    print("\nStep 6: Implementation Steps Summary")
    print("-" * 36)
    
    print("Steps taken to replace heterographs with hypergraphs:")
    print("1. Data Preprocessing:")
    print("   - Extracted patient features from MIMIC data")
    print("   - Created medical code vocabularies")
    print("   - Built patient visit sequences")
    
    print("\n2. Hypergraph Construction:")
    print("   - Used K-nearest neighbors to create hyperedges")
    print("   - Applied probabilistic edge weights")
    print("   - Generated normalized hypergraph Laplacian")
    
    print("\n3. Model Architecture:")
    print("   - Implemented HGNN layers with hypergraph convolution")
    print("   - Added multi-head attention mechanisms")
    print("   - Integrated temporal awareness")
    
    print("\n4. Training Pipeline:")
    print("   - Multi-label classification for diagnosis prediction")
    print("   - BCEWithLogitsLoss for multi-label targets")
    print("   - Comprehensive evaluation metrics")
    
    print("\n5. Analysis and Visualization:")
    print("   - Hypergraph structure analysis")
    print("   - Model performance visualization")
    print("   - Comparison with baseline methods")
    
    print("\nDemo completed successfully!")
    print("Check generated files and saved models for detailed results.")
    
    return demo_results

def compare_with_original_trans():
    """
    Provide detailed comparison with original TRANS model
    """
    print("\nDetailed Comparison: TRANS (Heterograph) vs HGNN (Hypergraph)")
    print("=" * 65)
    
    comparison_table = """
    | Aspect                | TRANS (Original)           | HGNN (Our Implementation) |
    |-----------------------|----------------------------|----------------------------|
    | Graph Type            | Heterogeneous Graph        | Hypergraph                |
    | Node Types            | Visit + Medical Events     | Patients (unified)        |
    | Edge Types            | Visit-Event, Visit-Visit   | Patient-Patient (via hyperedges) |
    | Relationship Modeling | Pairwise only              | High-order (3+)          |
    | Convolution           | HGTConv                    | HGNN_conv                 |
    | Temporal Handling     | Edge features + Time2Vec   | Temporal hypergraph conv  |
    | Message Passing       | Node→Edge→Node             | Node→Hyperedge→Node       |
    | Scalability           | Limited by graph size      | Better for large datasets |
    | Medical Patterns      | Visit-level                | Population-level          |
    """
    
    print(comparison_table)
    
    print("\nKey Algorithmic Differences:")
    print("-" * 29)
    
    print("1. Graph Construction:")
    print("   TRANS: G = (V_visit ∪ V_event, E_bipartite ∪ E_temporal)")
    print("   HGNN:  H ∈ R^(N×M), G = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)")
    
    print("\n2. Convolution Operation:")
    print("   TRANS: X^(l+1) = σ(Aggregate(X^(l), edge_features, attention))")
    print("   HGNN:  X^(l+1) = σ(G × X^(l) × Θ^(l))")
    
    print("\n3. Advantages of Our Approach:")
    print("   - Simpler mathematical formulation")
    print("   - Direct capture of patient similarities")
    print("   - Better handling of sparse medical data")
    print("   - More interpretable hyperedge structures")
    print("   - Flexible multi-modal integration")

if __name__ == "__main__":
    # Run the complete demo
    print("Starting HGNN Demo...")
    results = run_demo()
    
    if results:
        # Show detailed comparison
        compare_with_original_trans()
        
        print("\n" + "="*50)
        print("HGNN MIMIC Demo Completed Successfully!")
        print("="*50)
        
        print(f"Final test metrics: {results['test_metrics']}")
        print(f"Hypergraph statistics: {results['hypergraph_stats']}")
        
        print("\nGenerated Files:")
        print("- processed_datasets.pkl (datasets)")
        print("- demo_model.pt (trained model)")
        
    else:
        print("\nDemo encountered errors. Please check the error messages above.")
    
    print("\nTo run full training:")
    print("python train_hgnn.py --model_type basic --epochs 100 --batch_size 32")
    print("\nTo analyze results:")
    print("python analyze_results.py")