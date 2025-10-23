import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import Counter

# ============================================================================
# STEP 1: CREATE DATA PROCESSOR WITH TOP-K CLASSES
# ============================================================================

def reduce_to_top_k_classes(patient_visits, k=50):
    """
    Reduce dataset to only top-k most frequent diagnosis codes
    This dramatically improves learning with limited data
    """
    print(f"\n{'='*80}")
    print(f"REDUCING TO TOP-{k} MOST FREQUENT CLASSES")
    print(f"{'='*80}")
    
    # Count all diagnosis codes
    code_counter = Counter()
    for patient_id, visits in patient_visits.items():
        for visit in visits:
            if 'diagnoses' in visit:
                code_counter.update(visit['diagnoses'])
    
    # Get top-k codes
    top_k_codes = [code for code, count in code_counter.most_common(k)]
    
    print(f"\nOriginal codes: {len(code_counter)}")
    print(f"Top-{k} codes: {len(top_k_codes)}")
    print(f"Coverage: {sum([code_counter[c] for c in top_k_codes]) / sum(code_counter.values()) * 100:.1f}%")
    
    # Filter patient visits to only include top-k codes
    filtered_visits = {}
    for patient_id, visits in patient_visits.items():
        filtered_patient_visits = []
        for visit in visits:
            if 'diagnoses' in visit:
                # Keep only top-k codes
                filtered_diagnoses = [d for d in visit['diagnoses'] if d in top_k_codes]
                if filtered_diagnoses:  # Only keep visits with at least one diagnosis
                    new_visit = visit.copy()
                    new_visit['diagnoses'] = filtered_diagnoses
                    filtered_patient_visits.append(new_visit)
        
        if len(filtered_patient_visits) >= 2:  # Keep patients with at least 2 visits
            filtered_visits[patient_id] = filtered_patient_visits
    
    print(f"\nOriginal patients: {len(patient_visits)}")
    print(f"Filtered patients: {len(filtered_visits)}")
    
    return filtered_visits, top_k_codes


# ============================================================================
# STEP 2: CALCULATE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================

def calculate_pos_weight(labels):
    """
    Calculate positive class weights for BCEWithLogitsLoss
    Helps model focus on minority (positive) class
    """
    # Count positive and negative samples per class
    pos_count = labels.sum(axis=0)
    neg_count = len(labels) - pos_count
    
    # Avoid division by zero
    pos_count = np.maximum(pos_count, 1)
    
    # pos_weight = neg_count / pos_count
    pos_weight = neg_count / pos_count
    
    # Cap maximum weight to avoid extreme values
    pos_weight = np.minimum(pos_weight, 100)
    
    return torch.FloatTensor(pos_weight)


# ============================================================================
# STEP 3: FOCAL LOSS FOR EXTREME IMBALANCE
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples, downweights easy negatives
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal weight
        # For positive class: (1-p)^gamma
        # For negative class: p^gamma
        focal_weight = torch.where(
            targets == 1,
            (1 - probs) ** self.gamma,
            probs ** self.gamma
        )
        
        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()



import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import pickle
from collections import Counter

# Your existing imports
from ehr_data_processor import EHRDataProcessor
from hgnn_models import create_model, MODEL_CONFIGS
from hypergraph_utils import HypergraphUtils

# Add the new functions above (reduce_to_top_k_classes, calculate_pos_weight, FocalLoss)

def main():
    parser = argparse.ArgumentParser(description='FIXED: Train HGNN with proper data handling')
    parser.add_argument('--data_path', type=str, default='processed_mimic_data.pkl')
    parser.add_argument('--top_k_classes', type=int, default=50)  # NEW: Reduce classes
    parser.add_argument('--use_focal_loss', action='store_true')  # NEW: Use focal loss
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)  # CHANGED: Increased from 0.0001
    parser.add_argument('--hidden_dim', type=int, default=512)  # CHANGED: Increased from 256
    parser.add_argument('--feature_type', type=str, default='temporal')  # CHANGED: Use temporal
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("LOADING AND FILTERING DATA...")
    print("="*80)
    
    # Load raw data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'patient_visits' in data:
        patient_visits = data['patient_visits']
    else:
        patient_visits = data
    
    print(f"\\nOriginal dataset: {len(patient_visits)} patients")
    
    # CRITICAL FIX: Reduce to top-k classes
    patient_visits, top_k_codes = reduce_to_top_k_classes(patient_visits, k=args.top_k_classes)
    
    # Process data
    data_processor = EHRDataProcessor(min_visits=2, max_visits=20)
    datasets = data_processor.create_datasets(
        patient_visits,
        feature_type=args.feature_type,
        test_size=0.2,
        val_size=0.1
    )
    
    print("\\n" + "="*80)
    print("DATASET STATISTICS AFTER FILTERING")
    print("="*80)
    print(f"Train samples: {datasets['train']['features'].shape[0]}")
    print(f"Val samples: {datasets['val']['features'].shape[0]}")
    print(f"Features: {datasets['train']['features'].shape[1]}")
    print(f"Classes: {datasets['train']['labels'].shape[1]}")
    print(f"Positive labels (train): {np.sum(datasets['train']['labels'])}")
    print(f"Positive labels (val): {np.sum(datasets['val']['labels'])}")
    print(f"Avg labels per sample (train): {np.sum(datasets['train']['labels']) / len(datasets['train']['labels']):.2f}")
    print(f"Avg labels per sample (val): {np.sum(datasets['val']['labels']) / len(datasets['val']['labels']):.2f}")
    
    # CRITICAL FIX: Calculate class weights
    pos_weight = calculate_pos_weight(datasets['train']['labels'])
    print(f"\\nPos weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    print(f"Pos weight mean: {pos_weight.mean():.2f}")
    
    # Create data loaders
    train_loader = [(
        datasets['train']['features'],
        datasets['train']['labels'],
        datasets['train']['hypergraph_G']
    )]
    
    val_loader = [(
        datasets['val']['features'],
        datasets['val']['labels'],
        datasets['val']['hypergraph_G']
    )]
    
    # Create model
    model_config = {
        'in_features': datasets['train']['features'].shape[1],
        'n_classes': datasets['train']['labels'].shape[1],
        'hidden_dim': args.hidden_dim,
        'dropout': 0.3,
        'num_layers': 2
    }
    
    model = create_model('basic', model_config).to(device)
    print(f"\\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # CRITICAL FIX: Use weighted loss or focal loss
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print("Using Weighted BCE Loss")
    
    # Optimizer with higher learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        for features, labels, G in train_loader:
            features = torch.FloatTensor(features).to(device)
            labels = torch.FloatTensor(labels).to(device)
            G = torch.FloatTensor(G).to(device)
            
            optimizer.zero_grad()
            outputs = model(features, G)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels, G in val_loader:
                features = torch.FloatTensor(features).to(device)
                labels = torch.FloatTensor(labels).to(device)
                G = torch.FloatTensor(G).to(device)
                
                outputs = model(features, G)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_fixed.pt')
    
    print("\\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)

if __name__ == '__main__':
    main()

# ============================================================================
# SAVE COMPLETE SCRIPT
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("QUICK FIX SCRIPT GENERATOR")
    print("="*80)
    print("\\nThis script includes:")
    print("  ✅ Reduce to top-50 classes")
    print("  ✅ Weighted BCE Loss with pos_weight")
    print("  ✅ Focal Loss option")
    print("  ✅ Higher learning rate (0.01)")
    print("  ✅ Larger model (hidden_dim=512)")
    print("  ✅ Temporal features")
    print("\\nGenerate the complete training script with:")
    print("  script = quick_fix_training_script()")
    print("  with open('train_fixed_final.py', 'w') as f:")
    print("      f.write(script)")