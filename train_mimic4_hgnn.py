# train_mimic4_hgnn.py - Training with MIMIC-IV Data Loader

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from mimic4_data_loader import create_mimic4_datasets
from hgnn_models import create_model, MODEL_CONFIGS
from hypergraph_utils import HypergraphUtils


class TRANSHGNNTrainer:
    """TRANS-Compliant Trainer with MIMIC-IV Data"""
    
    def __init__(self, model, device, lr=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_metric = 0.0
    
    @staticmethod
    def visit_level_precision_at_k(logits, labels, k):
        """Visit-Level Precision@K"""
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        
        precisions = []
        for i in range(len(logits)):
            top_k = np.argsort(logits[i])[-k:]
            true_labels = np.where(labels[i] == 1)[0]
            
            if len(true_labels) == 0:
                continue
            
            correct = len(set(top_k) & set(true_labels))
            precision = correct / min(k, len(true_labels))
            precisions.append(precision)
        
        return float(np.mean(precisions)) if precisions else 0.0
    
    @staticmethod
    def code_level_accuracy_at_k(logits, labels, k):
        """Code-Level Accuracy@K"""
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        
        total_correct = 0
        total_true = 0
        
        for i in range(len(logits)):
            top_k = np.argsort(logits[i])[-k:]
            true_labels = np.where(labels[i] == 1)[0]
            
            correct = len(set(top_k) & set(true_labels))
            total_correct += correct
            total_true += len(true_labels)
        
        return float(total_correct / total_true) if total_true > 0 else 0.0
    
    def train_epoch(self, features, labels, hypergraph_G):
        """Train for one epoch"""
        self.model.train()
        
        features = torch.FloatTensor(features).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        hypergraph_G = torch.FloatTensor(hypergraph_G).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(features, hypergraph_G)
        
        if outputs.shape != labels.shape:
            if outputs.dim() == 3 and labels.dim() == 2:
                outputs = outputs.mean(dim=1)
        
        loss = self.criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, features, labels, hypergraph_G):
        """Evaluate using TRANS metrics"""
        self.model.eval()
        
        features = torch.FloatTensor(features).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        hypergraph_G = torch.FloatTensor(hypergraph_G).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features, hypergraph_G)
            
            if outputs.dim() == 3 and labels.dim() == 2:
                outputs = outputs.mean(dim=1)
            
            loss = self.criterion(outputs, labels)
        
        # Get logits for metrics
        logits = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        metrics = self.calculate_metrics(logits, labels_np)
        metrics['loss'] = loss.item()
        
        return metrics
    
    def calculate_metrics(self, logits, labels):
        """Calculate TRANS metrics"""
        metrics = {}
        
        for k in [10, 20, 30]:
            metrics[f'precision@{k}'] = self.visit_level_precision_at_k(logits, labels, k)
            metrics[f'accuracy@{k}'] = self.code_level_accuracy_at_k(logits, labels, k)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train HGNN on MIMIC-IV')
    parser.add_argument('--admissions_path', type=str, default='ADMISSIONS.csv')
    parser.add_argument('--diagnoses_path', type=str, default='DIAGNOSES_ICD.csv')
    parser.add_argument('--procedures_path', type=str, default='PROCEDURES_ICD.csv')
    parser.add_argument('--patients_path', type=str, default='PATIENTS.csv')
    parser.add_argument('--model_type', type=str, default='basic', choices=['basic', 'advanced'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_pkl', type=str, default='mimic4_processed.pkl')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load MIMIC-IV data
    print("\nLoading MIMIC-IV data from CSV files...")
    try:
        datasets = create_mimic4_datasets(
            args.admissions_path,
            args.diagnoses_path,
            args.procedures_path,
            args.patients_path
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Please ensure CSV files are in the current directory:")
        print(f"  - {args.admissions_path}")
        print(f"  - {args.diagnoses_path}")
        print(f"  - {args.procedures_path}")
        print(f"  - {args.patients_path}")
        return
    
    # Model configuration
    model_config = MODEL_CONFIGS[args.model_type].copy()
    model_config.update({
        'in_features': datasets['train']['features'].shape[1],
        'n_classes': datasets['train']['labels'].shape[1],
        'hidden_dim': args.hidden_dim,
        'dropout': 0.5
    })
    
    print(f"\nModel config: {model_config}")
    
    # Create model
    model = create_model(args.model_type, model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created {args.model_type} model with {n_params} parameters\n")
    
    # Create trainer
    trainer = TRANSHGNNTrainer(model, device, lr=args.lr)
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch(
            datasets['train']['features'],
            datasets['train']['labels'],
            datasets['train']['hypergraph_G']
        )
        
        # Evaluate
        val_metrics = trainer.evaluate(
            datasets['val']['features'],
            datasets['val']['labels'],
            datasets['val']['hypergraph_G']
        )
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}\n")
        
        print("Visit-Level Precision@K (TRANS metric):")
        print(f"  Precision@10: {val_metrics['precision@10']:.4f}")
        print(f"  Precision@20: {val_metrics['precision@20']:.4f}")
        print(f"  Precision@30: {val_metrics['precision@30']:.4f}\n")
        
        print("Code-Level Accuracy@K (TRANS metric):")
        print(f"  Accuracy@10: {val_metrics['accuracy@10']:.4f}")
        print(f"  Accuracy@20: {val_metrics['accuracy@20']:.4f}")
        print(f"  Accuracy@30: {val_metrics['accuracy@30']:.4f}")
        print(f"{'='*80}\n")
        
        # Track best model
        current_metric = val_metrics['precision@10']
        if current_metric > trainer.best_metric:
            trainer.best_metric = current_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_metric': trainer.best_metric,
                'model_config': model_config
            }, os.path.join(args.save_dir, f'best_model_mimic4.pt'))
            print(f"✓ New best model saved! Precision@10: {trainer.best_metric:.4f}\n")
        
        # Learning rate scheduling
        trainer.scheduler.step(val_metrics['loss'])
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    test_metrics = trainer.evaluate(
        datasets['test']['features'],
        datasets['test']['labels'],
        datasets['test']['hypergraph_G']
    )
    
    print(f"\nTest Set Performance:")
    print(f"Loss: {test_metrics['loss']:.4f}\n")
    print(f"Visit-Level Precision@K:")
    print(f"  Precision@10: {test_metrics['precision@10']:.4f}")
    print(f"  Precision@20: {test_metrics['precision@20']:.4f}")
    print(f"  Precision@30: {test_metrics['precision@30']:.4f}\n")
    print(f"Code-Level Accuracy@K:")
    print(f"  Accuracy@10: {test_metrics['accuracy@10']:.4f}")
    print(f"  Accuracy@20: {test_metrics['accuracy@20']:.4f}")
    print(f"  Accuracy@30: {test_metrics['accuracy@30']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()