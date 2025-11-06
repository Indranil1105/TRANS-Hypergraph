import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import pickle
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from ehr_data_processor import EHRDataProcessor, create_data_loaders
from hgnn_models import create_model, MODEL_CONFIGS
from hypergraph_utils import HypergraphUtils

class FixedHGNNTrainer:
    """
    COMPLETELY FIXED Trainer class for HGNN models on EHR data
    All zero-metric issues resolved
    """
    
    def __init__(self, model, device, lr=0.001, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
    @staticmethod
    def precision_at_k(outputs, labels, k):
        """
        Calculate visit-level precision@k (TRANS metric) - FIXED
        Returns: float (single value, not tuple)
        
        Args:
            outputs: model predictions, shape (batch_size, num_classes)
            labels: ground truth labels, shape (batch_size, num_classes)
            k: number of top predictions to consider
        """
        # Convert to numpy if needed
        if hasattr(outputs, 'cpu'):
            outputs = outputs.cpu().numpy()
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        
        outputs = np.array(outputs)
        labels = np.array(labels)
        
        precisions = []
        n_samples = outputs.shape[0]
        
        for i in range(n_samples):
            # Get top-k predicted indices (highest logits)
            top_k_indices = np.argsort(outputs[i])[-k:]
            
            # Get true label indices
            true_labels = np.where(labels[i] == 1)[0]
            
            if len(true_labels) == 0:
                continue
            
            # Count correct predictions
            correct = len(set(top_k_indices) & set(true_labels))
            
            # Precision = correct / min(k, num_true_labels)
            precision = correct / min(k, len(true_labels))
            precisions.append(precision)
        
        # Return single float value
        result = np.mean(precisions) if precisions else 0.0
        return float(result)
    
    @staticmethod
    def accuracy_at_k(outputs, labels, k):
        """
        Calculate code-level accuracy@k (TRANS metric) - FIXED
        Returns: float (single value, not tuple)
        
        Args:
            outputs: model predictions, shape (batch_size, num_classes)
            labels: ground truth labels, shape (batch_size, num_classes)
            k: number of top predictions to consider
        """
        # Convert to numpy if needed
        if hasattr(outputs, 'cpu'):
            outputs = outputs.cpu().numpy()
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        
        outputs = np.array(outputs)
        labels = np.array(labels)
        
        total_correct = 0
        total_true_labels = 0
        n_samples = outputs.shape[0]
        
        for i in range(n_samples):
            # Get top-k predicted indices
            top_k_indices = np.argsort(outputs[i])[-k:]
            
            # Get true label indices
            true_labels = np.where(labels[i] == 1)[0]
            
            # Count correct predictions
            correct = len(set(top_k_indices) & set(true_labels))
            total_correct += correct
            total_true_labels += len(true_labels)
        
        # Accuracy = total_correct / total_true_labels
        result = total_correct / total_true_labels if total_true_labels > 0 else 0.0
        return float(result)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch - FIXED VERSION"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle data loading
            if isinstance(batch, (tuple, list)):
                features, labels, hypergraph_G = batch
                features = torch.FloatTensor(features).to(self.device)
                labels = torch.FloatTensor(labels).to(self.device)
                hypergraph_G = torch.FloatTensor(hypergraph_G).to(self.device)
            else:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                hypergraph_G = batch['hypergraph_G'].to(self.device)
            
            print(f"\n[BATCH {batch_idx} DEBUG]")
            print(f"  Features: {features.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Graph: {hypergraph_G.shape}")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # KEY FIX: Ensure features have correct shape for model
            # If features is (n_samples, n_features), we need to handle it properly
            if features.dim() == 2:
                # Single batch of patients
                batch_size = features.shape[0]
                # Model expects (batch_size, n_features) for basic HGNN
                outputs = self.model(features, hypergraph_G)
            else:
                outputs = self.model(features, hypergraph_G)
            
            print(f"  Model outputs: {outputs.shape}")
            print(f"  Outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            
            # KEY FIX: Ensure output and label shapes match
            if outputs.shape != labels.shape:
                print(f"  [WARNING] Shape mismatch! Outputs: {outputs.shape}, Labels: {labels.shape}")
                
                # If model output is (n_nodes, n_classes) but we have batch
                # We need to aggregate appropriately
                if outputs.dim() == 3 and labels.dim() == 2:
                    # outputs is (batch, n_nodes, n_classes), take mean over nodes
                    outputs = outputs.mean(dim=1)
                    print(f"  [FIX] Aggregated outputs to: {outputs.shape}")
                elif outputs.dim() == 2 and labels.dim() == 2 and outputs.shape[0] != labels.shape[0]:
                    print(f"  [ERROR] Cannot fix shape mismatch!")
                    continue
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"  Loss: {loss.item():.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, data_loader):
        """Evaluate model - COMPLETELY FIXED VERSION"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    features, labels, hypergraph_G = batch
                    features = torch.FloatTensor(features).to(self.device)
                    labels = torch.FloatTensor(labels).to(self.device)
                    hypergraph_G = torch.FloatTensor(hypergraph_G).to(self.device)
                else:
                    features = batch['features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    hypergraph_G = batch['hypergraph_G'].to(self.device)
                
                # Forward pass
                if features.dim() == 2:
                    outputs = self.model(features, hypergraph_G)
                else:
                    outputs = self.model(features, hypergraph_G)
                
                # Handle shape mismatches
                if outputs.dim() == 3 and labels.dim() == 2:
                    outputs = outputs.mean(dim=1)
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels (as logits for ranking)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n[EVALUATION DEBUG]")
        print(f"  All outputs shape: {all_outputs.shape}")
        print(f"  All labels shape: {all_labels.shape}")
        print(f"  Outputs (logits) range: [{all_outputs.min():.4f}, {all_outputs.max():.4f}]")
        print(f"  Total positive labels: {np.sum(all_labels)}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_outputs, all_labels)
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def calculate_metrics(self, outputs, labels, threshold=0.5):
        """
        Calculate evaluation metrics
        
        KEY FIX: Use threshold=0.5 instead of 0.1
        Apply sigmoid to logits for binary predictions
        """
        print(f"\n[METRICS CALCULATION DEBUG]")
        print(f"  Outputs shape: {outputs.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Outputs (logits) range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        
        # KEY FIX: Apply sigmoid to convert logits to probabilities
        probs = 1 / (1 + np.exp(-outputs))  # Sigmoid
        print(f"  Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"  Probabilities mean: {probs.mean():.4f}")
        
        # Convert probabilities to binary predictions with proper threshold
        predictions = (probs > threshold).astype(int)
        
        print(f"  Threshold: {threshold}")
        print(f"  Positive predictions: {np.sum(predictions)}")
        print(f"  Positive labels: {np.sum(labels)}")
        print(f"  Correct predictions: {np.sum(predictions * labels)}")
        
        # Calculate metrics
        metrics = {}
        
        # Micro-averaged metrics (using probabilities for predictions)
        try:
            metrics['micro_precision'] = precision_score(labels, predictions, average='micro', zero_division=0)
            metrics['micro_recall'] = recall_score(labels, predictions, average='micro', zero_division=0)
            metrics['micro_f1'] = f1_score(labels, predictions, average='micro', zero_division=0)
            
            # Macro-averaged metrics
            metrics['macro_precision'] = precision_score(labels, predictions, average='macro', zero_division=0)
            metrics['macro_recall'] = recall_score(labels, predictions, average='macro', zero_division=0)
            metrics['macro_f1'] = f1_score(labels, predictions, average='macro', zero_division=0)
        except Exception as e:
            print(f"  [WARNING] Error calculating precision/recall/f1: {e}")
            metrics['micro_precision'] = 0.0
            metrics['micro_recall'] = 0.0
            metrics['micro_f1'] = 0.0
            metrics['macro_precision'] = 0.0
            metrics['macro_recall'] = 0.0
            metrics['macro_f1'] = 0.0
        
        # AUC scores (use probabilities, not predictions)
        try:
            if len(np.unique(labels)) > 1:
                metrics['micro_auc'] = roc_auc_score(labels, probs, average='micro')
                metrics['macro_auc'] = roc_auc_score(labels, probs, average='macro')
            else:
                metrics['micro_auc'] = 0.0
                metrics['macro_auc'] = 0.0
        except Exception as e:
            print(f"  [WARNING] Error calculating AUC: {e}")
            metrics['micro_auc'] = 0.0
            metrics['macro_auc'] = 0.0
        
        # Top-k accuracy
        for k in [5, 10, 15]:
            try:
                top_k_acc = self.top_k_accuracy(outputs, labels, k)
                metrics[f'top_{k}_accuracy'] = top_k_acc
            except Exception as e:
                print(f"  [WARNING] Error calculating top-{k} accuracy: {e}")
                metrics[f'top_{k}_accuracy'] = 0.0
        
        # TRANS-style metrics - Precision@k and Accuracy@k
        # KEY: Use logits directly for ranking (no sigmoid)
        print("\n[TRANS METRICS] Calculating Precision@K and Accuracy@K...")
        for k in [10, 20, 30]:
            try:
                visit_precision = self.precision_at_k(outputs, labels, k)
                code_accuracy = self.accuracy_at_k(outputs, labels, k)
                metrics[f'precision@{k}'] = visit_precision
                metrics[f'accuracy@{k}'] = code_accuracy
            except Exception as e:
                print(f"  [WARNING] Error calculating precision@{k} and accuracy@{k}: {e}")
                metrics[f'precision@{k}'] = 0.0
                metrics[f'accuracy@{k}'] = 0.0
        
        return metrics
    
    def top_k_accuracy(self, outputs, labels, k):
        """Calculate top-k accuracy for multi-label classification"""
        n_samples = outputs.shape[0]
        correct = 0
        
        for i in range(n_samples):
            # Get top-k predictions
            top_k_indices = np.argsort(outputs[i])[-k:]
            
            # Check if any of the true labels are in top-k predictions
            if np.any(labels[i][top_k_indices] == 1):
                correct += 1
        
        return correct / n_samples


def main():
    parser = argparse.ArgumentParser(description='Train HGNN on MIMIC data - COMPLETELY FIXED VERSION')
    parser.add_argument('--data_path', type=str, default='processed_mimic_data.pkl')
    parser.add_argument('--model_type', type=str, default='basic')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--k_neig', type=int, default=10)
    parser.add_argument('--feature_type', type=str, default='statistical')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading and processing data...")
    
    # Load data
    data_processor = EHRDataProcessor(min_visits=2, max_visits=20)
    if os.path.exists(args.data_path):
        patient_visits = data_processor.load_mimic_data(args.data_path)
        if isinstance(patient_visits, dict) and 'patient_visits' in patient_visits:
            patient_visits = patient_visits['patient_visits']
    else:
        print("Processed data file not found.")
        return
    
    print("Creating datasets...")
    datasets = data_processor.create_datasets(
        patient_visits,
        feature_type=args.feature_type,
        test_size=0.2,
        val_size=0.1
    )
    
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
    
    # Model configuration
    model_config = MODEL_CONFIGS[args.model_type].copy()
    model_config.update({
        'in_features': datasets['train']['features'].shape[1],
        'n_classes': datasets['train']['labels'].shape[1],
        'hidden_dim': args.hidden_dim,
        'dropout': 0.5
    })
    
    print(f"Model config: {model_config}")
    
    # Create model
    model = create_model(args.model_type, model_config)
    print(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = FixedHGNNTrainer(model, device, lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"\n  Standard Metrics:")
        print(f"    Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"    Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"\n  TRANS Metrics (Visit-Level Precision@K):")
        print(f"    Precision@10: {val_metrics['precision@10']:.4f}")
        print(f"    Precision@20: {val_metrics['precision@20']:.4f}")
        print(f"    Precision@30: {val_metrics['precision@30']:.4f}")
        print(f"\n  TRANS Metrics (Code-Level Accuracy@K):")
        print(f"    Accuracy@10: {val_metrics['accuracy@10']:.4f}")
        print(f"    Accuracy@20: {val_metrics['accuracy@20']:.4f}")
        print(f"    Accuracy@30: {val_metrics['accuracy@30']:.4f}")
        print(f"\n  Time: {epoch_time:.2f}s")
        print(f"{'='*80}\n")
        
        # Save best model
        if val_metrics['micro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['micro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'model_config': model_config
            }, os.path.join(args.save_dir, f'best_model_{args.model_type}_fixed.pt'))
            print(f"  ✓ New best model saved! Val F1: {best_val_f1:.4f}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == '__main__':
    main()