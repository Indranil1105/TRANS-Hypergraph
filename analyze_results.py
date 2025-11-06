# Data Analysis and Visualization for HGNN Results
# Analyze model performance and visualize hypergraph structures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
import torch
import networkx as nx
from hypergraph_utils import HypergraphUtils

class HGNNAnalyzer:
    """
    Analyzer for HGNN model results and hypergraph structures
    """
    def __init__(self, results_path=None):
        self.results = None
        if results_path:
            self.load_results(results_path)
    
    def load_results(self, results_path):
        """Load saved model results"""
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
    
    def plot_training_curves(self, log_dir):
        """Plot training and validation curves from tensorboard logs"""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            event_acc = EventAccumulator(log_dir)
            event_acc.Reload()
            
            # Get available scalar tags
            tags = event_acc.Tags()['scalars']
            print("Available metrics:", tags)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot loss curves
            if 'Loss/Train' in tags and 'Loss/Val' in tags:
                train_loss = event_acc.Scalars('Loss/Train')
                val_loss = event_acc.Scalars('Loss/Val')
                
                epochs = [s.step for s in train_loss]
                train_values = [s.value for s in train_loss]
                val_values = [s.value for s in val_loss]
                
                axes[0, 0].plot(epochs, train_values, label='Train')
                axes[0, 0].plot(epochs, val_values, label='Validation')
                axes[0, 0].set_title('Loss Curves')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Plot F1 scores
            if 'F1/Val_Micro' in tags and 'F1/Val_Macro' in tags:
                micro_f1 = event_acc.Scalars('F1/Val_Micro')
                macro_f1 = event_acc.Scalars('F1/Val_Macro')
                
                epochs = [s.step for s in micro_f1]
                micro_values = [s.value for s in micro_f1]
                macro_values = [s.value for s in macro_f1]
                
                axes[0, 1].plot(epochs, micro_values, label='Micro F1')
                axes[0, 1].plot(epochs, macro_values, label='Macro F1')
                axes[0, 1].set_title('F1 Score Curves')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Plot AUC scores
            if 'AUC/Val_Micro' in tags and 'AUC/Val_Macro' in tags:
                micro_auc = event_acc.Scalars('AUC/Val_Micro')
                macro_auc = event_acc.Scalars('AUC/Val_Macro')
                
                epochs = [s.step for s in micro_auc]
                micro_values = [s.value for s in micro_auc]
                macro_values = [s.value for s in macro_auc]
                
                axes[1, 0].plot(epochs, micro_values, label='Micro AUC')
                axes[1, 0].plot(epochs, macro_values, label='Macro AUC')
                axes[1, 0].set_title('AUC Score Curves')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('AUC Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Tensorboard not available. Please install tensorboard to plot training curves.")
    
    def analyze_hypergraph_structure(self, H, title="Hypergraph Analysis"):
        """Analyze and visualize hypergraph structure"""
        print(f"\n{title}")
        print("=" * len(title))
        
        # Basic statistics
        n_nodes, n_edges = H.shape
        total_connections = np.sum(H)
        avg_node_degree = np.mean(np.sum(H, axis=1))
        avg_edge_size = np.mean(np.sum(H, axis=0))
        
        print(f"Number of nodes: {n_nodes}")
        print(f"Number of hyperedges: {n_edges}")
        print(f"Total connections: {total_connections}")
        print(f"Average node degree: {avg_node_degree:.2f}")
        print(f"Average hyperedge size: {avg_edge_size:.2f}")
        print(f"Density: {total_connections / (n_nodes * n_edges):.4f}")
        
        # Degree distributions
        node_degrees = np.sum(H, axis=1)
        edge_sizes = np.sum(H, axis=0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Node degree distribution
        axes[0].hist(node_degrees, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title('Node Degree Distribution')
        axes[0].set_xlabel('Degree')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Edge size distribution
        axes[1].hist(edge_sizes, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_title('Hyperedge Size Distribution')
        axes[1].set_xlabel('Size')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Hypergraph adjacency matrix visualization
        im = axes[2].imshow(H, cmap='Blues', aspect='auto')
        axes[2].set_title('Hypergraph Incidence Matrix')
        axes[2].set_xlabel('Hyperedges')
        axes[2].set_ylabel('Nodes')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'hypergraph_analysis_{title.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_embeddings(self, features, labels, method='tsne', n_components=2):
        """Visualize patient embeddings using dimensionality reduction"""
        print(f"Visualizing embeddings using {method.upper()}...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Reduce dimensionality
        embeddings_2d = reducer.fit_transform(features)
        
        # Create labels for plotting (use first diagnosis as class)
        plot_labels = np.argmax(labels, axis=1)
        unique_labels = np.unique(plot_labels)
        n_classes = min(len(unique_labels), 10)  # Limit to 10 classes for visibility
        
        plt.figure(figsize=(12, 8))
        
        # Plot each class with different color
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i, label in enumerate(unique_labels[:n_classes]):
            mask = plot_labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        
        plt.title(f'Patient Embeddings Visualization ({method.upper()})')
        plt.xlabel(f'{method.upper()}_1')
        plt.ylabel(f'{method.upper()}_2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'embeddings_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, results_dict):
        """Compare performance across different models"""
        models = list(results_dict.keys())
        metrics = ['micro_f1', 'macro_f1', 'micro_auc', 'macro_auc', 'top_5_accuracy']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model]['test_metrics'][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_prediction_patterns(self, y_true, y_pred, class_names=None):
        """Analyze prediction patterns and errors"""
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate per-class metrics
        n_classes = y_true.shape[1]
        class_metrics = []
        
        for i in range(n_classes):
            precision = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            
            try:
                auc_score = auc(*roc_curve(y_true[:, i], y_pred[:, i])[:2])
            except:
                auc_score = 0.0
            
            class_metrics.append({
                'class': i,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score,
                'support': np.sum(y_true[:, i])
            })
        
        # Convert to DataFrame for easier analysis
        metrics_df = pd.DataFrame(class_metrics)
        metrics_df = metrics_df.sort_values('f1', ascending=False)
        
        # Plot top and bottom performing classes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top 20 classes by F1 score
        top_classes = metrics_df.head(20)
        axes[0, 0].barh(range(len(top_classes)), top_classes['f1'])
        axes[0, 0].set_yticks(range(len(top_classes)))
        axes[0, 0].set_yticklabels([f'Class {int(c)}' for c in top_classes['class']])
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_title('Top 20 Classes by F1 Score')
        axes[0, 0].invert_yaxis()
        
        # Bottom 20 classes by F1 score
        bottom_classes = metrics_df.tail(20)
        axes[0, 1].barh(range(len(bottom_classes)), bottom_classes['f1'])
        axes[0, 1].set_yticks(range(len(bottom_classes)))
        axes[0, 1].set_yticklabels([f'Class {int(c)}' for c in bottom_classes['class']])
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_title('Bottom 20 Classes by F1 Score')
        axes[0, 1].invert_yaxis()
        
        # Class frequency vs performance
        axes[1, 0].scatter(metrics_df['support'], metrics_df['f1'], alpha=0.6)
        axes[1, 0].set_xlabel('Class Support (Number of Positive Examples)')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Class Support vs F1 Score')
        axes[1, 0].set_xscale('log')
        
        # Distribution of F1 scores
        axes[1, 1].hist(metrics_df['f1'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('F1 Score')
        axes[1, 1].set_ylabel('Number of Classes')
        axes[1, 1].set_title('Distribution of F1 Scores Across Classes')
        axes[1, 1].axvline(metrics_df['f1'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {metrics_df["f1"].mean():.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nPrediction Analysis Summary:")
        print("=" * 30)
        print(f"Number of classes: {n_classes}")
        print(f"Mean F1 score: {metrics_df['f1'].mean():.3f}")
        print(f"Std F1 score: {metrics_df['f1'].std():.3f}")
        print(f"Classes with F1 > 0.5: {len(metrics_df[metrics_df['f1'] > 0.5])}")
        print(f"Classes with F1 > 0.3: {len(metrics_df[metrics_df['f1'] > 0.3])}")
        print(f"Classes with zero support: {len(metrics_df[metrics_df['support'] == 0])}")
        
        return metrics_df

def generate_analysis_report(data_path, results_paths, log_dir=None):
    """Generate comprehensive analysis report"""
    print("HGNN Analysis Report")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = HGNNAnalyzer()
    
    # Load and analyze data
    print("\n1. Data Analysis")
    print("-" * 20)
    
    try:
        with open(data_path, 'rb') as f:
            datasets = pickle.load(f)
        
        print(f"Training samples: {len(datasets['train']['features'])}")
        print(f"Validation samples: {len(datasets['val']['features'])}")
        print(f"Test samples: {len(datasets['test']['features'])}")
        print(f"Feature dimension: {datasets['train']['features'].shape[1]}")
        print(f"Number of classes: {datasets['train']['labels'].shape[1]}")
        
        # Analyze hypergraph structures
        analyzer.analyze_hypergraph_structure(
            datasets['train']['hypergraph_H'], "Training Hypergraph")
        
        # Visualize embeddings
        analyzer.visualize_embeddings(
            datasets['test']['features'], 
            datasets['test']['labels'], 
            method='tsne'
        )
        
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Load and compare model results
    print("\n2. Model Performance Analysis")
    print("-" * 30)
    
    results_dict = {}
    for model_name, path in results_paths.items():
        try:
            analyzer.load_results(path)
            results_dict[model_name] = analyzer.results
            
            print(f"\n{model_name} Results:")
            metrics = analyzer.results['test_metrics']
            print(f"  Micro F1: {metrics['micro_f1']:.3f}")
            print(f"  Macro F1: {metrics['macro_f1']:.3f}")
            print(f"  Micro AUC: {metrics['micro_auc']:.3f}")
            print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.3f}")
            
        except Exception as e:
            print(f"Error loading results for {model_name}: {e}")
    
    # Compare models
    if len(results_dict) > 1:
        analyzer.plot_performance_comparison(results_dict)
    
    # Plot training curves if available
    if log_dir:
        analyzer.plot_training_curves(log_dir)
    
    print("\nAnalysis completed! Check generated plots for detailed insights.")

if __name__ == "__main__":
    # Example usage
    data_path = "processed_datasets.pkl"
    results_paths = {
        "Basic HGNN": "checkpoints/test_results_basic.pkl",
        "Advanced HGNN": "checkpoints/test_results_advanced.pkl"
    }
    log_dir = "logs"
    
    # generate_analysis_report(data_path, results_paths, log_dir)