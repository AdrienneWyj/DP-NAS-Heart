# ============================================================
# DP-NAS + RANPAC FUSION MODEL EVALUATION
# WITH EXTENDED METRICS ON TWO DATASETS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import json

print("="*80)
print("TASK 2: FUSION MODEL EVALUATION ON TWO DATASETS")
print("="*80)

# ============================================================
# PART 1: METRICS AND DATASET CLASSES
# ============================================================

class MetricsCalculator:
    """Calculate and display comprehensive metrics"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None, model_name="", dataset_name=""):
        """Calculate all evaluation metrics"""
        
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['confusion_matrix'] = {
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'TP': int(tp)
            }
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC-AUC if probabilities available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = None
        
        # Store results
        key = f"{model_name}_{dataset_name}"
        self.results[key] = metrics
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics in formatted way"""
        print("\n" + "="*60)
        print(f"RESULTS: {metrics['model']} on {metrics['dataset']}")
        print("="*60)
        
        print("\n--- Primary Metrics ---")
        print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        if 'specificity' in metrics:
            print("\n--- Additional Metrics ---")
            print(f"Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
            print(f"Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
            print(f"NPV:         {metrics['npv']:.4f} ({metrics['npv']*100:.2f}%)")
            print(f"FPR:         {metrics['fpr']:.4f} ({metrics['fpr']*100:.2f}%)")
            print(f"FNR:         {metrics['fnr']:.4f} ({metrics['fnr']*100:.2f}%)")
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        if 'confusion_matrix' in metrics:
            print("\n--- Confusion Matrix ---")
            cm = metrics['confusion_matrix']
            print(f"              Predicted")
            print(f"              Neg    Pos")
            print(f"Actual Neg  [{cm['TN']:4d}, {cm['FP']:4d}]")
            print(f"       Pos  [{cm['FN']:4d}, {cm['TP']:4d}]")


class OptimizedDataset(Dataset):
    """Dataset class for EMG data"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# PART 2: DATA LOADING FOR TWO DATASETS
# ============================================================

def load_dataset1():
    """Load Dataset 1: Original EMG data"""
    print("\nLoading Dataset 1 (Original EMG)...")
    
    data_path = '/content/drive/MyDrive/emg/datasets'
    
    # Load data
    train_user = pd.read_excel(f'{data_path}/train_valid_user.xlsx')
    train_imposter = pd.read_excel(f'{data_path}/train_imposter.xlsx')
    test_user = pd.read_excel(f'{data_path}/test_valid_user.xlsx')
    test_imposter = pd.read_excel(f'{data_path}/test_imposter.xlsx')
    
    # Remove leaky features
    exclude_cols = ['Participant', 'Window Num', 'Awake', 'Class']
    feature_cols = [col for col in train_user.columns if col not in exclude_cols]
    
    # Extract features
    X_train = np.vstack([
        train_user[feature_cols].values,
        train_imposter[feature_cols].values
    ])
    y_train = np.array([1]*len(train_user) + [0]*len(train_imposter))
    
    X_test = np.vstack([
        test_user[feature_cols].values,
        test_imposter[feature_cols].values
    ])
    y_test = np.array([1]*len(test_user) + [0]*len(test_imposter))
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"  Dataset 1 - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def load_dataset2():
    """Load Dataset 2: Shuffled and resampled version"""
    print("\nLoading Dataset 2 (Resampled EMG)...")
    
    # First load original data
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_dataset1()
    
    # Combine all data
    X_all = np.vstack([X_train_orig, X_test_orig])
    y_all = np.hstack([y_train_orig, y_test_orig])
    
    # Shuffle and resample with different split ratio (80-20 instead of 90-10)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=123, stratify=y_all
    )
    
    print(f"  Dataset 2 - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================
# PART 3: DP-NAS MODEL
# ============================================================

class DPNASModel(nn.Module):
    """DP-NAS model with best architecture"""
    
    def __init__(self, input_dim, hidden_dims=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dims))
            layers.append(nn.BatchNorm1d(hidden_dims))
            layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dims
        
        # Don't add final layer here - we want features
        self.feature_extractor = nn.Sequential(*layers)
        self.output_dim = hidden_dims
    
    def forward(self, x):
        return self.feature_extractor(x)


# ============================================================
# PART 4: RANPAC MODEL
# ============================================================

class RanPACClassifier:
    """RanPAC classifier for fusion"""
    
    def __init__(self, input_dim, num_classes, device='cuda'):
        self.device = device
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01)
    
    def fit(self, x, y):
        """Incremental training"""
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(self.device)
        if not torch.is_tensor(y):
            y = torch.tensor(y).to(self.device)
        
        x = x.float().unsqueeze(0) if x.dim() == 1 else x.float()
        y = y.long().unsqueeze(0) if y.dim() == 0 else y.long()
        
        self.classifier.train()
        output = self.classifier(x)
        loss = F.cross_entropy(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def predict(self, x, return_probs=False):
        """Make predictions"""
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(self.device)
        x = x.float()
        
        self.classifier.eval()
        with torch.no_grad():
            output = self.classifier(x)
            if return_probs:
                return F.softmax(output, dim=1)
            else:
                return torch.argmax(output, dim=1)


# ============================================================
# PART 5: FUSION MODEL
# ============================================================

class DPNASRanPACFusion:
    """Fusion of DP-NAS feature extractor and RanPAC classifier"""
    
    def __init__(self, input_dim, device='cuda'):
        self.device = device
        
        # Initialize DP-NAS feature extractor
        self.dpnas = DPNASModel(input_dim).to(device)
        
        # Initialize RanPAC classifier
        self.ranpac = RanPACClassifier(self.dpnas.output_dim, 2, device)
        
        print(f"Fusion model initialized:")
        print(f"  DP-NAS feature dimension: {self.dpnas.output_dim}")
        print(f"  Total parameters: {sum(p.numel() for p in self.dpnas.parameters())}")
    
    def train_dpnas(self, train_loader, val_loader, epochs=15):
        """Pre-train DP-NAS feature extractor"""
        print("\nPhase 1: Training DP-NAS feature extractor...")
        
        # Add classification head for training
        classifier = nn.Linear(self.dpnas.output_dim, 2).to(self.device)
        model = nn.Sequential(self.dpnas, classifier)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 5 == 0:
                train_acc = 100. * correct / total
                print(f"  Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    def train_ranpac(self, train_loader):
        """Train RanPAC classifier with DP-NAS features"""
        print("\nPhase 2: Training RanPAC classifier...")
        
        self.dpnas.eval()  # Freeze DP-NAS
        
        # Extract features and train RanPAC incrementally
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Extract features using DP-NAS
            with torch.no_grad():
                features = self.dpnas(inputs)
            
            # Train RanPAC incrementally
            for feat, label in zip(features, targets):
                self.ranpac.fit(feat, label)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
    
    def evaluate(self, test_loader):
        """Evaluate fusion model"""
        self.dpnas.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Extract features with DP-NAS
                features = self.dpnas(inputs)
                
                # Classify with RanPAC
                probs = self.ranpac.predict(features, return_probs=True)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ============================================================
# PART 6: EVALUATION FUNCTION
# ============================================================

def evaluate_fusion_on_dataset(X_train, y_train, X_test, y_test, dataset_name):
    """Evaluate fusion model on a dataset"""
    print("\n" + "="*80)
    print(f"EVALUATING FUSION MODEL ON {dataset_name}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = OptimizedDataset(X_train_split, y_train_split)
    val_dataset = OptimizedDataset(X_val_split, y_val_split)
    test_dataset = OptimizedDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train fusion model
    fusion_model = DPNASRanPACFusion(X_train.shape[1], device)
    
    # Train DP-NAS
    fusion_model.train_dpnas(train_loader, val_loader, epochs=15)
    
    # Train RanPAC
    fusion_model.train_ranpac(train_loader)
    
    # Evaluate
    print("\nPhase 3: Evaluating fusion model...")
    y_true, y_pred, y_prob = fusion_model.evaluate(test_loader)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name='DP-NAS+RanPAC Fusion',
        dataset_name=dataset_name
    )
    
    metrics_calc.print_metrics(metrics)
    
    return metrics


# ============================================================
# PART 7: VISUALIZATION AND COMPARISON
# ============================================================

def create_comparison_visualization(results_dataset1, results_dataset2):
    """Create visualization comparing results on two datasets"""
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*80)
    
    # Prepare data for visualization
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    dataset1_values = [
        results_dataset1['accuracy'],
        results_dataset1['precision'],
        results_dataset1['recall'],
        results_dataset1['f1_score'],
        results_dataset1.get('specificity', 0)
    ]
    dataset2_values = [
        results_dataset2['accuracy'],
        results_dataset2['precision'],
        results_dataset2['recall'],
        results_dataset2['f1_score'],
        results_dataset2.get('specificity', 0)
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Bar comparison
    ax1 = axes[0]
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dataset1_values, width, label='Dataset 1', color='steelblue')
    bars2 = ax1.bar(x + width/2, dataset2_values, width, label='Dataset 2', color='coral')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Fusion Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Confusion Matrix Dataset 1
    ax2 = axes[1]
    cm1 = results_dataset1['confusion_matrix']
    cm1_array = np.array([[cm1['TN'], cm1['FP']],
                          [cm1['FN'], cm1['TP']]])
    
    sns.heatmap(cm1_array, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=ax2)
    ax2.set_title('Confusion Matrix - Dataset 1')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    # Subplot 3: Confusion Matrix Dataset 2
    ax3 = axes[2]
    cm2 = results_dataset2['confusion_matrix']
    cm2_array = np.array([[cm2['TN'], cm2['FP']],
                          [cm2['FN'], cm2['TP']]])
    
    sns.heatmap(cm2_array, annot=True, fmt='d', cmap='Oranges',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               ax=ax3)
    ax3.set_title('Confusion Matrix - Dataset 2')
    ax3.set_ylabel('Actual')
    ax3.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Metric': metrics_names + ['ROC-AUC'],
        'Dataset 1': dataset1_values + [results_dataset1.get('roc_auc', 0)],
        'Dataset 2': dataset2_values + [results_dataset2.get('roc_auc', 0)],
        'Difference': [d2 - d1 for d1, d2 in zip(
            dataset1_values + [results_dataset1.get('roc_auc', 0)],
            dataset2_values + [results_dataset2.get('roc_auc', 0)]
        )]
    })
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    return summary_df


def save_all_results(results_dataset1, results_dataset2, summary_df):
    """Save all results to files"""
    
    # Save to JSON
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset1_results': results_dataset1,
        'dataset2_results': results_dataset2,
        'summary': summary_df.to_dict()
    }
    
    with open('task2_fusion_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary to CSV
    summary_df.to_csv('task2_summary.csv', index=False)
    
    print("\nResults saved to:")
    print("  - task2_fusion_results.json")
    print("  - task2_summary.csv")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    print("\nStarting Task 2: Fusion Model Evaluation on Two Datasets")
    print("="*80)
    
    # Load Dataset 1
    X_train1, y_train1, X_test1, y_test1 = load_dataset1()
    
    # Load Dataset 2
    X_train2, y_train2, X_test2, y_test2 = load_dataset2()
    
    # Evaluate on Dataset 1
    results_dataset1 = evaluate_fusion_on_dataset(
        X_train1, y_train1, X_test1, y_test1, 
        "Dataset 1 (Original)"
    )
    
    # Evaluate on Dataset 2
    results_dataset2 = evaluate_fusion_on_dataset(
        X_train2, y_train2, X_test2, y_test2,
        "Dataset 2 (Resampled)"
    )
    
    # Create comparison visualization
    summary_df = create_comparison_visualization(results_dataset1, results_dataset2)
    
    # Save all results
    save_all_results(results_dataset1, results_dataset2, summary_df)
    
    print("\n" + "="*80)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nFinal Summary:")
    print(f"Dataset 1 - Accuracy: {results_dataset1['accuracy']:.4f}")
    print(f"Dataset 2 - Accuracy: {results_dataset2['accuracy']:.4f}")
    print(f"Average Accuracy: {(results_dataset1['accuracy'] + results_dataset2['accuracy'])/2:.4f}")
    
    return results_dataset1, results_dataset2, summary_df


if __name__ == '__main__':
    dataset1_results, dataset2_results, summary = main()