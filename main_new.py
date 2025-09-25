# ============================================================
# DP-NAS AND RANPAC INDIVIDUAL EVALUATION
# WITH EXTENDED METRICS (PRECISION, RECALL, F1, CONFUSION MATRIX)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import json

print("="*80)
print("TASK 1: INDIVIDUAL MODEL EVALUATION WITH EXTENDED METRICS")
print("="*80)

# ============================================================
# PART 1: SETUP AND DATA LOADING
# ============================================================

class MetricsCalculator:
    """Calculate and display comprehensive metrics"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None, model_name=""):
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
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC-AUC if probabilities available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = None
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics in formatted way"""
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {metrics['model']}")
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
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """Plot confusion matrix heatmap"""
        if 'confusion_matrix' not in metrics:
            return
        
        cm = metrics['confusion_matrix']
        cm_array = np.array([[cm['TN'], cm['FP']],
                            [cm['FN'], cm['TP']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {metrics["model"]}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()


class OptimizedDataset(Dataset):
    """Dataset class for EMG data"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_emg_data():
    """Load and prepare EMG data"""
    print("\nLoading EMG data...")
    
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
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================
# PART 2: DP-NAS MODEL AND EVALUATION
# ============================================================

class DPNASModel(nn.Module):
    """DP-NAS model with configurable architecture"""
    
    def __init__(self, input_dim, hidden_dims, num_layers, dropout, use_bn, activation='relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dims))
            
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dims))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dims
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def evaluate_dpnas(X_train, y_train, X_test, y_test):
    """Train and evaluate DP-NAS model"""
    print("\n" + "="*80)
    print("EVALUATING DP-NAS MODEL")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = OptimizedDataset(X_train, y_train)
    test_dataset = OptimizedDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Best architecture found from previous experiments
    model = DPNASModel(
        input_dim=X_train.shape[1],
        hidden_dims=256,
        num_layers=3,
        dropout=0.2,
        use_bn=True,
        activation='relu'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining DP-NAS...")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if (epoch + 1) % 5 == 0:
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        model_name='DP-NAS'
    )
    
    metrics_calc.print_metrics(metrics)
    metrics_calc.plot_confusion_matrix(metrics)
    
    return metrics


# ============================================================
# PART 3: RANPAC MODEL AND EVALUATION
# ============================================================

class SimpleRanPAC:
    """Simplified RanPAC for testing"""
    
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


def evaluate_ranpac(X_train, y_train, X_test, y_test):
    """Train and evaluate RanPAC model"""
    print("\n" + "="*80)
    print("EVALUATING RANPAC MODEL")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize RanPAC
    ranpac = SimpleRanPAC(X_train.shape[1], 2, device)
    
    print("\nTraining RanPAC (incremental learning)...")
    
    # Incremental training
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"  Epoch {epoch+1}/{num_epochs}")
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        
        # Train incrementally
        for i in indices[:len(indices)//10]:  # Use subset for faster training
            ranpac.fit(
                torch.tensor(X_train[i]).to(device),
                torch.tensor(y_train[i]).to(device)
            )
    
    # Complete training with all data
    print("  Final training pass...")
    for i in range(len(X_train)):
        ranpac.fit(
            torch.tensor(X_train[i]).to(device),
            torch.tensor(y_train[i]).to(device)
        )
    
    # Evaluation
    print("\nEvaluating on test set...")
    
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_test_tensor = torch.tensor(y_test).long().to(device)
    
    # Get predictions
    probs = ranpac.predict(X_test_tensor, return_probs=True)
    preds = torch.argmax(probs, dim=1)
    probs_positive = probs[:, 1]  # Probability of positive class
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(
        y_true=y_test_tensor.cpu().numpy(),
        y_pred=preds.cpu().numpy(),
        y_prob=probs_positive.cpu().numpy(),
        model_name='RanPAC'
    )
    
    metrics_calc.print_metrics(metrics)
    metrics_calc.plot_confusion_matrix(metrics)
    
    return metrics


# ============================================================
# PART 4: COMPARISON AND SUMMARY
# ============================================================

def create_comparison_table(dpnas_metrics, ranpac_metrics):
    """Create comparison table of results"""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC'],
        'DP-NAS': [
            dpnas_metrics['accuracy'],
            dpnas_metrics['precision'],
            dpnas_metrics['recall'],
            dpnas_metrics['f1_score'],
            dpnas_metrics.get('specificity', 0),
            dpnas_metrics.get('roc_auc', 0)
        ],
        'RanPAC': [
            ranpac_metrics['accuracy'],
            ranpac_metrics['precision'],
            ranpac_metrics['recall'],
            ranpac_metrics['f1_score'],
            ranpac_metrics.get('specificity', 0),
            ranpac_metrics.get('roc_auc', 0)
        ]
    })
    
    # Format as percentages
    for col in ['DP-NAS', 'RanPAC']:
        comparison[f'{col} (%)'] = comparison[col].apply(lambda x: f"{x*100:.2f}%")
    
    print("\nDetailed Comparison:")
    print(comparison[['Metric', 'DP-NAS (%)', 'RanPAC (%)']].to_string(index=False))
    
    # Identify winner for each metric
    print("\nBest Model per Metric:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']:
        dpnas_val = comparison.loc[comparison['Metric'] == metric, 'DP-NAS'].values[0]
        ranpac_val = comparison.loc[comparison['Metric'] == metric, 'RanPAC'].values[0]
        
        if dpnas_val > ranpac_val:
            winner = 'DP-NAS'
            value = dpnas_val
        else:
            winner = 'RanPAC'
            value = ranpac_val
        
        print(f"  {metric}: {winner} ({value*100:.2f}%)")
    
    return comparison


def save_results(dpnas_metrics, ranpac_metrics):
    """Save results to files"""
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dpnas': dpnas_metrics,
        'ranpac': ranpac_metrics
    }
    
    # Save to JSON
    with open('task1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to task1_results.json")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    print("\nStarting Task 1: Individual Model Evaluation")
    print("="*80)
    
    # Load data
    X_train, y_train, X_test, y_test = load_emg_data()
    
    # Evaluate DP-NAS
    dpnas_metrics = evaluate_dpnas(X_train, y_train, X_test, y_test)
    
    # Evaluate RanPAC
    ranpac_metrics = evaluate_ranpac(X_train, y_train, X_test, y_test)
    
    # Compare results
    comparison = create_comparison_table(dpnas_metrics, ranpac_metrics)
    
    # Save results
    save_results(dpnas_metrics, ranpac_metrics)
    
    print("\n" + "="*80)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return dpnas_metrics, ranpac_metrics, comparison


if __name__ == '__main__':
    dpnas_results, ranpac_results, comparison_table = main()