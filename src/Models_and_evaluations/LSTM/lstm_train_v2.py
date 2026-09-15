import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, precision_score
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.impute import SimpleImputer
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the enhanced LSTM model
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(2*hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context).squeeze()

# Define dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model pipeline save/load
def save_pipeline(model, preprocessor, filepath):
    """Save the complete model pipeline"""
    model_copy = copy.deepcopy(model).cpu()
    pipeline = {
        'model_state': model_copy.state_dict(),
        'model_meta': {
            'input_dim': model_copy.lstm.input_size,
            'hidden_dim': model_copy.lstm.hidden_size,
            'num_layers': model_copy.lstm.num_layers,
            'dropout': model_copy.lstm.dropout
        },
        'preprocessor': preprocessor
    }
    dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")

def load_pipeline(filepath, device='cpu'):
    """Loading the complete model pipeline"""
    pipeline = load(filepath)
    
    model = AttentionLSTM(
        input_dim=pipeline['model_meta']['input_dim'],
        hidden_dim=pipeline['model_meta']['hidden_dim'],
        num_layers=pipeline['model_meta']['num_layers'],
        dropout=pipeline['model_meta']['dropout']
    ).to(device)
    
    model.load_state_dict(pipeline['model_state'])
    model.eval()
    
    return model, pipeline['preprocessor']

# Hyperparameter optimization function
def objective(trial, X_train, y_train):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 3),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }
    
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        train_loader = DataLoader(FeatureDataset(X_train[train_idx], y_train[train_idx]), 
                                batch_size=128, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_train[val_idx], y_train[val_idx]), 
                              batch_size=256)
        
        # Initialize the model
        model = AttentionLSTM(
            input_dim=X_train.shape[1],
            **{k:v for k,v in params.items() if k in ['hidden_dim', 'num_layers', 'dropout']}
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                              lr=params['lr'],
                              weight_decay=params['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.BCEWithLogitsLoss()
        
        # Quick Training
        best_fold_auc = 0
        for epoch in range(10):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # verify
            model.eval()
            probs, labels = [], []
            with torch.no_grad():
                for inputs, batch_labels in val_loader:
                    probs.extend(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
                    labels.extend(batch_labels.numpy())
            
            fold_auc = roc_auc_score(labels, probs)
            trial.report(fold_auc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            best_fold_auc = max(best_fold_auc, fold_auc)
        
        fold_results.append(best_fold_auc)
    
    return np.mean(fold_results)

# Main training function
def train_with_best_params(X_train, y_train, best_params):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    histories = []
    best_global_auc = 0
    best_model_path = ""
    preprocessor = {'imputer': imputer, 'scaler': scaler, 'encoder': le}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\n=== Training Fold {fold+1} ===")
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        
        train_loader = DataLoader(FeatureDataset(X_train[train_idx], y_train[train_idx]),
                                batch_size=64, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_train[val_idx], y_train[val_idx]),
                              batch_size=128)
        
        # Initialize the model using optimal parameters.
        model = AttentionLSTM(
            input_dim=X_train.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(),
                              lr=best_params['lr'],
                              weight_decay=best_params['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        criterion = nn.BCEWithLogitsLoss()
        
        best_auc = 0
        early_stop_counter = 0
        
        for epoch in range(1, 31):

            model.train()
            train_loss = 0
            
            # Training steps
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Verification steps
            model.eval()
            val_loss, correct, total = 0, 0, 0
            all_probs, all_labels = [], []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs).squeeze()
                    val_loss += criterion(outputs, labels).item()
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculation indicators
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            val_auc = roc_auc_score(all_labels, all_probs)
            
            # Record Results
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Printing progress
            print(f"Epoch {epoch+1}/15 | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"AUC: {val_auc:.4f}")
            
            # Printing progress
            if val_auc > best_auc:
                
                model_path = f"best_fold{fold}.joblib"
                save_pipeline(model, preprocessor, model_path)

                if val_auc > best_global_auc:
                  best_global_auc = val_auc
                  best_model_path = model_path

                best_auc = val_auc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= 5:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
        histories.append(history)
    
    return histories, best_model_path


# Main program
if __name__ == "__main__":
    # Data loading and preprocessing
    data = pd.read_csv("D:/Education/Graduation Project/v2/url-classification-system/data/final_dataset_with_selected_features2.csv")
    feature_columns = data.columns.drop(['url', 'label'])
    X = data[feature_columns]
    y = data['label']
    
    # Pretreatment pipeline
    imputer = SimpleImputer(strategy='median').fit(X)
    X = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    le = LabelEncoder().fit(y)
    y_encoded = le.transform(y)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
   )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # Hyperparameter search phase
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    
    print(f"\nBest Params: {study.best_params}")
    print(f"Best AUC: {study.best_value:.4f}")
    
    # Final training using optimal parameters.
    histories, best_model_path = train_with_best_params(
    X_train, y_train, study.best_params
   )



    def evaluate_on_test(model, X_test, y_test):
        test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=256)
    
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
               inputs = inputs.to(device)
               outputs = model(inputs)
               probs = torch.sigmoid(outputs)

               all_probs.extend(probs.cpu().numpy())
               all_labels.extend(labels.numpy())
 
        # preds = (np.array(all_probs) > 0.5).astype(int)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(all_labels, preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, preds)
        Recall = recall_score(all_labels, preds, average='binary')
        sensitivity = Recall


        print("\n FINAL TEST RESULTS ")
        print("Accuracy:", accuracy_score(all_labels, preds))
        print("AUC:", roc_auc_score(all_labels, all_probs))
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {Recall:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")

        # Confusion Matrix 
        cm = confusion_matrix(all_labels, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=["Benign", "Malicious"],
                     yticklabels=["Benign", "Malicious"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Test Set")
        plt.show()
        plt.close()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Test Set")
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

    # Evaluate the best saved model
    best_model, _ = load_pipeline(best_model_path, device=device)

    evaluate_on_test(best_model, X_test, y_test)
  
    
    # Visualization results
    plt.figure(figsize=(15,5))
    for i, metric in enumerate(['loss', 'acc', 'auc']):
        plt.subplot(1,3,i+1)
        for fold, hist in enumerate(histories):
            plt.plot(hist[f'val_{metric}'], label=f'Fold {fold+1}')
        plt.title(f'Validation {metric.upper()}')
        plt.xlabel('Epoch')
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    plt.close()

    # ===== Plot Train & Validation Loss =====
    plt.figure(figsize=(7,5))

    for fold, hist in enumerate(histories):
        plt.plot(hist['train_loss'], linestyle='--', label=f'Fold {fold+1} Train Loss')
        plt.plot(hist['val_loss'], label=f'Fold {fold+1} Val Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()
