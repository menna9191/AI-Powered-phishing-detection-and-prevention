import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Dataset class for handling phishing data
class PhishingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# CNN model for URL classification
class PhishingCNN(nn.Module):
    def __init__(self, input_dim):
        super(PhishingCNN, self).__init__()

        # Calculate grid size for reshaping features into a 2D grid
        self.grid_size = int(np.ceil(np.sqrt(input_dim)))
        self.padded_size = self.grid_size ** 2

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate output size after convolutions
        conv_output_size = self.grid_size // 4  # After two max-pooling with kernel size 2
        conv_output_dim = 64 * conv_output_size * conv_output_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_dim, 128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape features to 2D grid
        batch_size = x.size(0)
        padded = torch.zeros(batch_size, self.padded_size, device=x.device)
        padded[:, :x.size(1)] = x
        x = padded.view(batch_size, 1, self.grid_size, self.grid_size)

        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        # Ensure output has proper dimensions
        return x.view(batch_size, -1)


# Load and preprocess data
def load_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Number of phishing URLs: {sum(df['label'] == 1)}")
    print(f"Number of legitimate URLs: {sum(df['label'] == 0)}")

    # Drop the URL column as it's not a numerical feature
    X = df.drop(['url', 'label'], axis=1)
    y = df['label']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, list(X.columns)


# Train model with cross-validation
def train_with_cross_validation(X, y, n_splits=5, batch_size=32, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create datasets and dataloaders
        train_dataset = PhishingDataset(X_train, y_train)
        val_dataset = PhishingDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        model = PhishingCNN(input_dim=X.shape[1]).to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs = model(features).view(-1)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)

                    outputs = model(features).view(-1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Save predictions and true labels
                    val_preds.extend(outputs.detach().cpu().numpy())
                    val_true.extend(labels.detach().cpu().numpy())

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Final evaluation
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)

        # Calculate metrics
        val_accuracy = accuracy_score(val_true, val_preds_binary)
        val_precision = precision_score(val_true, val_preds_binary, zero_division=0)
        val_recall = recall_score(val_true, val_preds_binary, zero_division=0)
        val_f1 = f1_score(val_true, val_preds_binary, zero_division=0)
        val_auc = roc_auc_score(val_true, val_preds)

        print(f"Fold {fold + 1} Results:")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        print(f"AUC: {val_auc:.4f}")

        # Save results
        fold_results.append({
            'model': model,
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        })

    # Find the best model based on validation F1 score
    best_fold = max(range(len(fold_results)), key=lambda i: fold_results[i]['f1'])
    best_model = fold_results[best_fold]['model']

    print(f"\nBest model is from fold {best_fold + 1}")
    print(f"Accuracy: {fold_results[best_fold]['accuracy']:.4f}")
    print(f"Precision: {fold_results[best_fold]['precision']:.4f}")
    print(f"Recall: {fold_results[best_fold]['recall']:.4f}")
    print(f"F1 Score: {fold_results[best_fold]['f1']:.4f}")
    print(f"AUC: {fold_results[best_fold]['auc']:.4f}")

    # Plot training and validation loss curves for the best model
    plt.figure(figsize=(10, 6))
    plt.plot(fold_results[best_fold]['train_losses'], label='Training Loss')
    plt.plot(fold_results[best_fold]['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses (Best Model)')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

    # Plot training and validation accuracy curves for the best model
    plt.figure(figsize=(10, 6))
    plt.plot(fold_results[best_fold]['train_accuracies'], label='Training Accuracy')
    plt.plot(fold_results[best_fold]['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies (Best Model)')
    plt.legend()
    plt.savefig('accuracy_curves.png')
    plt.close()

    return best_model, fold_results


# Evaluate model on the test set
def evaluate_model(model, X_test, y_test, batch_size=32):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    test_dataset = PhishingDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluation
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features).view(-1)

            # Save predictions and true labels
            y_pred.extend(outputs.detach().cpu().numpy())
            y_true.extend(labels.detach().cpu().numpy())

    # Convert to binary predictions
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

    # Calculate metrics
    test_accuracy = accuracy_score(y_true, y_pred_binary)
    test_precision = precision_score(y_true, y_pred_binary, zero_division=0)
    test_recall = recall_score(y_true, y_pred_binary, zero_division=0)
    test_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    test_auc = roc_auc_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm
    }


# Main function
def main():
    # File path
    file_path = '../data/final_dataset_with_features.csv'

    # Create output directory for results
    os.makedirs('../results', exist_ok=True)

    # Load and preprocess data
    X, y, feature_names = load_preprocess_data(file_path)

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model with cross-validation
    best_model, fold_results = train_with_cross_validation(
        X_train, y_train,
        n_splits=5,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001
    )

    # Evaluate on test set
    test_results = evaluate_model(best_model, X_test, y_test)

    # Save the model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_dim': X.shape[1],
        'feature_names': feature_names,
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1'],
        'test_auc': test_results['auc'],
    }, '../results/phishing_cnn_model.pth')

    print("\nModel saved to 'phishing_cnn_model.pth'")


if __name__ == "__main__":
    main()