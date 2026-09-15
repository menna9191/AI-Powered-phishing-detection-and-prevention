import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, \
    auc
from sklearn.preprocessing import StandardScaler
import joblib
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class PhishingURLClassifier:
    def __init__(self, csv_path):
        """Initialize the classifier with the path to the dataset"""
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.scaler = None

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            # Load data
            print("Loading dataset...")
            self.data = pd.read_csv(self.csv_path)

            # Display basic information
            print(f"Dataset shape: {self.data.shape}")
            print(f"Class distribution:\n{self.data['label'].value_counts(normalize=True)}")

            # Split features and target
            X = self.data.drop('label', axis=1)

            # Also drop 'url' column as it's not a feature
            if 'url' in X.columns:
                X = X.drop('url', axis=1)

            y = self.data['label']

            # Split data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

            print(f"Training set size: {self.X_train.shape[0]}")
            print(f"Testing set size: {self.X_test.shape[0]}")

        except Exception as e:
            print(f"Error in data loading and preprocessing: {str(e)}")
            raise

        return self

    def train_models(self):
        """Train and evaluate multiple models"""
        try:
            # Define models to train
            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
            }

            # Train and evaluate each model
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Check if binary or multiclass problem
            n_classes = len(np.unique(self.y_train))
            # Set average method for metrics if multiclass
            avg_method = 'binary' if n_classes == 2 else 'weighted'

            for name, model in models.items():
                print(f"\nTraining {name}...")
                start_time = time.time()

                # Perform cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted')

                # Train on full training set
                model.fit(self.X_train, self.y_train)

                # Predict on test set
                y_pred = model.predict(self.X_test)

                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average=avg_method)
                recall = recall_score(self.y_test, y_pred, average=avg_method)
                f1 = f1_score(self.y_test, y_pred, average=avg_method)

                # Calculate ROC curve and AUC (for binary classification)
                auc_score = 0
                if n_classes == 2 and hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                    auc_score = auc(fpr, tpr)

                # Store results
                train_time = time.time() - start_time
                self.results[name] = {
                    'model': model,
                    'cv_f1': cv_scores.mean(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score,
                    'train_time': train_time
                }

                print(
                    f"{name} - CV F1: {cv_scores.mean():.4f}, Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {train_time:.2f}s")

            # Store models
            self.models = {name: info['model'] for name, info in self.results.items()}

            # Find best model based on F1 score
            best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
            self.best_model = self.results[best_model_name]['model']
            self.best_model_name = best_model_name

            print(f"\nBest model: {best_model_name} with F1 score: {self.results[best_model_name]['f1']:.4f}")

            # Display detailed results for best model
            y_pred = self.best_model.predict(self.X_test)
            print("\nClassification Report for Best Model:")
            print(classification_report(self.y_test, y_pred))

        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

        return self

    def plot_heatmap(self):
        """Plot a heatmap of the evaluation metrics for each model"""
        try:
            # Create a DataFrame from the results
            metrics_df = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1', 'auc']]

            metrics_df = metrics_df.apply(pd.to_numeric, errors='coerce')
            # Plot the heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(metrics_df, annot=True, fmt=".4f", cmap="YlGnBu")
            plt.title("Model Evaluation Metrics")
            plt.show()

        except Exception as e:
            print(f"Error in plotting: {str(e)}")

        return self

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        try:
            print(f"\nPerforming hyperparameter tuning for {self.best_model_name}...")

            # Define hyperparameter grid for each model type
            param_grids = {
                'Decision Tree': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear', 'poly']
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'num_leaves': [31, 50, 70],
                    'min_child_samples': [5, 10, 20],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }

            # Get parameter grid for the best model
            param_grid = param_grids.get(self.best_model_name, {})

            if not param_grid:
                print("No parameter grid defined for this model type.")
                return self

            # Check if binary or multiclass problem
            n_classes = len(np.unique(self.y_train))
            # Set scoring method - for binary classification, use 'f1', not 'f1_binary'
            scoring = 'f1' if n_classes == 2 else 'f1_weighted'

            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=self.best_model,
                param_grid=param_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1
            )

            # Perform grid search
            grid_search.fit(self.X_train, self.y_train)

            # Get best parameters and model
            best_params = grid_search.best_params_
            self.best_model = grid_search.best_estimator_

            print(f"Best parameters: {best_params}")

            # Evaluate tuned model
            y_pred = self.best_model.predict(self.X_test)

            # Set average method for metrics if multiclass
            avg_method = 'binary' if n_classes == 2 else 'weighted'
            tuned_f1 = f1_score(self.y_test, y_pred, average=avg_method)

            print(f"Tuned model F1 score: {tuned_f1:.4f}")

            # Update results for the best model
            self.results[self.best_model_name]['tuned_f1'] = tuned_f1
            self.results[self.best_model_name]['best_params'] = best_params

        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")

        return self

    def save_model(self, filepath='phishing_url_classifier.joblib'):
        """Save the best model to disk"""
        try:
            # Save the model and scaler
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'model_name': self.best_model_name
            }

            joblib.dump(model_data, filepath)
            print(f"\nBest model and preprocessing objects saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

        return self

    def run_full_pipeline(self):
        """Run the complete pipeline from data loading to model saving"""
        try:
            (self
             .load_and_preprocess_data()
             .train_models()
             .plot_heatmap()
             .hyperparameter_tuning()
             .save_model()
             )
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")

        return self


if __name__ == "__main__":
    # Initialize and run the classifier
    classifier = PhishingURLClassifier('final_dataset_with_selected_features.csv')
    classifier.run_full_pipeline()