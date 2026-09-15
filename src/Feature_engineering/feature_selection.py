import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


# Load the dataset
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")

    # Separate features and target
    X = df.drop(['url', 'label'], axis=1)
    y = df['label']

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y, df  # Return original dataframe too


# STEP 1: Eliminate zero/near-zero variance features
def remove_low_variance_features(X, threshold=0.001):
    print(f"\n=== STEP 1: Removing Zero/Near-Zero Variance Features (threshold={threshold}) ===")

    # Calculate variance for each feature
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    # Get mask of selected features
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    removed_features = X.columns[~selected_mask]

    # Create new dataframe with selected features
    X_selected = X[selected_features]

    print(f"Removed {len(removed_features)} features with variance < {threshold}")
    print(f"Remaining features: {len(selected_features)}")

    if len(removed_features) > 0:
        print(f"Examples of removed features: {list(removed_features)[:5]}")

    return X_selected, selected_features, removed_features


# STEP 2: Remove highly correlated features
def remove_correlated_features(X, threshold=0.95):
    print(f"\n=== STEP 2: Removing Highly Correlated Features (threshold={threshold}) ===")

    # Calculate correlation matrix
    corr_matrix = X.corr().abs()

    # Create a mask for the upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    # Keep remaining features
    X_selected = X.drop(to_drop, axis=1)
    selected_features = X_selected.columns

    print(f"Removed {len(to_drop)} highly correlated features (correlation > {threshold})")
    print(f"Remaining features: {len(selected_features)}")

    if len(to_drop) > 0:
        print(f"Examples of removed features: {to_drop[:5]}")

        # Find examples of highly correlated pairs
        examples = []
        for col in to_drop[:5]:
            # Find correlations above threshold
            high_corr = corr_matrix[col][corr_matrix[col] > threshold].index.tolist()
            # Remove self-correlation
            high_corr = [h for h in high_corr if h != col]
            if high_corr:
                examples.append(f"{col} correlated with {high_corr[0]} (r={corr_matrix.loc[col, high_corr[0]]:.3f})")

        if examples:
            print("Examples of correlations:")
            for ex in examples:
                print(f"  - {ex}")

    return X_selected, selected_features, to_drop


# STEP 3: Use gradient boosting feature importance for non-linear relationships
def select_by_feature_importance(X, y, importance_threshold=None, top_percent=0.8):
    print(f"\n=== STEP 3: Selecting Features by Gradient Boosting Importance ===")

    # Train GradientBoosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X, y)

    # Get feature importances
    importances = gb.feature_importances_

    # Create DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Select features based on importance threshold or top percentage
    if importance_threshold is not None:
        print(f"Selecting features with importance > {importance_threshold}")
        selected_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature']
    else:
        # Select top% of features by importance
        n_features = max(1, int(len(X.columns) * top_percent))
        print(f"Selecting top {top_percent * 100:.0f}% of features by importance (n={n_features})")
        selected_features = feature_importance.head(n_features)['Feature']

    # Get selected data
    X_selected = X[selected_features]
    removed_features = [f for f in X.columns if f not in selected_features.values]

    print(f"Removed {len(removed_features)} features based on importance filtering")
    print(f"Remaining features: {len(selected_features)}")

    if len(selected_features) > 0:
        top_features = feature_importance.head(5)
        print("Top 5 most important features:")
        for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance']), 1):
            print(f"  {i}. {feature}: {importance:.4f}")

    return X_selected, selected_features.values, removed_features, feature_importance


# Save filtered dataset to CSV
def save_filtered_dataset(X, y, original_df, filename='final_dataset_with_selected_features.csv'):
    print(f"\n=== Saving Filtered Dataset to {filename} ===")

    # Create a new DataFrame with selected features
    filtered_df = X.copy()

    # Add the target column
    filtered_df['label'] = y

    # Add the URL column if it was in the original dataset
    if 'url' in original_df.columns:
        filtered_df['url'] = original_df['url']

    # Reorder columns to have url first, then label, then features
    feature_cols = [col for col in filtered_df.columns if col not in ['url', 'label']]
    if 'url' in filtered_df.columns:
        new_column_order = ['url', 'label'] + feature_cols
    else:
        new_column_order = ['label'] + feature_cols

    filtered_df = filtered_df[new_column_order]

    # Save to CSV
    filtered_df.to_csv(filename, index=False)

    print(f"Saved filtered dataset with {X.shape[1]} features and {X.shape[0]} samples.")
    print(
        f"Original dataset had {original_df.shape[1] - 1 - ('url' in original_df.columns)} features (excluding 'label' and 'url').")
    feature_reduction = original_df.shape[1] - 1 - ('url' in original_df.columns) - X.shape[1]
    reduction_percent = (feature_reduction / (original_df.shape[1] - 1 - ('url' in original_df.columns))) * 100
    print(f"Reduced features by {feature_reduction} ({reduction_percent:.1f}%).")
    print(f"Column order in saved CSV: url (first), label (second), followed by {len(feature_cols)} selected features")

    return filtered_df


# Visualize feature counts after each step
def plot_feature_reduction(feature_counts, step_names, output_file='feature_reduction.png'):
    plt.figure(figsize=(12, 8))

    # Create step labels
    x_labels = ['Original'] + step_names

    # Create bar plot with gradient colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(feature_counts)))
    bars = plt.bar(x_labels, feature_counts, color=colors)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Add percentage reduction labels
    for i in range(1, len(feature_counts)):
        prev_count = feature_counts[i - 1]
        curr_count = feature_counts[i]
        reduction_pct = ((prev_count - curr_count) / prev_count) * 100
        plt.text(i - 0.05, curr_count / 2,
                 f'-{reduction_pct:.1f}%', ha='center', va='center',
                 rotation=90, color='white', fontweight='bold')

    plt.title('Feature Reduction Across Pipeline Steps', fontsize=16)
    plt.ylabel('Number of Features', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature reduction plot to {output_file}")
    return plt


# Visualize correlation matrix
def plot_correlation_heatmap(X, selected_features=None, n_features=15, output_file='correlation_matrix.png'):
    # If specific features are provided, use those, otherwise use all features
    if selected_features is not None and len(selected_features) > 0:
        if len(selected_features) > n_features:
            print(f"Showing correlation heatmap for top {n_features} features")
            features_to_plot = selected_features[:n_features]
        else:
            features_to_plot = selected_features
    else:
        if X.shape[1] > n_features:
            print(f"Too many features to display. Showing top {n_features} features")
            features_to_plot = X.columns[:n_features]
        else:
            features_to_plot = X.columns

    # Calculate correlation for selected features
    corr = X[features_to_plot].corr()

    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {output_file}")
    return plt


# Visualize feature importance
def plot_feature_importance(importance_df, top_n=20, output_file='feature_importance.png'):
    plt.figure(figsize=(14, 10))

    # Sort by importance and take top N features
    plot_data = importance_df.sort_values('Importance', ascending=False).head(top_n)

    # Create horizontal bar plot
    bars = plt.barh(plot_data['Feature'], plot_data['Importance'])

    # Add importance values as labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        plt.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                 va='center')

    plt.title('Feature Importance from Gradient Boosting', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {output_file}")
    return plt


def main():
    # Load dataset
    file_path = 'final_dataset_with_features.csv'
    X, y, original_df = load_data(file_path)

    # Track original feature count
    original_feature_count = X.shape[1]

    # STEP 1: Remove low variance features
    X_var, var_selected, var_removed = remove_low_variance_features(X, threshold=0.001)

    # STEP 2: Remove highly correlated features
    X_corr, corr_selected, corr_removed = remove_correlated_features(X_var, threshold=0.95)

    # Plot correlation heatmap
    plot_correlation_heatmap(X_corr, output_file='correlation_matrix.png')

    # STEP 3: Select features by importance
    X_imp, imp_selected, imp_removed, feature_importance = select_by_feature_importance(
        X_corr, y, top_percent=0.8
    )

    # Plot feature importance
    plot_feature_importance(feature_importance, output_file='feature_importance.png')

    # Visualize feature reduction across all steps
    feature_counts = [
        original_feature_count,
        len(var_selected),
        len(corr_selected),
        len(imp_selected),
    ]

    step_names = [
        'After Variance Threshold',
        'After Correlation Analysis',
        'After Feature Importance'
    ]

    plot_feature_reduction(feature_counts, step_names, output_file='feature_reduction.png')

    # Save the filtered dataset to a CSV file
    save_filtered_dataset(X_imp, y, original_df, filename='final_dataset_with_selected_features.csv')


if __name__ == "__main__":
    main()