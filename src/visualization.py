import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc

def plot_feature_distributions(df, output_dir):
    """
    Plot distributions of features in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    output_dir : str
        Directory to save the visualizations
    """
    print("Plotting feature distributions...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get numerical features
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Class' in numerical_cols:
        numerical_cols.remove('Class')
    
    # Plot histograms for numerical features
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, n_rows * 4))
    
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_distributions.png'))
    plt.close()
    
    # Plot box plots by class
    if 'Class' in df.columns:
        n_cols = 2
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(16, n_rows * 5))
        
        for i, col in enumerate(numerical_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.boxplot(x='Class', y=col, data=df)
            plt.title(f'{col} by Class')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_boxplots_by_class.png'))
        plt.close()
    
    print(f"Saved feature distribution plots to {figures_dir}")

def plot_correlation_matrix(df, output_dir):
    """
    Plot correlation matrix of features
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    output_dir : str
        Directory to save the visualization
    """
    print("Plotting correlation matrix...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get numerical features
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Calculate correlation matrix
    corr = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"Saved correlation matrix plot to {figures_dir}")

def plot_pca_components(df_pca, output_dir):
    """
    Plot PCA components
    
    Parameters:
    -----------
    df_pca : pd.DataFrame
        The PCA-transformed dataset
    output_dir : str
        Directory to save the visualization
    """
    print("Plotting PCA components...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if we have 'Class' and at least 2 PCA components
    if 'Class' not in df_pca.columns or 'PC1' not in df_pca.columns or 'PC2' not in df_pca.columns:
        print("Warning: Cannot plot PCA components (missing columns)")
        return
    
    # Plot first two PCA components
    plt.figure(figsize=(10, 8))
    
    # Use a color palette suitable for the number of classes
    unique_classes = df_pca['Class'].unique()
    colors = sns.color_palette("tab10", len(unique_classes))
    
    # Plot each class
    for i, cls in enumerate(unique_classes):
        idx = df_pca['Class'] == cls
        plt.scatter(df_pca.loc[idx, 'PC1'], df_pca.loc[idx, 'PC2'], 
                    color=colors[i], alpha=0.7, label=f'Class {cls}')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Principal Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pca_components_2d.png'))
    plt.close()
    
    # If we have a third component, plot in 3D
    if 'PC3' in df_pca.columns:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, cls in enumerate(unique_classes):
            idx = df_pca['Class'] == cls
            ax.scatter(df_pca.loc[idx, 'PC1'], df_pca.loc[idx, 'PC2'], df_pca.loc[idx, 'PC3'],
                      color=colors[i], alpha=0.7, label=f'Class {cls}')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA: 3D Visualization')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'pca_components_3d.png'))
        plt.close()
    
    print(f"Saved PCA component plots to {figures_dir}")

def plot_lda_components(df_lda, output_dir):
    """
    Plot LDA components
    
    Parameters:
    -----------
    df_lda : pd.DataFrame
        The LDA-transformed dataset
    output_dir : str
        Directory to save the visualization
    """
    print("Plotting LDA components...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if we have 'Class' and at least 2 LDA components
    if 'Class' not in df_lda.columns or 'LD1' not in df_lda.columns or 'LD2' not in df_lda.columns:
        print("Warning: Cannot plot LDA components (missing columns)")
        return
    
    # Plot first two LDA components
    plt.figure(figsize=(10, 8))
    
    # Use a color palette suitable for the number of classes
    unique_classes = df_lda['Class'].unique()
    colors = sns.color_palette("tab10", len(unique_classes))
    
    # Plot each class
    for i, cls in enumerate(unique_classes):
        idx = df_lda['Class'] == cls
        plt.scatter(df_lda.loc[idx, 'LD1'], df_lda.loc[idx, 'LD2'], 
                    color=colors[i], alpha=0.7, label=f'Class {cls}')
    
    plt.xlabel('First Discriminant')
    plt.ylabel('Second Discriminant')
    plt.title('LDA: First Two Discriminants')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lda_components_2d.png'))
    plt.close()
    
    # If we have a third component, plot in 3D
    if 'LD3' in df_lda.columns:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, cls in enumerate(unique_classes):
            idx = df_lda['Class'] == cls
            ax.scatter(df_lda.loc[idx, 'LD1'], df_lda.loc[idx, 'LD2'], df_lda.loc[idx, 'LD3'],
                      color=colors[i], alpha=0.7, label=f'Class {cls}')
        
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
        ax.set_title('LDA: 3D Visualization')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'lda_components_3d.png'))
        plt.close()
    
    print(f"Saved LDA component plots to {figures_dir}")

def plot_roc_curves(results, output_dir):
    """
    Plot ROC curves for all classifiers and data representations
    
    Parameters:
    -----------
    results : dict
        Dictionary with model evaluation results
    output_dir : str
        Directory to save the visualizations
    """
    print("Plotting ROC curves...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Iterate over data representations
    for data_rep_name, classifiers in results['all_results'].items():
        
        # Iterate over classifiers
        for classifier_name, result in classifiers.items():
            roc_data = result.get('best_roc_data')
            
            if roc_data and roc_data.get('fpr') and roc_data.get('tpr') and roc_data.get('roc_auc'):
                plt.figure(figsize=(10, 8))
                
                # Plot ROC curve for each class
                for i, (fpr, tpr, roc_auc) in enumerate(zip(
                    roc_data['fpr'].values(), 
                    roc_data['tpr'].values(), 
                    roc_data['roc_auc'].values()
                )):
                    if i == 'micro':
                        plt.plot(fpr, tpr, label=f'Micro-average (AUC = {roc_auc:.2f})')
                    else:
                        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {classifier_name} on {data_rep_name.upper()} data')
                plt.legend(loc="lower right")
                plt.tight_layout()
                
                plt.savefig(os.path.join(figures_dir, f'roc_{classifier_name}_{data_rep_name}.png'))
                plt.close()
    
    print(f"Saved ROC curve plots to {figures_dir}")

def visualize_model_performance(performance_metrics, output_dir):
    """
    Visualize model performance metrics
    
    Parameters:
    -----------
    performance_metrics : pd.DataFrame
        DataFrame with performance metrics
    output_dir : str
        Directory to save the visualizations
    """
    print("Visualizing model performance...")
    
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Bar plot of accuracy by model and data representation
    plt.figure(figsize=(14, 8))
    
    # Pivot data for grouped bar plot
    pivot_acc = performance_metrics.pivot(index='Classifier', columns='Data Representation', values='Accuracy (Mean)')
    pivot_acc.plot(kind='bar', yerr=performance_metrics.pivot(
        index='Classifier', columns='Data Representation', values='Accuracy (Std)'
    ), capsize=5, rot=0, ax=plt.gca())
    
    plt.title('Model Accuracy by Data Representation')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)  # Adjust based on your actual accuracies
    plt.legend(title='Data Representation')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_accuracy.png'))
    plt.close()
    
    # 2. Bar plot of F1 score by model and data representation
    plt.figure(figsize=(14, 8))
    
    pivot_f1 = performance_metrics.pivot(index='Classifier', columns='Data Representation', values='F1 Score (Mean)')
    pivot_f1.plot(kind='bar', yerr=performance_metrics.pivot(
        index='Classifier', columns='Data Representation', values='F1 Score (Std)'
    ), capsize=5, rot=0, ax=plt.gca())
    
    plt.title('Model F1 Score by Data Representation')
    plt.ylabel('F1 Score')
    plt.ylim(0.7, 1.0)  # Adjust based on your actual F1 scores
    plt.legend(title='Data Representation')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_f1_score.png'))
    plt.close()
    
    # 3. Heatmap of all metrics
    plt.figure(figsize=(16, 12))
    
    # Create a new DataFrame with multi-index for the heatmap
    heatmap_data = performance_metrics.copy()
    heatmap_data['Model_Data'] = heatmap_data['Classifier'] + ' + ' + heatmap_data['Data Representation']
    heatmap_data = heatmap_data.set_index('Model_Data')
    heatmap_data = heatmap_data.drop(columns=['Classifier', 'Data Representation'])
    heatmap_data = heatmap_data.reindex(columns=[
        'Accuracy (Mean)', 'Precision (Mean)', 'Recall (Mean)', 'F1 Score (Mean)'
    ])
    
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    plt.title('Performance Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'performance_heatmap.png'))
    plt.close()
    
    print(f"Saved model performance visualizations to {figures_dir}")

def create_visualizations(data_representations, results, output_dir):
    """
    Create all visualizations for the project
    
    Parameters:
    -----------
    data_representations : dict
        Dictionary with different data representations
    results : dict
        Dictionary with model evaluation results
    output_dir : str
        Directory to save the visualizations
    """
    print("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature distributions for raw data
    if 'raw' in data_representations:
        plot_feature_distributions(data_representations['raw']['data'], output_dir)
    
    # 2. Correlation matrix for raw data
    if 'raw' in data_representations:
        plot_correlation_matrix(data_representations['raw']['data'], output_dir)
    
    # 3. PCA components
    if 'pca' in data_representations:
        plot_pca_components(data_representations['pca']['data'], output_dir)
    
    # 4. LDA components
    if 'lda' in data_representations:
        plot_lda_components(data_representations['lda']['data'], output_dir)
    
    # 5. ROC curves
    plot_roc_curves(results, output_dir)
    
    # 6. Model performance visualizations
    visualize_model_performance(results['performance_metrics'], output_dir)
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    # This allows this module to be run as a standalone script for testing
    from data_preprocessing import preprocess_data
    from feature_engineering import feature_engineering
    from model_training import train_models
    
    input_file = "data/raw/Dry_Bean_Dataset.xlsx"
    processed_dir = "data/processed"
    output_dir = "output"
    
    # First, preprocess the data
    df_preprocessed, _ = preprocess_data(input_file, processed_dir)
    
    # Then, apply feature engineering
    data_representations = feature_engineering(df_preprocessed, processed_dir)
    
    # Train and evaluate models
    results = train_models(data_representations, output_dir)
    
    # Create visualizations
    create_visualizations(data_representations, results, output_dir)
    print("Visualization process completed successfully!")