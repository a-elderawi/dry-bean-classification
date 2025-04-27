import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D

def apply_pca(df, output_dir=None):
    """
    Apply PCA for dimensionality reduction
    
    Parameters:
    -----------
    df : pd.DataFrame
        The preprocessed dataset
    output_dir : str, optional
        Directory to save the output visualizations
        
    Returns:
    --------
    pd.DataFrame, PCA
        The PCA-transformed data and the PCA object
    """
    print("Applying PCA for dimensionality reduction...")
    
    # Separate features and target
    X = df.drop('Class', axis=1) if 'Class' in df.columns else df
    y = df['Class'] if 'Class' in df.columns else None
    
    # Initialize and fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    
    # Determine number of components based on explained variance
    avg_var = 1.0 / len(explained_variance)
    n_components = sum(explained_variance > avg_var)
    print(f"Number of PCA components selected: {n_components} (average explained variance threshold: {avg_var:.4f})")
    
    # Create PCA with selected number of components
    pca_selected = PCA(n_components=n_components)
    X_pca_selected = pca_selected.fit_transform(X)
    
    # Create DataFrame with PCA components
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca_selected, columns=pca_cols)
    
    # Add back the target column if it exists
    if y is not None:
        df_pca['Class'] = y.values
    
    # Visualization: PCA explained variance
    if output_dir:
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Visualize explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.axhline(y=avg_var, color='r', linestyle='--', label=f'Average Variance ({avg_var:.4f})')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'pca_explained_variance.png'))
        plt.close()
        
        # Visualize first two components
        if y is not None and len(np.unique(y)) <= 10:  # only if we have a reasonable number of classes
            plt.figure(figsize=(10, 8))
            
            # Use a color palette suitable for the number of classes
            unique_classes = np.unique(y)
            colors = sns.color_palette("tab10", len(unique_classes))
            
            # Plot each class
            for i, cls in enumerate(unique_classes):
                idx = y == cls
                plt.scatter(X_pca[idx, 0], X_pca[idx, 1], color=colors[i], alpha=0.7, label=f'Class {cls}')
            
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA: First Two Principal Components')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'pca_visualization.png'))
            plt.close()
    
    return df_pca, pca_selected

def apply_lda(df, output_dir=None):
    """
    Apply LDA for dimensionality reduction
    
    Parameters:
    -----------
    df : pd.DataFrame
        The preprocessed dataset
    output_dir : str, optional
        Directory to save the output visualizations
        
    Returns:
    --------
    pd.DataFrame, LDA
        The LDA-transformed data and the LDA object
    """
    print("Applying LDA for dimensionality reduction...")
    
    # Separate features and target
    X = df.drop('Class', axis=1) if 'Class' in df.columns else df
    y = df['Class'] if 'Class' in df.columns else None
    
    if y is None:
        raise ValueError("LDA requires a target variable 'Class' in the dataset")
    
    # Initialize and fit LDA
    # Number of components is min(n_classes - 1, n_features)
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    n_components = min(n_classes - 1, n_features, 3)  # Capped at 3 as per project requirements

    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    
    # Create DataFrame with LDA components
    lda_cols = [f'LD{i+1}' for i in range(n_components)]
    df_lda = pd.DataFrame(X_lda, columns=lda_cols)
    
    # Add back the target column
    df_lda['Class'] = y.values
    
    # Visualization: LDA components
    if output_dir:
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Visualize first two components
        plt.figure(figsize=(10, 8))
        
        # Use a color palette suitable for the number of classes
        unique_classes = np.unique(y)
        colors = sns.color_palette("tab10", len(unique_classes))
        
        # Plot each class
        for i, cls in enumerate(unique_classes):
            idx = y == cls
            plt.scatter(X_lda[idx, 0], X_lda[idx, 1], color=colors[i], alpha=0.7, label=f'Class {cls}')
        
        plt.xlabel('First Discriminant')
        plt.ylabel('Second Discriminant')
        plt.title('LDA: First Two Discriminants')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'lda_visualization.png'))
        plt.close()
        
        # If we have 3 components, visualize in 3D
        if n_components >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, cls in enumerate(unique_classes):
                idx = y == cls
                ax.scatter(X_lda[idx, 0], X_lda[idx, 1], X_lda[idx, 2], 
                           color=colors[i], alpha=0.7, label=f'Class {cls}')
            
            ax.set_xlabel('First Discriminant')
            ax.set_ylabel('Second Discriminant')
            ax.set_zlabel('Third Discriminant')
            ax.set_title('LDA: 3D Visualization')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'lda_visualization_3d.png'))
            plt.close()
    
    return df_lda, lda

def feature_engineering(df_preprocessed, output_dir):
    """
    Apply feature engineering techniques to the preprocessed data
    
    Parameters:
    -----------
    df_preprocessed : pd.DataFrame
        The preprocessed dataset
    output_dir : str
        Directory to save the output
        
    Returns:
    --------
    dict
        Dictionary with the three data representations and the dimensionality reduction objects
    """
    print("Starting feature engineering process...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Raw data (only preprocessed)
    df_raw = df_preprocessed.copy()
    print(f"Raw data shape: {df_raw.shape}")
    
    # Save raw dataset
    raw_path = os.path.join(output_dir, 'Dry_Bean_Raw.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to {raw_path}")
    
    # 2. PCA-transformed data
    df_pca, pca = apply_pca(df_preprocessed, output_dir)
    print(f"PCA-transformed data shape: {df_pca.shape}")
    
    # Save PCA dataset
    pca_path = os.path.join(output_dir, 'Dry_Bean_PCA.csv')
    df_pca.to_csv(pca_path, index=False)
    print(f"Saved PCA-transformed dataset to {pca_path}")
    
    # 3. LDA-transformed data
    df_lda, lda = apply_lda(df_preprocessed, output_dir)
    print(f"LDA-transformed data shape: {df_lda.shape}")
    
    # Save LDA dataset
    lda_path = os.path.join(output_dir, 'Dry_Bean_LDA.csv')
    df_lda.to_csv(lda_path, index=False)
    print(f"Saved LDA-transformed dataset to {lda_path}")
    
    # Return all datasets and objects
    data_representations = {
        'raw': {'data': df_raw, 'path': raw_path},
        'pca': {'data': df_pca, 'model': pca, 'path': pca_path},
        'lda': {'data': df_lda, 'model': lda, 'path': lda_path}
    }
    
    print("Feature engineering completed successfully!")
    return data_representations

if __name__ == "__main__":
    # This allows this module to be run as a standalone script for testing
    from data_preprocessing import preprocess_data
    
    input_file = "data/raw/Dry_Bean_Dataset.xlsx"
    processed_dir = "data/processed"
    
    # First, preprocess the data
    df_preprocessed, _ = preprocess_data(input_file, processed_dir)
    
    # Then, apply feature engineering
    data_representations = feature_engineering(df_preprocessed, processed_dir)
    print("Feature engineering completed successfully!")