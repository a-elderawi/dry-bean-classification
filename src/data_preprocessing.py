import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the specified file path
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file
        
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    print(f"Loading data from {file_path}")
    
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    return df

def add_missing_values(df, save_path=None):
    """
    Add missing values to the dataset as specified:
    - 5% missing values to two randomly selected columns
    - 35% missing values to one column
    
    Parameters:
    -----------
    df : pd.DataFrame
        The original dataset
    save_path : str, optional
        Path to save the dataset with missing values
        
    Returns:
    --------
    pd.DataFrame
        The dataset with added missing values
    """
    print("Adding missing values to the dataset...")
    df_missing = df.copy()
    
    # Randomly select numerical columns (excluding the class column)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Class' in numerical_cols:
        numerical_cols.remove('Class')
    
    # Randomly select two columns for 5% missing values
    cols_5pct = np.random.choice(numerical_cols, 2, replace=False)
    for col in cols_5pct:
        df_missing.loc[df_missing.sample(frac=0.05, random_state=42).index, col] = np.nan
        print(f"Added 5% missing values to column: {col}")
    
    # Randomly select one column for 35% missing values (different from the previous two)
    remaining_cols = [col for col in numerical_cols if col not in cols_5pct]
    col_35pct = np.random.choice(remaining_cols, 1)[0]
    df_missing.loc[df_missing.sample(frac=0.35, random_state=1).index, col_35pct] = np.nan
    print(f"Added 35% missing values to column: {col_35pct}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_missing.to_csv(save_path, index=False)
        print(f"Saved dataset with missing values to {save_path}")
    
    # Return the dataset and the columns with missing values
    return df_missing, {"cols_5pct": cols_5pct.tolist(), "col_35pct": col_35pct}

def handle_missing_values(df, missing_cols):
    """
    Handle missing values in the dataset:
    - Fill columns with 5% missing values using mean/median
    - Drop column/rows with 35% missing values
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset with missing values
    missing_cols : dict
        Dictionary with info about columns with missing values
        
    Returns:
    --------
    pd.DataFrame
        The dataset with handled missing values
    """
    print("Handling missing values...")
    df_handled = df.copy()
    
    # Display missing values summary
    print("Missing values summary:")
    print(df_handled.isnull().sum())
    
    # Handle columns with 5% missing values
    for col in missing_cols["cols_5pct"]:
        # Choose mean or median based on distribution
        if df_handled[col].skew() > 1 or df_handled[col].skew() < -1:
            # For skewed distributions, use median
            df_handled[col].fillna(df_handled[col].median(), inplace=True)
            print(f"Filled missing values in {col} with median")
        else:
            # For normal distributions, use mean
            df_handled[col].fillna(df_handled[col].mean(), inplace=True)
            print(f"Filled missing values in {col} with mean")
    
    # Handle column with 35% missing values
    col_35pct = missing_cols["col_35pct"]
    # Decision: Drop the column if it's not crucial, otherwise drop rows
    # For this example, we'll drop the column as 35% is quite high
    print(f"Dropping column {col_35pct} with 35% missing values")
    df_handled.drop(columns=[col_35pct], inplace=True)
    
    # Display missing values summary after handling
    print("Missing values after handling:")
    print(df_handled.isnull().sum())
    
    return df_handled

def detect_and_handle_outliers(df, method='IQR'):
    """
    Detect and handle outliers in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    method : str, optional
        Method for outlier detection ('IQR' or 'Z-score')
        
    Returns:
    --------
    pd.DataFrame
        The dataset with handled outliers
    """
    print(f"Detecting outliers using {method} method...")
    df_no_outliers = df.copy()
    
    # Get numerical columns (excluding class column)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Class' in numerical_cols:
        numerical_cols.remove('Class')
    
    outliers_summary = {}
    
    for col in numerical_cols:
        if method == 'IQR':
            # IQR method
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Detect outliers
            outliers = ((df_no_outliers[col] < lower_bound) | (df_no_outliers[col] > upper_bound))
            outliers_count = outliers.sum()
            
            # Handle outliers by replacing with boundary values
            df_no_outliers.loc[df_no_outliers[col] < lower_bound, col] = lower_bound
            df_no_outliers.loc[df_no_outliers[col] > upper_bound, col] = upper_bound
            
        elif method == 'Z-score':
            # Z-score method
            z_scores = (df_no_outliers[col] - df_no_outliers[col].mean()) / df_no_outliers[col].std()
            abs_z_scores = abs(z_scores)
            
            # Detect outliers
            threshold = 3
            outliers = (abs_z_scores > threshold)
            outliers_count = outliers.sum()
            
            # Handle outliers by replacing with boundary values
            mean = df_no_outliers[col].mean()
            std = df_no_outliers[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            df_no_outliers.loc[z_scores < -threshold, col] = lower_bound
            df_no_outliers.loc[z_scores > threshold, col] = upper_bound
        
        outliers_summary[col] = outliers_count
    
    print("Outliers summary:")
    for col, count in outliers_summary.items():
        print(f"{col}: {count} outliers")
    
    return df_no_outliers

def scale_features(df, scaler_type='standard'):
    """
    Scale numerical features in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    scaler_type : str, optional
        Type of scaler to use ('standard' or 'minmax')
        
    Returns:
    --------
    pd.DataFrame, object
        The dataset with scaled features and the scaler object
    """
    print(f"Scaling numerical features using {scaler_type} scaler...")
    
    # Separate features and target
    X = df.drop('Class', axis=1) if 'Class' in df.columns else df.copy()
    y = df['Class'] if 'Class' in df.columns else None
    
    # Get numerical columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    # Initialize scaler
    if scaler_type == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:  # minmax
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Scale numerical columns
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Reconstruct dataset
    df_scaled = X_scaled.copy()
    if y is not None:
        df_scaled['Class'] = y
    
    print("Features have been scaled")
    return df_scaled, scaler

def encode_categorical(df):
    """
    Encode categorical features in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
        
    Returns:
    --------
    pd.DataFrame, dict
        The dataset with encoded features and the encoder mappings
    """
    print("Encoding categorical features...")
    df_encoded = df.copy()
    encoder_mappings = {}
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle Class label separately
    if 'Class' in categorical_cols:
        categorical_cols.remove('Class')
        
        # Encode Class labels
        le = LabelEncoder()
        df_encoded['Class'] = le.fit_transform(df['Class'])
        
        # Store mapping for interpretation
        encoder_mappings['Class'] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Class mapping: {encoder_mappings['Class']}")
    
    # Encode other categorical features using one-hot encoding if any exist
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
        print(f"One-hot encoded categorical features: {categorical_cols}")
    
    return df_encoded, encoder_mappings

def preprocess_data(input_file, output_dir):
    """
    Full preprocessing pipeline for the dataset
    
    Parameters:
    -----------
    input_file : str
        Path to the input dataset file
    output_dir : str
        Directory to save the processed data
        
    Returns:
    --------
    pd.DataFrame, dict
        The fully preprocessed dataset and preprocessing info
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    df = load_data(input_file)
    
    # Add missing values
    missing_path = os.path.join(output_dir, 'Dry_Bean_Missing.csv')
    df_missing, missing_cols = add_missing_values(df, save_path=missing_path)
    
    # Handle missing values
    df_handled = handle_missing_values(df_missing, missing_cols)
    
    # Detect and handle outliers
    df_no_outliers = detect_and_handle_outliers(df_handled, method='IQR')
    
    # Encode categorical features (including Class label)
    df_encoded, encoder_mappings = encode_categorical(df_no_outliers)
    
    # Scale features
    df_scaled, scaler = scale_features(df_encoded, scaler_type='standard')
    
    # Save the preprocessed dataset
    preprocessed_path = os.path.join(output_dir, 'Dry_Bean_Preprocessed.csv')
    df_scaled.to_csv(preprocessed_path, index=False)
    print(f"Saved preprocessed dataset to {preprocessed_path}")
    
    preprocessing_info = {
        'missing_cols': missing_cols,
        'encoder_mappings': encoder_mappings,
        'original_shape': df.shape,
        'preprocessed_shape': df_scaled.shape
    }
    
    return df_scaled, preprocessing_info

def visualize_preprocessing(df_original, df_preprocessed, output_dir):
    """
    Create visualizations to illustrate preprocessing effects
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        The original dataset
    df_preprocessed : pd.DataFrame
        The preprocessed dataset
    output_dir : str
        Directory to save the visualizations
    """
    # Create output directory for figures if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Sample a few numerical columns for visualization
    numerical_cols = df_original.select_dtypes(include=np.number).columns.tolist()[:3]
    
    # 1. Distribution before and after preprocessing
    for col in numerical_cols:
        if col in df_preprocessed.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Before preprocessing
            sns.histplot(df_original[col], ax=ax1, kde=True)
            ax1.set_title(f'{col} - Before Preprocessing')
            
            # After preprocessing
            sns.histplot(df_preprocessed[col], ax=ax2, kde=True)
            ax2.set_title(f'{col} - After Preprocessing')
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'preprocessing_{col}.png'))
            plt.close()
    
    # 2. Correlation matrix before and after preprocessing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Before preprocessing (numerical columns only)
    corr_before = df_original.select_dtypes(include=np.number).corr()
    sns.heatmap(corr_before, ax=ax1, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title('Correlation Matrix - Before Preprocessing')
    
    # After preprocessing (numerical columns only)
    corr_after = df_preprocessed.select_dtypes(include=np.number).corr()
    sns.heatmap(corr_after, ax=ax2, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title('Correlation Matrix - After Preprocessing')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'preprocessing_correlation.png'))
    plt.close()
    
    print(f"Saved preprocessing visualizations to {figures_dir}")

if __name__ == "__main__":
    # This allows this module to be run as a standalone script for testing
    input_file = "data/raw/Dry_Bean_Dataset.xlsx"
    output_dir = "data/processed"
    
    df_preprocessed, preprocessing_info = preprocess_data(input_file, output_dir)
    print("Preprocessing completed successfully!")