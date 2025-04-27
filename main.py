import os
import pandas as pd
import numpy as np
import time
import argparse
from src.data_preprocessing import preprocess_data, visualize_preprocessing
from src.feature_engineering import feature_engineering
from src.model_training import train_models
from src.visualization import create_visualizations

def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'output/figures',
        'output/models',
        'results',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to run the entire project"""
    parser = argparse.ArgumentParser(description='Dry Bean Classification Project')
    parser.add_argument('--data-path', type=str, default='data/raw/Dry_Bean_Dataset.xlsx',
                        help='Path to the raw dataset file')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Skip feature engineering step')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create project structure
    create_project_structure()
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n" + "="*50)
        print("Step 1: Data Preprocessing")
        print("="*50)
        
        # Load the original dataset for visualization comparison
        try:
            df_original = pd.read_excel(args.data_path)
            print(f"Loaded original dataset: {args.data_path}")
        except Exception as e:
            print(f"Error loading original dataset: {e}")
            print("Make sure to download the Dry Bean Dataset from UCI ML Repository")
            print("and place it in the data/raw/ directory as Dry_Bean_Dataset.xlsx")
            return
        
        df_preprocessed, preprocessing_info = preprocess_data(args.data_path, args.processed_dir)
        visualize_preprocessing(df_original, df_preprocessed, args.output_dir)
        
        print(f"Data preprocessing completed in {(time.time() - start_time):.2f} seconds")
        preprocessing_time = time.time()
    else:
        print("\nSkipping preprocessing, loading preprocessed data...")
        preprocessed_path = os.path.join(args.processed_dir, 'Dry_Bean_Preprocessed.csv')
        
        if not os.path.exists(preprocessed_path):
            print(f"Error: Preprocessed data not found at {preprocessed_path}")
            print("Run the script without --skip-preprocessing first")
            return
        
        df_preprocessed = pd.read_csv(preprocessed_path)
        print(f"Loaded preprocessed data from {preprocessed_path}")
        preprocessing_time = time.time()
    
    # Step 2: Feature Engineering
    if not args.skip_feature_engineering:
        print("\n" + "="*50)
        print("Step 2: Feature Engineering")
        print("="*50)
        
        data_representations = feature_engineering(df_preprocessed, args.processed_dir)
        
        print(f"Feature engineering completed in {(time.time() - preprocessing_time):.2f} seconds")
        feature_eng_time = time.time()
    else:
        print("\nSkipping feature engineering, loading engineered data...")
        
        # Check if engineered data files exist
        raw_path = os.path.join(args.processed_dir, 'Dry_Bean_Raw.csv')
        pca_path = os.path.join(args.processed_dir, 'Dry_Bean_PCA.csv')
        lda_path = os.path.join(args.processed_dir, 'Dry_Bean_LDA.csv')
        
        if not all(os.path.exists(path) for path in [raw_path, pca_path, lda_path]):
            print("Error: Engineered data files not found")
            print("Run the script without --skip-feature-engineering first")
            return
        
        data_representations = {
            'raw': {'data': pd.read_csv(raw_path), 'path': raw_path},
            'pca': {'data': pd.read_csv(pca_path), 'path': pca_path},
            'lda': {'data': pd.read_csv(lda_path), 'path': lda_path}
        }
        
        print("Loaded engineered data representations")
        feature_eng_time = time.time()
    
    # Step 3: Model Training and Evaluation
    print("\n" + "="*50)
    print("Step 3: Model Training and Evaluation")
    print("="*50)
    
    results = train_models(data_representations, args.output_dir)
    
    print(f"Model training and evaluation completed in {(time.time() - feature_eng_time):.2f} seconds")
    training_time = time.time()
    
    # Step 4: Create Visualizations
    print("\n" + "="*50)
    print("Step 4: Creating Visualizations")
    print("="*50)
    
    create_visualizations(data_representations, results, args.output_dir)
    
    print(f"Visualization creation completed in {(time.time() - training_time):.2f} seconds")
    
    # Summary
    print("\n" + "="*50)
    print("Project Execution Summary")
    print("="*50)
    print(f"Total execution time: {(time.time() - start_time):.2f} seconds")
    
    # Print best model information
    best_model = results['best_model']
    print("\nBest Performing Model:")
    print(f"Classifier: {best_model['classifier']}")
    print(f"Data Representation: {best_model['data_representation']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"F1 Score: {best_model['f1_score']:.4f}")
    
    print("\nProject completed successfully!")
    print(f"Check {args.output_dir}/figures/ for visualizations")
    print(f"Check {args.results_dir}/ for performance metrics")

if __name__ == "__main__":
    main()