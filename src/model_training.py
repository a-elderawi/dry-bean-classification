import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, roc_auc_score, confusion_matrix)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Define classifier configurations
CLASSIFIERS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
}

def train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier_name, params, random_state=42):
    """
    Train and evaluate a single model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    classifier_name : str
        Name of the classifier
    params : dict
        Hyperparameters for the classifier
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with model, best parameters, and performance metrics
    """
    # Get classifier
    classifier_info = CLASSIFIERS[classifier_name]
    classifier = classifier_info['model']
    param_grid = params or classifier_info['params']
    
    # Inner cross-validation for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv,
                              scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # For ROC curves, need to handle multi-class
    try:
        # Try to get probability predictions
        y_prob = best_model.predict_proba(X_test)
        
        # Initialize ROC curve info
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        # Calculate ROC curve and ROC area for each class
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        y_test_bin = pd.get_dummies(y_test).values
        y_prob_flat = y_prob.reshape(-1)
        y_test_flat = y_test_bin.reshape(-1)
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_flat, y_prob_flat)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except:
        # If probabilities not available, just skip ROC
        fpr = tpr = roc_auc = None
    
    # Return results
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc},
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def nested_cross_validation(df, classifier_name, data_representation='raw', output_dir=None, n_outer=5, n_inner=3):
    """
    Perform nested cross-validation for model evaluation
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    classifier_name : str
        Name of the classifier
    data_representation : str
        Type of data representation ('raw', 'pca', or 'lda')
    output_dir : str, optional
        Directory to save the results
    n_outer : int, optional
        Number of folds for outer cross-validation
    n_inner : int, optional
        Number of folds for inner cross-validation
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    print(f"Performing nested cross-validation for {classifier_name} on {data_representation} data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    # Outer cross-validation
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    
    # Initialize lists to store results
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    best_params_list = []
    best_models = []
    best_roc_data = None
    best_confusion_matrix = None
    best_accuracy = -1
    
    # Perform outer cross-validation
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold {i+1}/{n_outer}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and evaluate model
        result = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, 
            classifier_name, None, random_state=i
        )
        
        # Store results
        accuracy_scores.append(result['accuracy'])
        precision_scores.append(result['precision'])
        recall_scores.append(result['recall'])
        f1_scores.append(result['f1'])
        best_params_list.append(result['best_params'])
        best_models.append(result['model'])
        
        # Keep track of best model for ROC curve
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_roc_data = result['roc_curve']
            best_confusion_matrix = result['confusion_matrix']
    
    # Calculate mean and std of metrics
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Mean Precision: {mean_precision:.4f} (±{std_precision:.4f})")
    print(f"Mean Recall: {mean_recall:.4f} (±{std_recall:.4f})")
    print(f"Mean F1 Score: {mean_f1:.4f} (±{std_f1:.4f})")
    
    # Save or visualize ROC curves
    if output_dir and best_roc_data:
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curves for all classes
        for i, (fpr, tpr, roc_auc) in enumerate(zip(
            best_roc_data['fpr'].values(), 
            best_roc_data['tpr'].values(), 
            best_roc_data['roc_auc'].values()
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
        plt.title(f'ROC Curves - {classifier_name} on {data_representation.upper()} data')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        roc_path = os.path.join(figures_dir, f'roc_{classifier_name}_{data_representation}.png')
        plt.savefig(roc_path)
        plt.close()
        
        # Plot confusion matrix
        if best_confusion_matrix is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title(f'Confusion Matrix - {classifier_name} on {data_representation.upper()} data')
            plt.tight_layout()
            
            cm_path = os.path.join(figures_dir, f'confusion_matrix_{classifier_name}_{data_representation}.png')
            plt.savefig(cm_path)
            plt.close()
    
    # Return results
    return {
        'classifier': classifier_name,
        'data_representation': data_representation,
        'accuracy': {'mean': mean_accuracy, 'std': std_accuracy},
        'precision': {'mean': mean_precision, 'std': std_precision},
        'recall': {'mean': mean_recall, 'std': std_recall},
        'f1': {'mean': mean_f1, 'std': std_f1},
        'best_params': best_params_list,
        'best_roc_data': best_roc_data,
        'best_models': best_models
    }

def train_models(data_representations, output_dir):
    """
    Train and evaluate models on different data representations
    
    Parameters:
    -----------
    data_representations : dict
        Dictionary with different data representations
    output_dir : str
        Directory to save the results
        
    Returns:
    --------
    dict
        Dictionary with all evaluation results
    """
    print("Starting model training and evaluation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results dictionary
    all_results = {}
    
    # Initialize DataFrame for performance metrics
    performance_metrics = pd.DataFrame(columns=[
        'Classifier', 'Data Representation',
        'Accuracy (Mean)', 'Accuracy (Std)',
        'Precision (Mean)', 'Precision (Std)',
        'Recall (Mean)', 'Recall (Std)',
        'F1 Score (Mean)', 'F1 Score (Std)'
    ])
    
    # Iterate over data representations
    for data_rep_name, data_rep_info in data_representations.items():
        all_results[data_rep_name] = {}
        
        df = data_rep_info['data']
        
        # Iterate over classifiers
        for classifier_name in CLASSIFIERS.keys():
            print(f"\nTraining {classifier_name} on {data_rep_name} data...")
            
            # Perform nested cross-validation
            result = nested_cross_validation(df, classifier_name, data_rep_name, output_dir)
            
            # Store result
            all_results[data_rep_name][classifier_name] = result
            
            # Add to performance metrics DataFrame
            performance_metrics = pd.concat([performance_metrics, pd.DataFrame({
                'Classifier': [classifier_name],
                'Data Representation': [data_rep_name],
                'Accuracy (Mean)': [result['accuracy']['mean']],
                'Accuracy (Std)': [result['accuracy']['std']],
                'Precision (Mean)': [result['precision']['mean']],
                'Precision (Std)': [result['precision']['std']],
                'Recall (Mean)': [result['recall']['mean']],
                'Recall (Std)': [result['recall']['std']],
                'F1 Score (Mean)': [result['f1']['mean']],
                'F1 Score (Std)': [result['f1']['std']]
            })], ignore_index=True)
    
    # Save performance metrics
    metrics_path = os.path.join(results_dir, 'performance_metrics.csv')
    performance_metrics.to_csv(metrics_path, index=False)
    print(f"Saved performance metrics to {metrics_path}")
    
    # Find the best model
    best_row = performance_metrics.loc[performance_metrics['F1 Score (Mean)'].idxmax()]
    best_classifier = best_row['Classifier']
    best_data_rep = best_row['Data Representation']
    best_accuracy = best_row['Accuracy (Mean)']
    best_f1 = best_row['F1 Score (Mean)']
    
    print("\nBest Performing Model:")
    print(f"Classifier: {best_classifier}")
    print(f"Data Representation: {best_data_rep}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 Score: {best_f1:.4f}")
    
    # Save the best model information
    with open(os.path.join(results_dir, 'best_model_info.txt'), 'w') as f:
        f.write(f"Best Performing Model:\n")
        f.write(f"Classifier: {best_classifier}\n")
        f.write(f"Data Representation: {best_data_rep}\n")
        f.write(f"Accuracy: {best_accuracy:.4f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
    
    # Generate comparative visualizations
    create_comparative_visualizations(performance_metrics, output_dir)
    
    return {
        'all_results': all_results,
        'performance_metrics': performance_metrics,
        'best_model': {
            'classifier': best_classifier,
            'data_representation': best_data_rep,
            'accuracy': best_accuracy,
            'f1_score': best_f1
        }
    }

def create_comparative_visualizations(performance_metrics, output_dir):
    """
    Create comparative visualizations of model performance
    
    Parameters:
    -----------
    performance_metrics : pd.DataFrame
        DataFrame with performance metrics
    output_dir : str
        Directory to save the visualizations
    """
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
    plt.savefig(os.path.join(figures_dir, 'comparative_accuracy.png'))
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
    plt.savefig(os.path.join(figures_dir, 'comparative_f1.png'))
    plt.close()
    
    # 3. Heatmap of performance metrics
    plt.figure(figsize=(14, 10))
    
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
    
    print(f"Saved comparative visualizations to {figures_dir}")

if __name__ == "__main__":
    # This allows this module to be run as a standalone script for testing
    from data_preprocessing import preprocess_data
    from feature_engineering import feature_engineering
    
    input_file = "data/raw/Dry_Bean_Dataset.xlsx"
    processed_dir = "data/processed"
    output_dir = "output"
    
    # First, preprocess the data
    df_preprocessed, _ = preprocess_data(input_file, processed_dir)
    
    # Then, apply feature engineering
    data_representations = feature_engineering(df_preprocessed, processed_dir)
    
    # Finally, train and evaluate models
    results = train_models(data_representations, output_dir)
    print("Model training and evaluation completed successfully!")