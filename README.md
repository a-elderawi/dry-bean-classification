# Dry Bean Classification Project

This project involves implementing a complete machine learning workflow on the Dry Bean Dataset, including data preprocessing, feature engineering, and model evaluation.

## Project Overview

In this project, we work with the Dry Bean Dataset to perform:
- Data preprocessing (handling missing values, outliers)
- Feature scaling and categorical encoding
- Feature selection and dimensionality reduction (PCA, LDA)
- Model training using various classifiers
- Model evaluation using nested cross-validation
- Performance visualization with ROC curves

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/dry-bean-classification.git
cd dry-bean-classification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

3. Download the Dry Bean Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) and place it in the `data/raw/` directory.

4. Run the main script:
```bash
python main.py
```

## Project Structure

- `data/`: Contains raw and processed datasets
- `src/`: Source code for data preprocessing, feature engineering, and model training
- `output/`: Generated figures and model outputs
- `results/`: Performance metrics and final results
- `main.py`: Entry point for running the entire project

## Deliverables

1. GitHub repository with all code, visual outputs, and comments
2. A report including:
   - Code and visual outputs with comments
   - Tabular summary of performance metrics for each model
   - ROC curves
   - Interpretation of the best-performing model