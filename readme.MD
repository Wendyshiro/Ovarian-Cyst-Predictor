# Ovarian Cyst Predictor

## Overview
This project predicts ovarian cyst behavior using advanced machine learning techniques. The latest version includes:
- **Advanced feature engineering** for improved predictive power
- **Model ensembling** (Random Forest + XGBoost) for higher accuracy
- **Analysis scripts** to interpret model results and confidence

## Recent Changes
- Added new engineered features (ratios, categories, risk scores, polynomial features)
- Feature selection using Random Forest importance
- Hyperparameter tuning (GridSearchCV) for both Random Forest and XGBoost
- Combined models using a VotingClassifier ensemble
- Output predictions and confidence scores for all models
- Added `analyze_results.py` and `quick_analysis.py` for result interpretation

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have `scikit-learn`, `xgboost`, `imblearn`, and `pandas` installed.

2. **Prepare your data:**
   - Place your data as `Ovarian Cyst Track Data.csv` in the project directory.

## Usage
1. **Run the main predictor:**
   ```bash
   python "Ovarian Cyst Predictor Final.py"
   ```
   This will:
   - Engineer features
   - Train Random Forest and XGBoost (with grid search)
   - Combine them in an ensemble
   - Output results to `ovarian_predictions_final.xlsx`

2. **Analyze results:**
   ```bash
   python analyze_results.py
   # or
   python quick_analysis.py
   ```
   These scripts will:
   - Summarize prediction confidence
   - Compare model predictions
   - Show accuracy and confusion matrices

## Interpreting Results
- **ovarian_predictions_final.xlsx** contains predictions and confidence scores for each model and the ensemble.
- **High confidence scores (>80-90%)** indicate the model is very certain about those predictions.
- **Ensemble predictions** are recommended for best performance.

## Contributing
Feel free to fork, open issues, or submit pull requests for further improvements!

## Author
Wendy Wanjiru
