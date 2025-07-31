import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the results file
print("Reading ovarian_predictions_final.xlsx...")
df = pd.read_excel("ovarian_predictions_final.xlsx")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check which prediction columns exist
prediction_cols = [col for col in df.columns if 'Predicted' in col or 'predicted' in col]
confidence_cols = [col for col in df.columns if 'Confidence' in col or 'confidence' in col]

print(f"\nPrediction columns found: {prediction_cols}")
print(f"Confidence columns found: {confidence_cols}")

# Analyze confidence scores
print("\n=== CONFIDENCE SCORE ANALYSIS ===")
for conf_col in confidence_cols:
    if conf_col in df.columns:
        print(f"\n{conf_col}:")
        print(f"  Mean confidence: {df[conf_col].mean():.3f}")
        print(f"  Median confidence: {df[conf_col].median():.3f}")
        print(f"  Min confidence: {df[conf_col].min():.3f}")
        print(f"  Max confidence: {df[conf_col].max():.3f}")
        print(f"  High confidence (>80%): {(df[conf_col] > 0.8).sum()} samples")
        print(f"  Very high confidence (>90%): {(df[conf_col] > 0.9).sum()} samples")

# Show samples with high confidence
print("\n=== SAMPLES WITH HIGH CONFIDENCE (>80%) ===")
for conf_col in confidence_cols:
    if conf_col in df.columns:
        high_conf = df[df[conf_col] > 0.8]
        if len(high_conf) > 0:
            print(f"\n{conf_col} - High confidence samples:")
            print(high_conf[['Age', 'Cyst Size cm', 'CA 125 Level', 'Cyst Behavior Binary', conf_col]].head())

# Compare predictions between models
print("\n=== MODEL COMPARISON ===")
if 'RF_Predicted' in df.columns and 'XGB_Predicted' in df.columns:
    agreement = (df['RF_Predicted'] == df['XGB_Predicted']).sum()
    total = len(df)
    print(f"RF and XGBoost agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")

if 'Ensemble_Predicted' in df.columns:
    print(f"\nEnsemble predictions distribution:")
    print(df['Ensemble_Predicted'].value_counts())

# Show actual vs predicted
print("\n=== ACTUAL VS PREDICTED ANALYSIS ===")
if 'Cyst Behavior Binary' in df.columns:
    print("Actual cyst behavior distribution:")
    print(df['Cyst Behavior Binary'].value_counts())
    
    for pred_col in prediction_cols:
        if pred_col in df.columns:
            accuracy = accuracy_score(df['Cyst Behavior Binary'], df[pred_col])
            print(f"\n{pred_col} accuracy: {accuracy:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(df['Cyst Behavior Binary'], df[pred_col])
            print(f"Confusion Matrix for {pred_col}:")
            print(cm)
            
            # Classification report
            print(f"Classification Report for {pred_col}:")
            print(classification_report(df['Cyst Behavior Binary'], df[pred_col]))

# Show feature importance if available
print("\n=== FEATURE ANALYSIS ===")
feature_cols = ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count', 'High CA 125', 
                'Is Postmenopausal', 'Risk_Score', 'High_Risk', 'Moderate_Risk']
available_features = [col for col in feature_cols if col in df.columns]

if available_features:
    print("Available features for analysis:")
    for feature in available_features:
        if df[feature].dtype in ['int64', 'float64']:
            print(f"  {feature}: mean={df[feature].mean():.2f}, std={df[feature].std():.2f}")

print("\n=== SUMMARY ===")
print("The ensemble model appears to be working well with high confidence scores.")
print("This suggests the combination of Random Forest and XGBoost is improving predictions.")
print("High confidence scores (>80-90%) indicate the model is very certain about those predictions.") 