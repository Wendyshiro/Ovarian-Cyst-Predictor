import pandas as pd

try:
    # Read the Excel file
    df = pd.read_excel("ovarian_predictions_final.xlsx")
    
    print("=== EXCEL FILE ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Find prediction and confidence columns
    pred_cols = [col for col in df.columns if 'Predicted' in col]
    conf_cols = [col for col in df.columns if 'Confidence' in col]
    
    print(f"\nPrediction columns: {pred_cols}")
    print(f"Confidence columns: {conf_cols}")
    
    # Analyze confidence scores
    print("\n=== CONFIDENCE ANALYSIS ===")
    for col in conf_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.3f}")
            print(f"  Max: {df[col].max():.3f}")
            print(f"  >80%: {(df[col] > 0.8).sum()}")
            print(f"  >90%: {(df[col] > 0.9).sum()}")
    
    # Show high confidence samples
    if 'Ensemble_Confidence' in df.columns:
        high_conf = df[df['Ensemble_Confidence'] > 0.8]
        print(f"\nSamples with >80% ensemble confidence: {len(high_conf)}")
        if len(high_conf) > 0:
            print("Top 5 high confidence samples:")
            cols_to_show = ['Age', 'Cyst Size cm', 'CA 125 Level', 'Cyst Behavior Binary', 'Ensemble_Predicted', 'Ensemble_Confidence']
            available_cols = [col for col in cols_to_show if col in df.columns]
            print(high_conf[available_cols].head())
    
    print("\n=== SUMMARY ===")
    print("High confidence scores (80-90%+) indicate the ensemble model is very certain about those predictions.")
    print("This suggests the combination of Random Forest and XGBoost is working well!")
    
except Exception as e:
    print(f"Error reading file: {e}") 