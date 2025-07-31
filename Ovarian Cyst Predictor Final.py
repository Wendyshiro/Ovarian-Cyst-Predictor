
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("Ovarian Cyst Track Data.csv")
df = df.dropna(axis=1, how='all')

# Enhanced Feature Engineering
print("Creating enhanced features...")

# Basic features
df['High CA 125'] = (df['CA 125 Level'].astype(float) > 35).astype(int)
df['Symptom Count'] = df['Reported Symptoms'].fillna('').apply(lambda x: len([s for s in x.split(',') if s.strip()]))
df['Is Postmenopausal'] = (df['Menopause Status'] == 'Post-menopausal').astype(int)
df['Large PostMeno'] = ((df['Cyst Size cm'].astype(float) > 5) & (df['Is Postmenopausal'] == 1)).astype(int)

# Advanced interaction features
df['CA125_PostMeno'] = df['High CA 125'] * df['Is Postmenopausal']
df['LargeCyst_HighCA'] = (df['Cyst Size cm'] > 5).astype(int) * df['High CA 125']
df['SymptomSeverity'] = (df['Symptom Count'] >= 2).astype(int)

# New engineered features
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3]).astype(int)
df['Cyst_Size_Category'] = pd.cut(df['Cyst Size cm'], bins=[0, 3, 5, 7, 20], labels=[0, 1, 2, 3]).astype(int)
df['CA125_Category'] = pd.cut(df['CA 125 Level'], bins=[0, 35, 100, 200, 1000], labels=[0, 1, 2, 3]).astype(int)

# Ratio features
df['Age_Size_Ratio'] = df['Age'] / (df['Cyst Size cm'] + 1)  # +1 to avoid division by zero
df['CA125_Size_Ratio'] = df['CA 125 Level'] / (df['Cyst Size cm'] + 1)
df['Symptom_Age_Ratio'] = df['Symptom Count'] / (df['Age'] + 1)

# Polynomial features
df['Age_Squared'] = df['Age'] ** 2
df['Cyst_Size_Squared'] = df['Cyst Size cm'] ** 2
df['CA125_Squared'] = df['CA 125 Level'] ** 2

# Risk score features
df['Risk_Score'] = (df['High CA 125'] * 2 + df['Is Postmenopausal'] * 2 + 
                    (df['Cyst Size cm'] > 5).astype(int) * 2 + df['Symptom Count'])

# Complex interaction features
df['High_Risk'] = ((df['High CA 125'] == 1) & (df['Is Postmenopausal'] == 1) & 
                   (df['Cyst Size cm'] > 5)).astype(int)
df['Moderate_Risk'] = ((df['High CA 125'] == 1) | (df['Is Postmenopausal'] == 1) | 
                       (df['Cyst Size cm'] > 5)).astype(int)

# Binary cyst behavior (numeric for ML)
def cyst_behavior(rate):
    return 1 if float(rate) > 0 else 0

df['Cyst Behavior Binary'] = df['Cyst Growth Rate cm/month'].apply(cyst_behavior)

# Rule-based recommendation logic
def recommend_management(row):
    if row['Is Postmenopausal'] and row['Cyst Size cm'] > 5 and row['High CA 125']:
        return 'Surgery'
    elif row['Cyst Size cm'] < 4 and not row['High CA 125']:
        return 'Observation'
    elif row['Symptom Count'] >= 2 and row['High CA 125']:
        return 'Medication'
    else:
        return 'Review'

df['Rule-Based Recommendation'] = df.apply(recommend_management, axis=1)

# Feature Selection using Random Forest
print("Performing feature selection...")

# All engineered features
all_features = ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count',
                'High CA 125', 'Is Postmenopausal', 'Large PostMeno',
                'CA125_PostMeno', 'LargeCyst_HighCA', 'SymptomSeverity',
                'Age_Group', 'Cyst_Size_Category', 'CA125_Category',
                'Age_Size_Ratio', 'CA125_Size_Ratio', 'Symptom_Age_Ratio',
                'Age_Squared', 'Cyst_Size_Squared', 'CA125_Squared',
                'Risk_Score', 'High_Risk', 'Moderate_Risk']

X_all = df[all_features].astype(float)
y = df['Cyst Behavior Binary']

# Use Random Forest to get feature importance
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_all, y)

# Get feature importance and select top features
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Select top 15 features
top_features = feature_importance.head(15)['feature'].tolist()
print(f"\nSelected top {len(top_features)} features: {top_features}")

X = df[top_features].astype(float)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count', 
                     'Age_Size_Ratio', 'CA125_Size_Ratio', 'Symptom_Age_Ratio',
                     'Age_Squared', 'Cyst_Size_Squared', 'CA125_Squared', 'Risk_Score']

# Only scale features that exist in top_features
features_to_scale = [f for f in numerical_features if f in top_features]
if features_to_scale:
    X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# Train/test split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Starting Random Forest Grid Search...")
# --- RANDOM FOREST GRID SEARCH ---
try:
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    print("Fitting Random Forest grid search...")
    rf_grid.fit(X_train_res, y_train_res)
    best_rf = rf_grid.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_proba = best_rf.predict_proba(X)[:, 1]

    print("\nBest Random Forest Params:", rf_grid.best_params_)
    print("Random Forest Results:")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    df["RF_Predicted"] = best_rf.predict(X)
    df["RF_Confidence"] = rf_proba
    print("Random Forest completed with enhanced features!")
    
except Exception as e:
    print(f"Error in Random Forest: {e}")
    # Fallback to simple Random Forest
    print("Using fallback Random Forest...")
    best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    best_rf.fit(X_train_res, y_train_res)
    rf_pred = best_rf.predict(X_test)
    rf_proba = best_rf.predict_proba(X)[:, 1]
    
    print("Random Forest Results (Fallback):")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    
    df["RF_Predicted"] = best_rf.predict(X)
    df["RF_Confidence"] = rf_proba
    print("Random Forest completed with fallback!")

print("Starting XGBoost Grid Search...")
# --- XGBOOST GRID SEARCH ---
try:
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 2, 5]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    print("Fitting XGBoost grid search...")
    xgb_grid.fit(X_train_res, y_train_res)
    best_xgb = xgb_grid.best_estimator_
    xgb_pred = best_xgb.predict(X_test)
    xgb_proba = best_xgb.predict_proba(X)[:, 1]

    print("\nBest XGBoost Params:", xgb_grid.best_params_)
    print("XGBoost Results:")
    print(confusion_matrix(y_test, xgb_pred))
    print(classification_report(y_test, xgb_pred))

    df["XGB_Predicted"] = best_xgb.predict(X)
    df["XGB_Confidence"] = xgb_proba
    print("XGBoost completed with enhanced features!")
    
except Exception as e:
    print(f"Error in XGBoost: {e}")
    # Fallback to simple XGBoost
    print("Using fallback XGBoost...")
    best_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_xgb.fit(X_train_res, y_train_res)
    xgb_pred = best_xgb.predict(X_test)
    xgb_proba = best_xgb.predict_proba(X)[:, 1]
    
    print("XGBoost Results (Fallback):")
    print(confusion_matrix(y_test, xgb_pred))
    print(classification_report(y_test, xgb_pred))
    
    df["XGB_Predicted"] = best_xgb.predict(X)
    df["XGB_Confidence"] = xgb_proba
    print("XGBoost completed with fallback!")

print("Starting Voting Classifier Ensemble...")
try:
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('xgb', best_xgb)
        ],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train_res, y_train_res)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_proba = ensemble.predict_proba(X)[:, 1]

    print("\nVoting Classifier (Ensemble) Results:")
    print(confusion_matrix(y_test, ensemble_pred))
    print(classification_report(y_test, ensemble_pred))

    df["Ensemble_Predicted"] = ensemble.predict(X)
    df["Ensemble_Confidence"] = ensemble_proba
    print("Ensemble completed!")
    
except Exception as e:
    print(f"Error in Ensemble: {e}")

# Export to Excel
df.to_excel("ovarian_predictions_final.xlsx", index=False)
print("Results exported to ovarian_predictions_final.xlsx")
