import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("Ovarian-Cyst-Track-Data.csv")
df = df.dropna(axis=1, how='all')

# Feature engineering
df['High CA 125'] = (df['CA 125 Level'].astype(float) > 35).astype(int)
df['Symptom Count'] = df['Reported Symptoms'].fillna('').apply(lambda x: len([s for s in x.split(',') if s.strip()]))
df['Is Postmenopausal'] = (df['Menopause Status'] == 'Post-menopausal').astype(int)
df['Large PostMeno'] = ((df['Cyst Size cm'].astype(float) > 5) & (df['Is Postmenopausal'] == 1)).astype(int)

# Binary cyst behavior
def cyst_behavior(rate):
    rate = float(rate)
    return 'Unstable' if rate > 0 else 'Stable'

df['Cyst Behavior Binary'] = df['Cyst Growth Rate cm/month'].apply(cyst_behavior)

# Rule-based treatment recommendation
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

# ML features and target
features = ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count',
            'High CA 125', 'Is Postmenopausal', 'Large PostMeno']
X = df[features].astype(float)
y = df['Cyst Behavior Binary']

# Scale numerical columns
scaler = StandardScaler()
X[['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count']] = scaler.fit_transform(
    X[['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count']]
)

# Train/test split with SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Add predictions
df['Predicted Cyst Behavior'] = model.predict(X)
df['Prediction Match'] = df['Cyst Behavior Binary'] == df['Predicted Cyst Behavior']

# Export results
df.to_excel("ovarian_predictions_output.xlsx", index=False)
print("Results exported to ovarian_predictions_output.xlsx")
