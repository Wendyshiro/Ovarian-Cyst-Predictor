import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

#load dataset
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Mock Up\Ovarian-Cyst-Track-Data.csv")
df = df.dropna(axis=1, how='all')
                 
#preview
print(df.head())
#validate columns
expected_cols = [
    'Patient ID', 'Age', 'Menopause Status', 'Cyst Size cm', 'Cyst Growth Rate cm/month',
    'CA 125 Level', 'Ultrasound Features', 'Reported Symptoms',
    'Recommended Management', 'Date of Exam', 'Region'
]


#check for missing values

missing = [col for col in expected_cols if col not in df.columns]
if missing:
    print("Missing columns:", missing)
    exit()
#encode categorical variables
label_encoder ={}
for column in {'Menopause Status', 'Ultrasound Features', 'Recommended Management', 'Region'}:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoder[column] = le
#feature engineering
#Cyst Size Category
def cyst_size_growth(size):
    if size <4 :
        return 'Small'
    elif size > 7:
        return "Medium"
    else:
        return "Large"

df['Cyst Size Category']= df["Cyst Size cm"].astype(float).apply(cyst_size_growth)
#Age gROUP
df['Age Group'] = pd.cut(df['Age'].astype(int), bins=[0,40,50,60,100])
#symptom count
df['Reported Symptoms'] = df['Reported Symptoms'].fillna('')
df['Symptom Count'] = df['Reported Symptoms'].apply(lambda x: len([s for s in x.split(',')if  s.strip()]))
#high CA125
df['High CA 125'] = (df['CA 125 Level'].astype(float) > 35 ).astype(int)
#is postmenopausal
df['Is Postmenopausal'] = (df['Menopause Status'] == label_encoder['Menopause Status'].transform(['Post-menopausal'])[0]).astype(int)
#large cyst & postmenopausal
df['Large PostMeno'] = ((df['Cyst Size cm'].astype(float) > 5) & (df['Is Postmenopausal'] == 1)).astype(int)
#One hot encode Cyst Size Category, Age GROUP, Ultrasound Features, Region
df = pd.get_dummies(df, columns=['Cyst Size Category', 'Age Group', 'Ultrasound Features', 'Region'])

#one hot encode symptoms
symptom_dummies = df['Reported Symptoms'].str.get_dummies(sep=",")
df =pd.concat([df, symptom_dummies], axis=1)
symptom_features = list(symptom_dummies.columns)
df.drop('Reported Symptoms', axis=1, inplace=True)
# target engineering
#categorize cyst growth rate cm/month into behavior categories
def cyst_behavior(rate):
    rate = float(rate)
    if rate > 1:
        return 'Rapid Growth'
    elif rate > 0:
        return 'Slow Growth'
    else:
        return 'No Growth'
df['Cyst Behavior'] = df['Cyst Growth Rate cm/month'].astype(float).apply(cyst_behavior)

#feature columns
featue_cols =[
     'Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count', 'High CA 125', 'Is Postmenopausal', 'Large PostMeno'
] + [col for col in df.columns if col.startswith('Cyst Size Category') or
                                 col.startswith('Age Group') or
                                    col.startswith('Ultrasound Features') or
                                    col.startswith('Region')] + symptom_features

# Predicting both 'cyst_growth' (cyst behavior) and 'recommended_management' based on patient data
X = df[featue_cols] 
y_behavior = df['Cyst Behavior']  
y_management = df['Recommended Management']

#scale numerical features
for col in ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count']:
    X[col] = X[col].astype(float)
scaler = StandardScaler()
X.loc[:, ['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count']] = scaler.fit_transform(X[['Age', 'Cyst Size cm', 'CA 125 Level', 'Symptom Count']])

#Evaluate models
models ={
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced',random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}
print("\nCross-Validation Scores for Cyst Behavior Prediction:\n")
for name, model in models.items():
    scores = cross_val_score(model, X, y_behavior, cv=5, scoring='f1_macro')
    print(f"{name}: Mean F1 (macro) = {scores.mean():.3f}")

#split for cyst behavior
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X,y_behavior, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X,y_management,test_size=0.2, random_state=42)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_b, y_train_b = smote.fit_resample(X_train_b, y_train_b)
X_train_m, y_train_m = smote.fit_resample(X_train_m, y_train_m)

# Train final models
final_behavior_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
final_behavior_model.fit(X_train_b, y_train_b)
y_pred_b = final_behavior_model.predict(X_test_b)

print("\nCyst Behavior Prediction:")
print(confusion_matrix(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b))

final_management_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
final_management_model.fit(X_train_m, y_train_m)
y_pred_m = final_management_model.predict(X_test_m)

print("\nRecommended Management Prediction:")
print(confusion_matrix(y_test_m, y_pred_m))
print(classification_report(y_test_m, y_pred_m))

# Add predictions to dataframe
df['Predicted Cyst Behavior'] = final_behavior_model.predict(X)
df['Predicted Management'] = final_management_model.predict(X)
df['Predicted Management'] = label_encoder['Recommended Management'].inverse_transform(df['Predicted Management'])


#  Export to Excel 
df.to_excel("ovarian_predictions_output.xlsx", index=False)
print("Results exported to ovarian_predictions_output.xlsx")


