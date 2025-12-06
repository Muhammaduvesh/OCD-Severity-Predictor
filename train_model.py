import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report  #ADDED THIS LINE
import joblib
import os

# Create directories
# os.makedirs('models', exist_ok=True)
# os.makedirs('data', exist_ok=True)

print("🚀 Training OCD Severity Prediction Model...")

# Load and preprocess data
df = pd.read_csv('OCD-data.csv')
df['OCD Diagnosis Date'] = pd.to_datetime(df['OCD Diagnosis Date'])
df['Previous Diagnoses'] = df['Previous Diagnoses'].fillna('None')
df['Total Y-BOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']

def severity_category(score):
    if score <= 15: return 'Mild'
    elif score <= 30: return 'Moderate'
    else: return 'Severe'
df['Severity Category'] = df['Total Y-BOCS Score'].apply(severity_category)

print(f"Dataset ready: {df.shape[0]} patients")
print("Severity distribution:")
print(df['Severity Category'].value_counts())

# SAFE ENCODING (same as Cell 6)
categorical_cols = ['Gender', 'Ethnicity', 'Marital Status', 'Education Level', 
                   'Previous Diagnoses', 'Family History of OCD', 'Obsession Type', 
                   'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis']

le_dict = {}
encoded_cols = []

print("\nEncoding categorical columns...")
for col in categorical_cols:
    if col in df.columns:
        print(f"Encoding: {col}")
        le = LabelEncoder()
        df[col] = df[col].fillna('Missing')
        df[col + '_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
        encoded_cols.append(col + '_encoded')
    else:
        print(f"Missing: {col}")

# DYNAMIC FEATURES
numeric_features = ['Age', 'Duration of Symptoms (months)']
feature_cols = [col for col in numeric_features + encoded_cols if col in df.columns]
print(f"\nUsing {len(feature_cols)} features")

X = df[feature_cols].copy()
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df['Severity Category'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, 
                                                    random_state=42, stratify=y_encoded)

# Cross-validation + GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
)

print("\nTraining model with GridSearchCV...")
rf_grid.fit(X_train, y_train)

# Results
print(f"\nBest parameters: {rf_grid.best_params_}")
print(f"Best CV Accuracy: {rf_grid.best_score_:.3f}")

best_rf = rf_grid.best_estimator_
y_pred = best_rf.predict(X_test)

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols, 
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 Features:")
print(importances.head())

# SAVE ALL
joblib.dump(best_rf, 'models/ocd_model.pkl')
joblib.dump(le_dict, 'models/label_encoders.pkl')
joblib.dump(le_target, 'models/target_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

print("\nPRODUCTION MODEL SAVED!")
print("Run: streamlit run streamlit_app.py")

