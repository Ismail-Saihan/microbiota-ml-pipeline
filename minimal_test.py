#!/usr/bin/env python3
"""
Minimal test of GridSearchCV with SMOTE pipeline
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import time

print("=== MINIMAL GRIDSEARCHCV TEST ===")

# Load data
data = np.load("processed_microbiota_data.npz", allow_pickle=True)
X_train = data['X_train'][:1000]  # Use only first 1000 samples for speed
y_train = data['y_train'][:1000]
X_test = data['X_test'][:500]     # Use only first 500 test samples
y_test = data['y_test'][:500]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Classes in training: {np.unique(y_train)}")

# Create very simple pipeline
rf_model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', rf_model)
])

print("Pipeline created successfully")

# Very minimal parameter grid
params = {
    'model__max_depth': [5, 10]
}

print("Starting GridSearchCV...")
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    cv=cv,
    scoring='f1_macro',
    n_jobs=1,
    verbose=3
)

start_time = time.time()
search.fit(X_train, y_train)
end_time = time.time()

print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds")
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# Test prediction
y_pred = search.best_estimator_.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Test F1-Score: {test_f1:.4f}")

print("✅ Test completed successfully!")
