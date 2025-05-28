#!/usr/bin/env python3
"""
Debug script to identify the hanging issue in final_model_tuning.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== DEBUGGING FINAL MODEL TUNING ===")

# Test data loading step by step
data_path = "processed_microbiota_data.npz"
print(f"Loading data from: {data_path}")

data = np.load(data_path, allow_pickle=True)
print(f"Data keys: {list(data.keys())}")

X_train = data['X_train']
X_val = data['X_val'] 
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
print(f"y_val shape: {y_val.shape}, dtype: {y_val.dtype}")
print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")

# Check for any NaN or infinite values
print(f"X_train NaN count: {np.isnan(X_train).sum()}")
print(f"X_train infinite count: {np.isinf(X_train).sum()}")
print(f"y_train unique values: {np.unique(y_train)}")

# Combine train and validation for hyperparameter tuning
X_tune = np.vstack([X_train, X_val])
y_tune = np.hstack([y_train, y_val])

print(f"X_tune shape: {X_tune.shape}")
print(f"y_tune shape: {y_tune.shape}")
print(f"y_tune unique values: {np.unique(y_tune)}")
print(f"y_tune distribution: {np.bincount(y_tune.astype(int))}")

# Test basic model creation
print("\nTesting basic model creation...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    # Test DummyClassifier
    dummy_model = DummyClassifier(strategy='stratified', random_state=42)
    print("DummyClassifier created successfully")
    
    # Test SMOTE
    smote = SMOTE(random_state=42)
    print("SMOTE created successfully")
    
    # Test RandomForest
    rf_model = RandomForestClassifier(random_state=42, n_jobs=1)
    print("RandomForest created successfully")
    
    # Test ImbPipeline
    pipeline = ImbPipeline([
        ('smote', smote),
        ('model', rf_model)
    ])
    print("ImbPipeline created successfully")
    
    print("\nAll components created successfully!")
    
except Exception as e:
    print(f"Error during model creation: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")
