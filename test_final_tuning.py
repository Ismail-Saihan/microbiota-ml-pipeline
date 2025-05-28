#!/usr/bin/env python3
"""
Test script to debug final_model_tuning.py issues
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting test script...")

# Test 1: Data loading
print("Test 1: Loading data...")
try:
    data_path = "processed_microbiota_data.npz"
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {os.path.abspath(data_path)}")
    else:
        print(f"Data file found: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        print(f"Data keys: {list(data.keys())}")
        print(f"X_train shape: {data['X_train'].shape}")
        print(f"y_train shape: {data['y_train'].shape}")
        print("Data loading successful!")
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import dependencies
print("\nTest 2: Importing dependencies...")
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.dummy import DummyClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    print("All imports successful!")
except Exception as e:
    print(f"Error importing dependencies: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Basic pipeline creation
print("\nTest 3: Creating basic pipeline...")
try:
    rf_model = RandomForestClassifier(random_state=42, n_jobs=1)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', rf_model)
    ])
    print("Pipeline creation successful!")
except Exception as e:
    print(f"Error creating pipeline: {e}")
    import traceback
    traceback.print_exc()

print("\nTest script completed!")
