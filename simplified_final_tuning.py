#!/usr/bin/env python3
"""
Simplified Final Hyperparameter Tuning for Testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.dummy import DummyClassifier
import joblib
import json
import time
import warnings
from datetime import datetime
import os
from typing import Any, Dict, List, Union

# Imblearn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    print("=== SIMPLIFIED FINAL HYPERPARAMETER TUNING ===")
    print("Testing with reduced parameter grids for faster execution")
    print("=" * 60)
    
    # Load data
    data_path = "processed_microbiota_data.npz"
    print(f"Loading data from: {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    X_val = data['X_val'] 
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Combine train and validation for tuning
    X_tune = np.vstack([X_train, X_val])
    y_tune = np.hstack([y_train, y_val])
    
    print(f"Tuning set: {X_tune.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes distribution: {np.bincount(y_tune)}")
    
    # Create results directory
    results_dir = "final_tuning_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    
    # 1. Baseline - DummyClassifier
    print("\n" + "="*50)
    print("1. BASELINE - DUMMYCLASSIFIER")
    print("="*50)
    
    dummy_model = DummyClassifier(strategy='stratified', random_state=42)
    dummy_model.fit(X_tune, y_tune)
    y_pred_dummy = dummy_model.predict(X_test)
    
    dummy_result = {
        'model_name': 'DummyClassifier',
        'test_accuracy': accuracy_score(y_test, y_pred_dummy),
        'test_f1': f1_score(y_test, y_pred_dummy, average='macro', zero_division=0),
        'tuning_time': 0.0
    }
    
    print(f"Test Accuracy: {dummy_result['test_accuracy']:.4f}")
    print(f"Test F1-Score: {dummy_result['test_f1']:.4f}")
    results.append(dummy_result)
    
    # 2. Random Forest with simplified grid
    print("\n" + "="*50)
    print("2. RANDOM FOREST (SIMPLIFIED GRID)")
    print("="*50)
    
    # Simplified parameter grid for faster testing
    rf_params = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20],
        'model__min_samples_split': [2, 5],
        'model__max_features': ['sqrt', 0.5]
    }
    
    rf_base_model = RandomForestClassifier(random_state=42, n_jobs=1)
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', rf_base_model)
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    start_time = time.time()
    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_params,
        cv=cv,
        scoring=scoring,
        refit='f1_macro',
        n_jobs=1,  # Single job to avoid resource conflicts
        verbose=2
    )
    
    print("Starting RandomForest tuning...")
    rf_search.fit(X_tune, y_tune)
    rf_tuning_time = time.time() - start_time
    
    # Evaluate on test set
    rf_best_model = rf_search.best_estimator_
    y_pred_rf = rf_best_model.predict(X_test)
    
    rf_result = {
        'model_name': 'RandomForest',
        'best_params': rf_search.best_params_,
        'cv_score': rf_search.best_score_,
        'test_accuracy': accuracy_score(y_test, y_pred_rf),
        'test_f1': f1_score(y_test, y_pred_rf, average='macro', zero_division=0),
        'tuning_time': rf_tuning_time
    }
    
    print(f"Best CV F1-Score: {rf_result['cv_score']:.4f}")
    print(f"Test Accuracy: {rf_result['test_accuracy']:.4f}")
    print(f"Test F1-Score: {rf_result['test_f1']:.4f}")
    print(f"Tuning time: {rf_tuning_time:.1f} seconds")
    print(f"Best parameters: {rf_search.best_params_}")
    
    results.append(rf_result)
    
    # Save model
    model_path = os.path.join(results_dir, 'best_randomforest_simplified.pkl')
    joblib.dump(rf_best_model, model_path)
    
    # Generate simple comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    print(f"{'Model':<20} {'Test Accuracy':<15} {'Test F1-Score':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    for result in results:
        model_name = result['model_name']
        test_acc = result['test_accuracy']
        test_f1 = result['test_f1']
        tuning_time = result['tuning_time']
        print(f"{model_name:<20} {test_acc:<15.4f} {test_f1:<15.4f} {tuning_time:<10.1f}")
    
    # Save results to JSON
    json_path = os.path.join(results_dir, 'simplified_tuning_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Best model
    best_model = max(results, key=lambda x: x['test_f1'])
    print(f"\nBest Model: {best_model['model_name']}")
    print(f"Best Test F1-Score: {best_model['test_f1']:.4f}")
    
    print("\n✅ Simplified tuning completed successfully!")

if __name__ == "__main__":
    main()
