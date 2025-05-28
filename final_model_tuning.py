#!/usr/bin/env python3
"""
Final Comprehensive Hyperparameter Tuning for Gut Microbiota Classification
Addresses issues in previous tuning and provides consistent evaluation methodology
Includes SMOTE in CV pipeline and expanded hyperparameter grids.
"""

import numpy as np
import pandas as pd # Keep for DataFrame creation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer # type: ignore # Removed classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import joblib # type: ignore
import json
import time
import warnings
from datetime import datetime
import os
from typing import Any, Dict, List, Union # Removed Tuple

# Imblearn imports - ensure imbalanced-learn is installed
from imblearn.pipeline import Pipeline as ImbPipeline # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore

# Suppress warnings
warnings.filterwarnings('ignore')

class FinalGutMicrobiotaTuning:
    """
    Final comprehensive hyperparameter tuning with consistent evaluation methodology,
    SMOTE in CV pipeline, and expanded grids.
    """
    
    def __init__(self, data_path: str):
        """Initialize with data loading"""
        self.data_path = data_path
        self.results_dir = "final_tuning_results"
        self.plots_dir = "final_tuning_plots"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.X_train: np.ndarray[Any, Any]
        self.X_val: np.ndarray[Any, Any]
        self.X_test: np.ndarray[Any, Any]
        self.y_train: np.ndarray[Any, Any] # Typically int, but Any for flexibility
        self.y_val: np.ndarray[Any, Any]   # Typically int
        self.y_test: np.ndarray[Any, Any]  # Typically int
        self.X_tune: np.ndarray[Any, Any]
        self.y_tune: np.ndarray[Any, Any]  # Typically int

        self.rf_params: Dict[str, Any]
        self.xgb_params: Dict[str, Any]
        self.lgb_params: Dict[str, Any]
        self.nn_params: Dict[str, Any]
        
        self.load_data()
        self.setup_models_and_grids() # Renamed
        
    def load_data(self):
        """Load and prepare data using same methodology as original models"""
        print("Loading preprocessed data...")
        
        if not os.path.exists(self.data_path):
            print(f"ERROR: Data file not found at {os.path.abspath(self.data_path)}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path, allow_pickle=True) # Added allow_pickle=True
        
        self.X_train = data['X_train']
        self.X_val = data['X_val'] 
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']
        
        # Combine train and validation for hyperparameter tuning
        self.X_tune = np.vstack([self.X_train, self.X_val])
        self.y_tune = np.hstack([self.y_train, self.y_val])
        
        print(f"Tuning set: {self.X_tune.shape}")
        print(f"Test set: {self.X_test.shape}")
        # Ensure y_tune is integer type for np.bincount
        if not np.issubdtype(self.y_tune.dtype, np.integer):
            try:
                self.y_tune = self.y_tune.astype(int)
            except ValueError:
                print("Warning: Could not convert y_tune to integer. Bincount might fail.")
        
        # Ensure y_tune is 1D array for bincount
        if self.y_tune.ndim > 1:
             self.y_tune = self.y_tune.ravel()

        print(f"Classes distribution in tuning: {np.bincount(self.y_tune)}")


    def setup_models_and_grids(self): # Renamed
        """Setup models with comprehensive parameter grids for use with SMOTE pipeline"""
        self.rf_params = {
            'model__n_estimators': [100, 200, 300, 400],
            'model__max_depth': [10, 20, 30, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', 0.5, 0.7], # Added float options
            # 'model__class_weight': ['balanced', 'balanced_subsample', None] # SMOTE handles imbalance
        }
        
        self.xgb_params = {
            'model__n_estimators': [100, 200, 300, 400],
            'model__max_depth': [5, 7, 9, 11],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'model__gamma': [0, 0.1, 0.2, 0.3], # Added gamma
            'model__reg_alpha': [0, 0.01, 0.1, 1.0],
            'model__reg_lambda': [0.1, 1, 1.5, 2.0],
            # 'model__scale_pos_weight': [self.class_weights[i] for i in range(len(self.class_weights))] # Complex for multi-class grid search
                                                                                                    # SMOTE preferred
        }
        
        self.lgb_params = {
            'model__n_estimators': [100, 200, 300, 400],
            'model__max_depth': [10, 15, 20, -1],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__num_leaves': [31, 50, 70, 100],
            'model__subsample': [0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'model__reg_alpha': [0, 0.01, 0.1, 1.0],
            'model__reg_lambda': [0, 0.01, 0.1, 1.0],
            'model__boosting_type': ['gbdt', 'dart'], # Added dart
            # 'model__class_weight': ['balanced', None], # SMOTE handles imbalance
            # 'model__is_unbalance': [True, False] # Alternative to class_weight, SMOTE preferred
        }
        
        self.nn_params = {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (128, 64), (100, 50, 25)],
            'model__activation': ['tanh', 'relu'],
            'model__solver': ['adam', 'sgd'],
            'model__alpha': [0.0001, 0.001, 0.01, 0.05],
            'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'model__max_iter': [300, 500, 700], # Increased iterations
            'model__early_stopping': [True],
            'model__batch_size': [32, 64, 128], # Removed 'auto' as it can be problematic with SMOTE pipeline
            'model__validation_fraction': [0.1], # For early stopping
            'model__n_iter_no_change': [10] # For early stopping
        }
        
    def tune_model(self, base_model: Any, params: Dict[str, Any], model_name: str, search_type: str = 'grid', n_iter: int = 50, use_smote: bool = True) -> Dict[str, Any]:
        """
        Tune a single model using specified search strategy, optionally with SMOTE in pipeline.
        """
        print(f"\\n{'='*50}")
        print(f"Tuning {model_name} with {search_type} search" + (" using SMOTE" if use_smote else ""))
        print(f"{'='*50}")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Increased to 5 folds
        
        # Create pipeline with SMOTE if enabled
        pipeline: Union[ImbPipeline, Any] # More specific type for pipeline
        if use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)), # type: ignore
                ('model', base_model)
            ])
        else:
            pipeline = base_model # Use model directly if SMOTE is not needed (e.g. DummyClassifier)

        start_time = time.time()
        
        # Define scoring: ensure roc_auc_score handles multi_class='ovr' and requires probabilities
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'roc_auc_ovr': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
        }
        refit_metric = 'f1_macro' # Choose the metric to optimize for

        search: Union[GridSearchCV, RandomizedSearchCV]
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=params,
                cv=cv,
                scoring=scoring,
                refit=refit_metric,
                n_jobs=-1,
                verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=params,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                refit=refit_metric,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported search_type: {search_type}")
        
        search.fit(self.X_tune, self.y_tune)
        tuning_time = time.time() - start_time
        
        best_model_pipeline = search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model_pipeline.predict(self.X_test)
        
        test_roc_auc = np.nan
        if hasattr(best_model_pipeline, "predict_proba"):
            try:
                y_pred_proba = best_model_pipeline.predict_proba(self.X_test)
                # Ensure y_test has unique classes that match y_pred_proba columns
                num_classes_test = len(np.unique(self.y_test))
                if y_pred_proba.shape[1] == num_classes_test and num_classes_test > 1 : # ROC AUC needs at least 2 classes
                     test_roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='macro')
                elif num_classes_test <=1:
                    print(f"Warning: ROC AUC not applicable for single class in y_test for {model_name}")
                else:
                    print(f"Warning: y_pred_proba shape {y_pred_proba.shape} not matching num_classes {num_classes_test} for {model_name}")
            except Exception as e:
                print(f"Could not calculate ROC AUC for {model_name}: {e}")
        
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        
        result: Dict[str, Any] = {
            'model_name': model_name,
            'search_type': search_type,
            'best_params': search.best_params_,
            'cv_score': search.best_score_, # This is the score for the refit_metric
            'cv_results_all_metrics': search.cv_results_, # Contains all scores
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'tuning_time': tuning_time,
        }
        
        print(f"Best CV F1-Score (refit metric): {search.best_score_:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score (macro): {test_f1:.4f}")
        print(f"Test ROC-AUC (macro OVR): {test_roc_auc:.4f if not np.isnan(test_roc_auc) else 'N/A'}")
        print(f"Tuning time: {tuning_time:.1f} seconds")
        
        model_path = os.path.join(self.results_dir, f'best_{model_name.lower().replace(" ", "_")}_{search_type}.pkl')
        joblib.dump(best_model_pipeline, model_path) # type: ignore
        
        return result
        
    def run_comprehensive_tuning(self) -> List[Dict[str, Any]]:
        """Run comprehensive hyperparameter tuning for all models"""
        print("Starting Comprehensive Hyperparameter Tuning with SMOTE in CV and Expanded Grids")
        print("=" * 70)
        
        results: List[Dict[str, Any]] = []

        # 0. Dummy Classifier (Baseline)
        dummy_model = DummyClassifier(strategy='stratified', random_state=42)
        # No hyperparameter tuning for DummyClassifier, just fit and evaluate
        dummy_model.fit(self.X_tune, self.y_tune)
        y_pred_dummy = dummy_model.predict(self.X_test)
        dummy_roc_auc_val = np.nan
        if hasattr(dummy_model, "predict_proba"):
            try:
                y_pred_proba_dummy = dummy_model.predict_proba(self.X_test)
                num_classes_test_dummy = len(np.unique(self.y_test))
                if y_pred_proba_dummy.shape[1] == num_classes_test_dummy and num_classes_test_dummy > 1:
                    dummy_roc_auc_val = roc_auc_score(self.y_test, y_pred_proba_dummy, multi_class='ovr', average='macro')
                elif num_classes_test_dummy <=1:
                    print("Warning: ROC AUC not applicable for single class in y_test for DummyClassifier")
            except Exception: # noqa: E722 (bare except)
                pass # Keep it as NaN

        dummy_result: Dict[str, Any] = {
            'model_name': 'DummyClassifier',
            'search_type': 'N/A',
            'best_params': {},
            'cv_score': f1_score(self.y_test, y_pred_dummy, average='macro', zero_division=0), # Using test F1 as CV score for consistency
            'cv_results_all_metrics': None,
            'test_accuracy': accuracy_score(self.y_test, y_pred_dummy),
            'test_f1': f1_score(self.y_test, y_pred_dummy, average='macro', zero_division=0),
            'test_roc_auc': dummy_roc_auc_val,
            'tuning_time': 0.0
        }
        print(f"\\n{'='*50}")
        print("Baseline DummyClassifier (Stratified)")
        print(f"Test Accuracy: {dummy_result['test_accuracy']:.4f}")
        print(f"Test F1-Score (macro): {dummy_result['test_f1']:.4f}")
        print(f"Test ROC-AUC (macro OVR): {dummy_result['test_roc_auc']:.4f if not np.isnan(dummy_result['test_roc_auc']) else 'N/A'}")
        print(f"{'='*50}")
        results.append(dummy_result)
        
        # 1. Random Forest
        # For RF, n_jobs in the model itself can conflict with n_jobs in GridSearchCV/RandomizedSearchCV.
        # It's often better to set n_jobs=-1 in the search and n_jobs=1 in the model, or manage carefully.
        # Here, we let the search parallelize the SMOTE + model fitting.
        rf_base_model = RandomForestClassifier(random_state=42, n_jobs=1) 
        rf_result = self.tune_model(rf_base_model, self.rf_params, 'RandomForest', 'random', n_iter=75, use_smote=True)
        results.append(rf_result)
        
        # 2. XGBoost
        # XGBoost can use all cores by default if n_jobs is not set or set to -1.
        # If using ImbPipeline, the n_jobs of GridSearchCV will handle parallel CV folds.
        xgb_base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False) # type: ignore
        xgb_result = self.tune_model(xgb_base_model, self.xgb_params, 'XGBoost', 'random', n_iter=75, use_smote=True)
        results.append(xgb_result)
        
        # 3. LightGBM
        # LightGBM also has its own n_jobs parameter.
        lgb_base_model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1) # type: ignore
        lgb_result = self.tune_model(lgb_base_model, self.lgb_params, 'LightGBM', 'random', n_iter=75, use_smote=True)
        results.append(lgb_result)
        
        # 4. Neural Network
        nn_base_model = MLPClassifier(random_state=42) # early_stopping is now in params
        nn_result = self.tune_model(nn_base_model, self.nn_params, 'NeuralNetwork', 'random', n_iter=50, use_smote=True)
        results.append(nn_result)
        
        return results
        
    def create_comprehensive_visualizations(self, results: List[Dict[str, Any]]):
        """Create comprehensive visualization suite"""
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid') # Updated style
        sns.set_palette("husl") # You can choose other palettes like "viridis", "magma"
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Adjusted size
        fig.suptitle('Final Hyperparameter Tuning Results - Comprehensive Analysis', fontsize=18, fontweight='bold')
        
        # Filter out results where metrics might be NaN (e.g. Dummy ROC AUC if predict_proba failed)
        valid_results = [r for r in results if not any(np.isnan(val) for val in [r.get('cv_score', np.nan), r.get('test_f1', np.nan), r.get('test_accuracy', np.nan), r.get('test_roc_auc', np.nan)])]
        if not valid_results:
            print("No valid results to plot after filtering NaNs.")
            plt.close(fig)
            return
            
        models = [str(r['model_name']) for r in valid_results] # Ensure model names are strings
        cv_scores = [float(r['cv_score']) for r in valid_results] 
        test_f1 = [float(r['test_f1']) for r in valid_results]
        test_accuracy = [float(r['test_accuracy']) for r in valid_results]
        # Handle potential NaN for test_roc_auc before plotting
        test_roc_auc = [float(r['test_roc_auc']) if not np.isnan(r['test_roc_auc']) else 0.0 for r in valid_results]


        x_pos = np.arange(len(models))
        width = 0.25 # Adjusted width for more metrics
        
        # CV F1-Score vs Test F1 Score
        axes[0,0].bar(x_pos - width, cv_scores, width, label='CV F1-Score (Refit)', alpha=0.8, color='skyblue')
        axes[0,0].bar(x_pos, test_f1, width, label='Test F1-Score', alpha=0.8, color='salmon')
        axes[0,0].bar(x_pos + width, test_accuracy, width, label='Test Accuracy', alpha=0.8, color='lightgreen')
        axes[0,0].set_title('Model Performance: F1-Scores and Accuracy', fontsize=14)
        axes[0,0].set_xlabel('Models', fontsize=12)
        axes[0,0].set_ylabel('Score', fontsize=12)
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45, ha="right")
        axes[0,0].legend(fontsize=10)
        axes[0,0].grid(True, linestyle='--', alpha=0.7)
        all_scores_ax00 = [s for s in cv_scores + test_f1 + test_accuracy if not np.isnan(s)] # Filter NaNs for max
        axes[0,0].set_ylim(0, max(all_scores_ax00) * 1.1 + 0.05 if all_scores_ax00 else 0.1)


        # Test Performance Metrics (Focus on ROC-AUC and F1)
        axes[0,1].bar(x_pos - width/2, test_f1, width, label='Test F1-Score', alpha=0.8, color='salmon')
        axes[0,1].bar(x_pos + width/2, test_roc_auc, width, label='Test ROC-AUC', alpha=0.8, color='gold')
        axes[0,1].set_title('Test Set Performance: F1-Score & ROC-AUC', fontsize=14)
        axes[0,1].set_xlabel('Models', fontsize=12)
        axes[0,1].set_ylabel('Score', fontsize=12)
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(models, rotation=45, ha="right")
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, linestyle='--', alpha=0.7)
        all_scores_ax01 = [s for s in test_f1 + test_roc_auc if not np.isnan(s)] # Filter NaNs for max
        axes[0,1].set_ylim(0, max(all_scores_ax01) * 1.1 + 0.05 if all_scores_ax01 else 0.1)

        # Tuning Time Analysis
        tuning_times = [float(r['tuning_time']) for r in valid_results]
        axes[1,0].bar(models, tuning_times, color='mediumpurple', alpha=0.7)
        axes[1,0].set_title('Hyperparameter Tuning Time', fontsize=14)
        axes[1,0].set_xlabel('Models', fontsize=12)
        axes[1,0].set_ylabel('Time (seconds)', fontsize=12)
        axes[1,0].tick_params(axis='x', rotation=45, ha="right")
        axes[1,0].grid(True, linestyle='--', alpha=0.7)
        
        # Performance vs Time Trade-off (Test F1 vs Time)
        scatter_colors = sns.color_palette("viridis", n_colors=len(models))
        for i, model_name_scatter in enumerate(models):
            axes[1,1].scatter(tuning_times[i], test_f1[i], s=150, alpha=0.8, label=model_name_scatter if i < 10 else "", color=scatter_colors[i]) # Label only a few to avoid clutter
        # Add annotations if not too many models
        if len(models) <= 10:
            for i, model_name_anno in enumerate(models):
                axes[1,1].annotate(model_name_anno, (tuning_times[i], test_f1[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1,1].set_title('Performance (Test F1) vs. Tuning Time', fontsize=14)
        axes[1,1].set_xlabel('Tuning Time (seconds)', fontsize=12)
        axes[1,1].set_ylabel('Test F1-Score', fontsize=12)
        axes[1,1].grid(True, linestyle='--', alpha=0.7)
        if len(models) > 10 : axes[1,1].legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)


        plt.tight_layout(rect=(0, 0, 1, 0.96)) # Adjust layout to make space for suptitle
        plot_path = os.path.join(self.plots_dir, 'final_tuning_comprehensive.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\\\\nComprehensive tuning plot saved to: {plot_path}")
        plt.close(fig)
        
        # 2. Best Parameters Visualization
        self.visualize_best_parameters(results) # Pass original results to include all models
        
    def visualize_best_parameters(self, results: List[Dict[str, Any]]):
        """Visualize best parameters for each model"""
        
        # Filter out DummyClassifier as it has no tuned params
        tuned_results = [r for r in results if r['model_name'] != 'DummyClassifier' and r.get('best_params')]
        
        if not tuned_results:
            print("No tuned models with parameters to visualize.")
            return

        num_models = len(tuned_results)
        # Adjust subplot layout based on number of models
        ncols = min(2, num_models) if num_models > 0 else 1
        nrows = (num_models + ncols - 1) // ncols if num_models > 0 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
        fig.suptitle('Best Hyperparameters by Model', fontsize=18, fontweight='bold')
        axes_flat = axes.flatten()
        
        current_model_idx = 0 # Initialize index for tuned_results
        for i in range(len(axes_flat)): # Iterate through all potential subplot axes
            if current_model_idx < len(tuned_results):
                result = tuned_results[current_model_idx]
                ax = axes_flat[i]
                
                params = result['best_params']
                # Filter out 'model__' prefix for display
                display_params = {key.replace('model__', ''): value for key, value in params.items()}

                param_text = "\\n".join([f"{name}: {val}" for name, val in display_params.items()])
                
                ax.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=9, wrap=True,
                        bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="steelblue", lw=1))
                
                cv_f1_val = result.get('cv_score', np.nan)
                test_f1_val = result.get('test_f1', np.nan)
                cv_f1_display = f"{cv_f1_val:.4f}" if not np.isnan(cv_f1_val) else "N/A"
                test_f1_display = f"{test_f1_val:.4f}" if not np.isnan(test_f1_val) else "N/A"

                ax.set_title(f"{result['model_name']}\\nCV F1: {cv_f1_display} | Test F1: {test_f1_display}", fontsize=12)
                ax.axis('off')
                current_model_idx +=1
            else:
                # Hide unused subplots if any remain
                fig.delaxes(axes_flat[i])

            
        plt.tight_layout(rect=(0, 0, 1, 0.95)) # Use tuple for rect
        plot_path = os.path.join(self.plots_dir, 'final_best_parameters.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Best parameters plot saved to: {plot_path}")
        plt.close(fig)
        
    def generate_final_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        valid_models_for_best = [r for r in results if r['model_name'] != 'DummyClassifier' and not np.isnan(r.get('test_f1', np.nan))]
        best_model: Dict[str, Any]
        if not valid_models_for_best:
            # Fallback: if only Dummy or all have NaN test_f1, pick one with highest test_f1 (even if NaN)
            # or the first one if all are NaN.
            non_nan_f1_results = [r for r in results if not np.isnan(r.get('test_f1', np.nan))]
            if non_nan_f1_results:
                 best_model = max(non_nan_f1_results, key=lambda x: x.get('test_f1', -float('inf')))
            else: # All results have NaN test_f1 or results is empty
                best_model = results[0] if results else {"model_name": "N/A", "test_f1": np.nan, "test_accuracy": np.nan, "test_roc_auc": np.nan}

        else:
            best_model = max(valid_models_for_best, key=lambda x: x['test_f1'])
        
        report = f"""# Final Gut Microbiota Classification - Hyperparameter Tuning Report

## Executive Summary
- **Total Models Tuned**: {len([r for r in results if r['model_name'] != 'DummyClassifier'])}
- **Best Performing Model (by Test F1-Score)**: {best_model['model_name']}
- **Best Test F1-Score**: {best_model.get('test_f1', np.nan):.4f}
- **Best Test Accuracy**: {best_model.get('test_accuracy', np.nan):.4f}
- **Best Test ROC-AUC**: {best_model.get('test_roc_auc', np.nan):.4f}

## Tuning Methodology
- **Dataset**: Advanced Feature Engineered Gut Microbiota Data
- **Features**: {self.X_tune.shape[1]}
- **Tuning Samples**: {self.X_tune.shape[0]}
- **Test Samples**: {self.X_test.shape[0]}
- **Cross-Validation**: 5-fold StratifiedKFold
- **Imbalance Handling**: SMOTE applied within each CV fold via imblearn.pipeline
- **Primary Evaluation Metric (for refit)**: F1-Score (macro average)
- **Other Metrics Tracked**: Accuracy, ROC-AUC (OVR Macro)

## Detailed Results

| Model             | CV F1 (Refit) | Test Accuracy | Test F1-Score | Test ROC-AUC | Tuning Time (s) | Search Type |
|-------------------|---------------|---------------|---------------|--------------|-----------------|-------------|
"""
        
        sorted_results = sorted(results, key=lambda x: x.get('test_f1', -float('inf')), reverse=True)

        for result in sorted_results:
            cv_score_val = result.get('cv_score', np.nan)
            test_acc_val = result.get('test_accuracy', np.nan)
            test_f1_val = result.get('test_f1', np.nan)
            test_roc_auc_val = result.get('test_roc_auc', np.nan)
            
            cv_score_display = f"{cv_score_val:.4f}" if not np.isnan(cv_score_val) else "N/A"
            test_acc_display = f"{test_acc_val:.4f}" if not np.isnan(test_acc_val) else "N/A"
            test_f1_display = f"{test_f1_val:.4f}" if not np.isnan(test_f1_val) else "N/A"
            test_roc_auc_display = f"{test_roc_auc_val:.4f}" if not np.isnan(test_roc_auc_val) else "N/A"
            tuning_time_display = f"{result.get('tuning_time', 0.0):.1f}"
            search_type_display = result.get('search_type', 'N/A')
            report += f"| {str(result['model_name']):<17} | {cv_score_display:<13} | {test_acc_display:<13} | {test_f1_display:<13} | {test_roc_auc_display:<12} | {tuning_time_display:<15} | {search_type_display:<11} |\\n"
        
        report += f"""
## Best Model Details: {best_model['model_name']}

### Performance Metrics
- **Cross-Validation F1-Score (Refit Metric)**: {best_model.get('cv_score', np.nan):.4f}
- **Test Accuracy**: {best_model.get('test_accuracy', np.nan):.4f}
- **Test F1-Score (macro)**: {best_model.get('test_f1', np.nan):.4f}
- **Test ROC-AUC (macro OVR)**: {best_model.get('test_roc_auc', np.nan):.4f}

### Best Parameters
"""
        if best_model['model_name'] == 'DummyClassifier':
            report += "- Strategy: stratified (or as configured)\\n"
        elif best_model.get('best_params'):
            for param, value in best_model['best_params'].items():
                display_param = param.replace('model__', '')
                report += f"- **{display_param}**: {value}\\n"
        else:
            report += "- No parameters tuned for this model.\\n"
        
        dummy_f1_val = next((r.get('test_f1', np.nan) for r in results if r['model_name'] == 'DummyClassifier'), np.nan)
        dummy_f1_display = f"{dummy_f1_val:.4f}" if not np.isnan(dummy_f1_val) else "N/A"

        report += f"""
## Key Insights

1. **Model Performance**: The best tuned model ({best_model['model_name']}) achieves {best_model.get('test_f1', np.nan):.4f} F1-score on the test data.
   Compare this to the DummyClassifier F1-score of {dummy_f1_display}.
2. **SMOTE Impact**: Using SMOTE within the cross-validation pipeline helps in handling class imbalance robustly during model training and selection.
3. **Optimization Impact**: Hyperparameter tuning aimed to find better configurations. The difference between the DummyClassifier and other models shows the learning achieved.
4. **Efficiency**: Total tuning time for all models was approximately {sum(r.get('tuning_time', 0.0) for r in results)/60:.1f} minutes.

## Recommendations

1. **Deployment Candidate**: Consider deploying **{best_model['model_name']}** with the optimized parameters.
2. **Further Analysis**: Deep dive into the `cv_results_all_metrics` for the best model to understand parameter sensitivity.
3. **Error Analysis**: Perform an error analysis on the predictions of the best model to understand where it performs poorly and identify potential areas for improvement (e.g., specific classes, feature interactions).
4. **Monitoring**: If deployed, monitor performance on new data and retune periodically.
5. **Alternative Strategies**: If performance is still unsatisfactory, consider:
    - More advanced feature engineering or selection.
    - Trying different model architectures (e.g., CatBoost, TabNet if not already used).
    - Ensemble methods combining predictions from several top models.

## Technical Notes
- All models (except DummyClassifier) were tuned using {str(best_model.get('search_type', 'N/A'))} search.
- Stratified 5-fold cross-validation was used.
- Test set was held out during the entire tuning process.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_path = os.path.join(self.results_dir, 'final_tuning_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        json_results_path = os.path.join(self.results_dir, 'final_tuning_results.json')
        
        serializable_results = []
        for res_dict in results: # Renamed to avoid conflict
            s_result: Dict[str, Any] = {}
            for key, value in res_dict.items():
                if key == 'cv_results_all_metrics': 
                    s_result[key] = "See detailed CV results in pickled search object if needed."
                    continue
                try:
                    json.dumps(value) 
                    s_result[key] = value
                except TypeError:
                    if isinstance(value, (np.generic, np.ndarray)):
                        s_result[key] = value.item() if hasattr(value, 'item') and isinstance(value, np.generic) and value.size == 1 else str(value)
                    elif isinstance(value, dict): # Handle dicts with numpy types
                        s_result[key] = {str(k_dict): (v_dict.item() if hasattr(v_dict, 'item') and isinstance(v_dict, np.generic) and v_dict.size == 1 else str(v_dict)) for k_dict, v_dict in value.items()}
                    else:
                        s_result[key] = str(value)
            serializable_results.append(s_result)
        
        with open(json_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        df_data = []
        for r_df in results: # Renamed to avoid conflict
            df_row: Dict[str, Any] = {
                'Model': r_df['model_name'],
                'CV_F1_Score_Refit': r_df.get('cv_score', np.nan),
                'Test_Accuracy': r_df.get('test_accuracy', np.nan),
                'Test_F1_Score': r_df.get('test_f1', np.nan),
                'Test_ROC_AUC': r_df.get('test_roc_auc', np.nan),
                'Tuning_Time_Seconds': r_df.get('tuning_time', 0.0),
                'Search_Type': r_df.get('search_type', 'N/A')
            }
            df_data.append(df_row)

        df_results = pd.DataFrame(df_data)
        csv_path = os.path.join(self.results_dir, 'final_tuning_comparison.csv')
        df_results.to_csv(csv_path, index=False)
        
        print(f"\\nReports saved to:")
        print(f"- Markdown Report: {report_path}")
        print(f"- JSON Results: {json_results_path}")
        print(f"- CSV Comparison: {csv_path}")
        
        return best_model

def main():
    """Main execution function"""
    
    print("Final Comprehensive Hyperparameter Tuning for Gut Microbiota Classification")
    print("Includes SMOTE in CV pipeline, expanded grids, and DummyClassifier baseline.")
    print("=" * 80)
    
    data_path = "processed_microbiota_data.npz" 
    # Check for data_path moved to FinalGutMicrobiotaTuning.__init__ via load_data

    try:
        tuner = FinalGutMicrobiotaTuning(data_path)
        results = tuner.run_comprehensive_tuning()
        
        if results:
            print("\\nGenerating comprehensive visualizations...")
            tuner.create_comprehensive_visualizations(results)
            
            print("\\nGenerating final report...")
            best_model_info = tuner.generate_final_report(results)
            
            print("\\n" + "=" * 80)
            print("FINAL TUNING COMPLETE!")
            print("=" * 80)
            # Removed isinstance check as best_model_info is typed Dict[str, Any]
            if best_model_info: 
                print(f"Best Model (by Test F1-Score): {best_model_info.get('model_name', 'N/A')}")
                test_f1_val = best_model_info.get('test_f1', np.nan)
                test_acc_val = best_model_info.get('test_accuracy', np.nan)
                test_roc_auc_val = best_model_info.get('test_roc_auc', np.nan)

                print(f"  Test F1-Score: {test_f1_val:.4f if not np.isnan(test_f1_val) else 'N/A'}")
                print(f"  Test Accuracy: {test_acc_val:.4f if not np.isnan(test_acc_val) else 'N/A'}")
                print(f"  Test ROC-AUC: {test_roc_auc_val:.4f if not np.isnan(test_roc_auc_val) else 'N/A'}")
            else:
                print("No best model identified or best_model_info is not a dictionary.")
            print("=" * 80)
        else:
            print("Tuning did not produce any results. Please check the logs.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure 'processed_microbiota_data.npz' is in the correct location relative to the script.")
        return # Exit if data not found
    except Exception as e:
        print(f"An unexpected error occurred during the tuning process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Removed unused imports that were previously here (uniform, randint)
    # Imports are now at the top of the file.
    main()
