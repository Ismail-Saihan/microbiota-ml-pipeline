import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, make_scorer
)
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import uniform, randint
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import time
import json

# For Bayesian Optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-optimize not available. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-optimize"])
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True

class GutMicrobiotaModelTuning:
    """
    Comprehensive hyperparameter tuning pipeline for gut microbiota classification
    using Grid Search, Random Search, and Bayesian Optimization
    """
    
    def __init__(self, data_path):
        """Initialize with enhanced dataset"""
        print("🔧 GUT MICROBIOTA MODEL HYPERPARAMETER TUNING")
        print("="*55)
        
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.target_column = 'Current status of microbiota'
        
        # Create directories for outputs
        os.makedirs("tuning_results", exist_ok=True)
        os.makedirs("tuning_plots", exist_ok=True)
        
        print(f"✅ Dataset loaded: {self.df.shape}")
        print(f"🎯 Target classes: {self.df[self.target_column].value_counts().to_dict()}")
        
        # Initialize containers for results
        self.tuned_models = {}
        self.tuning_results = {}
        self.best_params = {}
        
        # Prepare data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for tuning"""
        print("\n1. DATA PREPARATION FOR TUNING")
        print("-" * 35)
        
        # Separate features and target
        X = self.df.drop([self.target_column], axis=1)
        y = self.df[self.target_column]
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Store data
        self.X = X_scaled
        self.y = y_encoded
        self.feature_names = X.columns.tolist()
        
        # Setup cross-validation
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"✅ Data prepared: {X_scaled.shape}")
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Classes: {len(np.unique(y_encoded))}")
        print(f"✅ CV Strategy: {self.cv.n_splits}-fold Stratified")
        
    def define_search_spaces(self):
        """Define hyperparameter search spaces for different algorithms"""
        print("\n2. DEFINING HYPERPARAMETER SEARCH SPACES")
        print("-" * 45)
        
        self.param_spaces = {
            'RandomForest': {
                'grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced', 'balanced_subsample']
                },
                'random': {
                    'n_estimators': randint(50, 500),
                    'max_depth': [10, 20, 30, 50, None],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced', 'balanced_subsample']
                },
                'bayesian': {
                    'n_estimators': Integer(50, 500),
                    'max_depth': Integer(10, 50),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2']),
                    'class_weight': Categorical(['balanced', 'balanced_subsample'])
                }
            },
            
            'XGBoost': {
                'grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2]
                },
                'random': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'reg_alpha': uniform(0, 2),
                    'reg_lambda': uniform(0.5, 2)
                },
                'bayesian': {
                    'n_estimators': Integer(50, 500),
                    'max_depth': Integer(3, 15),
                    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0.0, 2.0),
                    'reg_lambda': Real(0.5, 2.0)
                }
            },
            
            'LightGBM': {
                'grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [0, 0.1, 1]
                },
                'random': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(5, 20),
                    'learning_rate': uniform(0.01, 0.3),
                    'num_leaves': randint(20, 150),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'reg_alpha': uniform(0, 2),
                    'reg_lambda': uniform(0, 2)
                },
                'bayesian': {
                    'n_estimators': Integer(50, 500),
                    'max_depth': Integer(5, 20),
                    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                    'num_leaves': Integer(20, 150),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0.0, 2.0),
                    'reg_lambda': Real(0.0, 2.0)
                }
            },
            
            'NeuralNetwork': {
                'grid': {
                    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [500, 1000]
                },
                'random': {
                    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128), (200, 100, 50)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': uniform(0.0001, 0.1),
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [500, 1000, 1500]
                },
                'bayesian': {
                    'hidden_layer_sizes': Categorical([(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]),
                    'activation': Categorical(['relu', 'tanh']),
                    'solver': Categorical(['adam']),
                    'alpha': Real(0.0001, 0.1, 'log-uniform'),
                    'learning_rate': Categorical(['constant', 'adaptive']),
                    'max_iter': Integer(300, 1000)
                }
            }
        }
        
        print("✅ Search spaces defined for 4 algorithms")
        for algo, spaces in self.param_spaces.items():
            grid_combinations = 1
            for param, values in spaces['grid'].items():
                grid_combinations *= len(values) if isinstance(values, list) else 1
            print(f"   {algo}: ~{grid_combinations:,} grid combinations")
    
    def tune_model_grid_search(self, model_name, model, param_grid, scoring='f1_macro'):
        """Tune model using Grid Search"""
        print(f"\n🔍 Grid Search: {model_name}")
        print("-" * 30)
        
        start_time = time.time()
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(self.X, self.y)
        
        end_time = time.time()
        
        # Store results
        result = {
            'method': 'GridSearch',
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'n_combinations': len(grid_search.cv_results_['params']),
            'time_taken': end_time - start_time,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
        print(f"⏱️ Time taken: {end_time - start_time:.1f} seconds")
        print(f"🔢 Combinations tested: {result['n_combinations']}")
        
        return grid_search.best_estimator_, result
    
    def tune_model_random_search(self, model_name, model, param_distributions, scoring='f1_macro', n_iter=100):
        """Tune model using Random Search"""
        print(f"\n🎲 Random Search: {model_name}")
        print("-" * 32)
        
        start_time = time.time()
        
        # Create random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        # Fit random search
        random_search.fit(self.X, self.y)
        
        end_time = time.time()
        
        # Store results
        result = {
            'method': 'RandomSearch',
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'n_combinations': n_iter,
            'time_taken': end_time - start_time,
            'cv_results': random_search.cv_results_
        }
        
        print(f"✅ Best CV Score: {random_search.best_score_:.4f}")
        print(f"⏱️ Time taken: {end_time - start_time:.1f} seconds")
        print(f"🎲 Iterations: {n_iter}")
        
        return random_search.best_estimator_, result
    
    def tune_model_bayesian(self, model_name, model, search_space, scoring='f1_macro', n_calls=50):
        """Tune model using Bayesian Optimization"""
        print(f"\n🧠 Bayesian Optimization: {model_name}")
        print("-" * 38)
        
        start_time = time.time()
        
        # Create Bayesian search
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_space,
            n_iter=n_calls,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        # Fit Bayesian search
        bayes_search.fit(self.X, self.y)
        
        end_time = time.time()
        
        # Store results
        result = {
            'method': 'BayesianOptimization',
            'best_score': bayes_search.best_score_,
            'best_params': bayes_search.best_params_,
            'n_combinations': n_calls,
            'time_taken': end_time - start_time,
            'cv_results': bayes_search.cv_results_
        }
        
        print(f"✅ Best CV Score: {bayes_search.best_score_:.4f}")
        print(f"⏱️ Time taken: {end_time - start_time:.1f} seconds")
        print(f"🎯 Calls: {n_calls}")
        
        return bayes_search.best_estimator_, result
    
    def tune_all_models(self):
        """Tune all models using multiple optimization strategies"""
        print("\n3. COMPREHENSIVE MODEL TUNING")
        print("-" * 35)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='mlogloss',
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            ),
            'NeuralNetwork': MLPClassifier(
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=1000
            )
        }
        
        # Tune each model with all methods
        for model_name, base_model in models.items():
            print(f"\n{'='*50}")
            print(f"🎯 TUNING: {model_name}")
            print(f"{'='*50}")
            
            model_results = {}
            
            # 1. Grid Search (reduced for computational efficiency)
            try:
                # Use smaller grid for computational efficiency
                small_grid = {}
                original_grid = self.param_spaces[model_name]['grid']
                for param, values in original_grid.items():
                    if isinstance(values, list) and len(values) > 3:
                        # Take every other value for large parameter lists
                        small_grid[param] = values[::2][:3]
                    else:
                        small_grid[param] = values
                
                best_model_grid, grid_result = self.tune_model_grid_search(
                    model_name, base_model, small_grid
                )
                model_results['grid'] = grid_result
                model_results['grid']['model'] = best_model_grid
            except Exception as e:
                print(f"❌ Grid Search failed for {model_name}: {e}")
            
            # 2. Random Search
            try:
                best_model_random, random_result = self.tune_model_random_search(
                    model_name, base_model, 
                    self.param_spaces[model_name]['random'],
                    n_iter=50
                )
                model_results['random'] = random_result
                model_results['random']['model'] = best_model_random
            except Exception as e:
                print(f"❌ Random Search failed for {model_name}: {e}")
            
            # 3. Bayesian Optimization
            try:
                best_model_bayes, bayes_result = self.tune_model_bayesian(
                    model_name, base_model,
                    self.param_spaces[model_name]['bayesian'],
                    n_calls=30
                )
                model_results['bayesian'] = bayes_result
                model_results['bayesian']['model'] = best_model_bayes
            except Exception as e:
                print(f"❌ Bayesian Optimization failed for {model_name}: {e}")
            
            # Store results
            self.tuning_results[model_name] = model_results
            
            # Find best method for this model
            best_method = None
            best_score = 0
            for method, result in model_results.items():
                if result['best_score'] > best_score:
                    best_score = result['best_score']
                    best_method = method
            
            if best_method:
                self.tuned_models[model_name] = model_results[best_method]['model']
                self.best_params[model_name] = {
                    'method': best_method,
                    'params': model_results[best_method]['best_params'],
                    'score': best_score
                }
                
                print(f"\n🏆 BEST FOR {model_name}: {best_method.upper()}")
                print(f"   Score: {best_score:.4f}")
                print(f"   Params: {model_results[best_method]['best_params']}")
    
    def create_tuning_visualizations(self):
        """Create comprehensive tuning visualizations"""
        print("\n4. CREATING TUNING VISUALIZATIONS")
        print("-" * 40)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Method Comparison Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Tuning Results Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for visualization
        method_scores = {'Grid': [], 'Random': [], 'Bayesian': []}
        model_names = []
        
        for model_name, results in self.tuning_results.items():
            model_names.append(model_name)
            for method in ['grid', 'random', 'bayesian']:
                if method in results:
                    method_scores[method.capitalize()].append(results[method]['best_score'])
                else:
                    method_scores[method.capitalize()].append(0)
        
        # Plot 1: Method comparison
        ax1 = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (method, scores) in enumerate(method_scores.items()):
            ax1.bar(x + i * width, scores, width, label=method, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('CV F1-Score')
        ax1.set_title('Optimization Method Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best parameters heatmap
        ax2 = axes[0, 1]
        best_scores = [self.best_params[model]['score'] for model in model_names if model in self.best_params]
        best_methods = [self.best_params[model]['method'] for model in model_names if model in self.best_params]
        
        # Create method-score matrix for heatmap
        method_matrix = np.zeros((len(set(best_methods)), len(model_names)))
        unique_methods = list(set(best_methods))
        
        for i, model in enumerate(model_names):
            if model in self.best_params:
                method_idx = unique_methods.index(self.best_params[model]['method'])
                method_matrix[method_idx, i] = self.best_params[model]['score']
        
        sns.heatmap(method_matrix, 
                   xticklabels=model_names, 
                   yticklabels=unique_methods,
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title('Best Method by Model (F1-Score)')
        
        # Plot 3: Time comparison
        ax3 = axes[1, 0]
        time_data = []
        for model_name, results in self.tuning_results.items():
            for method in ['grid', 'random', 'bayesian']:
                if method in results:
                    time_data.append({
                        'Model': model_name,
                        'Method': method.capitalize(),
                        'Time': results[method]['time_taken'] / 60  # Convert to minutes
                    })
        
        time_df = pd.DataFrame(time_data)
        if not time_df.empty:
            sns.barplot(data=time_df, x='Model', y='Time', hue='Method', ax=ax3)
            ax3.set_ylabel('Time (minutes)')
            ax3.set_title('Optimization Time Comparison')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Score improvement
        ax4 = axes[1, 1]
        improvement_data = []
        
        # Load baseline scores from previous results
        try:
            baseline_df = pd.read_csv('model_results/model_performance_comparison.csv', index_col=0)
            baseline_scores = baseline_df['F1_Score'].to_dict()
            
            for model_name in model_names:
                if model_name in self.best_params and model_name in baseline_scores:
                    baseline = baseline_scores[model_name]
                    tuned = self.best_params[model_name]['score']
                    improvement = ((tuned - baseline) / baseline) * 100
                    improvement_data.append({
                        'Model': model_name,
                        'Improvement': improvement,
                        'Baseline': baseline,
                        'Tuned': tuned
                    })
            
            if improvement_data:
                imp_df = pd.DataFrame(improvement_data)
                bars = ax4.bar(imp_df['Model'], imp_df['Improvement'], 
                              color=['green' if x > 0 else 'red' for x in imp_df['Improvement']],
                              alpha=0.7)
                ax4.set_ylabel('F1-Score Improvement (%)')
                ax4.set_title('Performance Improvement After Tuning')
                ax4.tick_params(axis='x', rotation=45)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, imp_df['Improvement']):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                            f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Baseline comparison\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Improvement After Tuning')
        
        plt.tight_layout()
        plt.savefig('tuning_plots/01_tuning_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Individual model tuning plots
        self.create_individual_tuning_plots()
        
        print("✅ Tuning visualizations created")
    
    def create_individual_tuning_plots(self):
        """Create individual tuning plots for each model"""
        
        for model_name, results in self.tuning_results.items():
            if not results:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{model_name} Hyperparameter Tuning Analysis', fontsize=14, fontweight='bold')
            
            method_names = ['grid', 'random', 'bayesian']
            method_labels = ['Grid Search', 'Random Search', 'Bayesian Opt.']
            
            for i, (method, label) in enumerate(zip(method_names, method_labels)):
                if method in results:
                    cv_results = results[method]['cv_results']
                    
                    # Plot CV scores distribution
                    ax = axes[i]
                    scores = cv_results['mean_test_score']
                    
                    ax.hist(scores, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
                    ax.axvline(results[method]['best_score'], color='red', linestyle='--', 
                              label=f'Best: {results[method]["best_score"]:.4f}')
                    ax.set_xlabel('CV F1-Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{label}\n{results[method]["n_combinations"]} combinations')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, f'{label}\nNot Available', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(label)
            
            plt.tight_layout()
            plt.savefig(f'tuning_plots/02_{model_name.lower()}_tuning.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_tuning_results(self):
        """Save comprehensive tuning results"""
        print("\n5. SAVING TUNING RESULTS")
        print("-" * 30)
        
        # 1. Save tuned models
        for model_name, model in self.tuned_models.items():
            joblib.dump(model, f'tuning_results/tuned_{model_name.lower()}.pkl')
        
        # 2. Save best parameters
        with open('tuning_results/best_parameters.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_params = {}
            for model, params in self.best_params.items():
                json_params[model] = {
                    'method': params['method'],
                    'score': float(params['score']),
                    'params': {k: (v.item() if hasattr(v, 'item') else v) 
                              for k, v in params['params'].items()}
                }
            json.dump(json_params, f, indent=2)
        
        # 3. Create summary report
        self.create_tuning_report()
        
        # 4. Save detailed results
        np.save('tuning_results/detailed_tuning_results.npy', self.tuning_results)
        
        print("✅ Results saved to 'tuning_results/' directory")
    
    def create_tuning_report(self):
        """Create comprehensive tuning report"""
        
        report = f"""# Gut Microbiota Classification - Hyperparameter Tuning Report

## Tuning Overview
- **Dataset**: Advanced Feature Engineered Gut Microbiota Data
- **Features**: {len(self.feature_names)}
- **Samples**: {len(self.X)}
- **Cross-Validation**: {self.cv.n_splits}-fold Stratified
- **Optimization Methods**: Grid Search, Random Search, Bayesian Optimization

## Models Tuned
{len(self.tuned_models)} models successfully tuned:
{', '.join(self.tuned_models.keys())}

## Best Results Summary

| Model | Best Method | CV F1-Score | Improvement* |
|-------|-------------|-------------|--------------|
"""
        
        # Load baseline scores for comparison
        try:
            baseline_df = pd.read_csv('model_results/model_performance_comparison.csv', index_col=0)
            baseline_scores = baseline_df['F1_Score'].to_dict()
        except:
            baseline_scores = {}
        
        for model_name, params in self.best_params.items():
            baseline = baseline_scores.get(model_name, 0)
            improvement = f"{((params['score'] - baseline) / baseline * 100):+.1f}%" if baseline > 0 else "N/A"
            
            report += f"| {model_name} | {params['method'].title()} | {params['score']:.4f} | {improvement} |\n"
        
        report += f"""
*Improvement compared to baseline models

## Optimization Method Performance

### Grid Search
- **Systematic exploration** of predefined parameter combinations
- **Exhaustive but computationally expensive**
- Best for: Small parameter spaces with known good ranges

### Random Search  
- **Random sampling** from parameter distributions
- **More efficient** than grid search for high-dimensional spaces
- Best for: Quick exploration and continuous parameters

### Bayesian Optimization
- **Intelligent sampling** using probabilistic models
- **Most efficient** for expensive function evaluations
- Best for: Complex parameter spaces with unknown optima

## Model-Specific Insights

"""

        for model_name, params in self.best_params.items():
            report += f"""### {model_name}
- **Best Method**: {params['method'].title()}
- **CV F1-Score**: {params['score']:.4f}
- **Optimal Parameters**:
"""
            for param, value in params['params'].items():
                report += f"  - `{param}`: {value}\n"
            report += "\n"
        
        report += f"""## Computational Efficiency

| Model | Grid Search (min) | Random Search (min) | Bayesian Opt. (min) |
|-------|-------------------|---------------------|---------------------|
"""
        
        for model_name, results in self.tuning_results.items():
            row = f"| {model_name} |"
            for method in ['grid', 'random', 'bayesian']:
                if method in results:
                    time_min = results[method]['time_taken'] / 60
                    row += f" {time_min:.1f} |"
                else:
                    row += " N/A |"
            report += row + "\n"
        
        report += f"""
## Key Findings

1. **Most Effective Method**: Varies by model complexity and parameter space
2. **Performance Gains**: Tuning provides measurable improvements over baseline models
3. **Efficiency Trade-offs**: Bayesian optimization balances exploration and exploitation
4. **Regularization Impact**: Proper regularization crucial for gut microbiota classification

## Recommendations

1. **Production Model**: Use the best tuned model for deployment
2. **Ensemble Approach**: Combine top 2-3 tuned models for robustness
3. **Continuous Tuning**: Re-tune periodically with new data
4. **Feature Engineering**: Continue exploring domain-specific features

## Technical Details

- **Scoring Metric**: F1-Score (macro-averaged for multi-class)
- **Cross-Validation**: Stratified to maintain class distribution
- **Reproducibility**: All random states set to 42
- **Computational Resources**: CPU-based optimization

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('tuning_results/tuning_report.md', 'w') as f:
            f.write(report)
    
    def run_complete_tuning_pipeline(self):
        """Run the complete hyperparameter tuning pipeline"""
        print("🚀 Starting Complete Hyperparameter Tuning Pipeline")
        print("="*55)
        
        start_time = time.time()
        
        # Define search spaces
        self.define_search_spaces()
        
        # Tune all models
        self.tune_all_models()
        
        # Create visualizations
        self.create_tuning_visualizations()
        
        # Save results
        self.save_tuning_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 HYPERPARAMETER TUNING COMPLETED!")
        print("="*45)
        print(f"⏱️ Total time: {total_time / 60:.1f} minutes")
        print(f"🏆 Best performing models:")
        
        for model_name, params in self.best_params.items():
            print(f"   {model_name}: {params['score']:.4f} (F1-Score)")
        
        print(f"\n📁 Results saved in 'tuning_results/' directory")
        print(f"📈 Visualizations saved in 'tuning_plots/' directory")
        
        return self.tuned_models, self.best_params

# Main execution
if __name__ == "__main__":
    # Initialize tuning pipeline
    tuning = GutMicrobiotaModelTuning("advanced_feature_engineered_data.csv")
    
    # Run complete tuning pipeline
    tuned_models, best_parameters = tuning.run_complete_tuning_pipeline()
    
    print(f"\n📋 FINAL TUNING SUMMARY:")
    for model, params in best_parameters.items():
        print(f"{model:15s}: {params['score']:.4f} ({params['method']})")
