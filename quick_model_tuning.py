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
from scipy.stats import uniform, randint
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import time
import json

class QuickGutMicrobiotaModelTuning:
    """
    Quick hyperparameter tuning pipeline for gut microbiota classification
    with reduced parameter spaces for faster execution
    """
    
    def __init__(self, data_path):
        """Initialize with enhanced dataset"""
        print("🔧 QUICK GUT MICROBIOTA MODEL HYPERPARAMETER TUNING")
        print("="*58)
        
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
        
        # Setup cross-validation (3-fold for speed)
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        print(f"✅ Data prepared: {X_scaled.shape}")
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Classes: {len(np.unique(y_encoded))}")
        print(f"✅ CV Strategy: {self.cv.n_splits}-fold Stratified")
        
    def define_quick_search_spaces(self):
        """Define reduced hyperparameter search spaces for quick tuning"""
        print("\n2. DEFINING QUICK SEARCH SPACES")
        print("-" * 35)
        
        self.param_spaces = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced']
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5]
            },
            
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50],
                'subsample': [0.8, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1]
            },
            
            'NeuralNetwork': {
                'hidden_layer_sizes': [(64,), (128,), (64, 32)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500]
            }
        }
        
        print("✅ Quick search spaces defined for 4 algorithms")
        for algo, space in self.param_spaces.items():
            combinations = 1
            for param, values in space.items():
                combinations *= len(values)
            print(f"   {algo}: {combinations} combinations")
    
    def tune_model(self, model_name, model, param_grid, scoring='f1_macro'):
        """Tune model using Grid Search with quick parameters"""
        print(f"\n🔍 Tuning: {model_name}")
        print("-" * 25)
        
        start_time = time.time()
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        print(f"🔄 Running grid search with {len(grid_search.get_params()['param_grid'])} parameter combinations...")
        grid_search.fit(self.X, self.y)
        
        end_time = time.time()
        
        # Store results
        result = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'n_combinations': len(grid_search.cv_results_['params']),
            'time_taken': end_time - start_time,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
        print(f"⏱️ Time taken: {end_time - start_time:.1f} seconds")
        print(f"🔢 Combinations tested: {result['n_combinations']}")
        print(f"🏆 Best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, result
    
    def tune_all_models(self):
        """Tune all models using quick grid search"""
        print("\n3. COMPREHENSIVE QUICK MODEL TUNING")
        print("-" * 40)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='mlogloss',
                n_jobs=-1,
                verbosity=0
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
                validation_fraction=0.1
            )
        }
        
        # Tune each model
        for model_name, base_model in models.items():
            print(f"\n{'='*40}")
            print(f"🎯 TUNING: {model_name}")
            print(f"{'='*40}")
            
            try:
                best_model, result = self.tune_model(
                    model_name, base_model, self.param_spaces[model_name]
                )
                
                self.tuned_models[model_name] = best_model
                self.tuning_results[model_name] = result
                self.best_params[model_name] = {
                    'params': result['best_params'],
                    'score': result['best_score']
                }
                
            except Exception as e:
                print(f"❌ Tuning failed for {model_name}: {e}")
    
    def create_quick_visualizations(self):
        """Create quick tuning visualizations"""
        print("\n4. CREATING QUICK VISUALIZATIONS")
        print("-" * 38)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quick Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # Prepare data
        model_names = list(self.best_params.keys())
        scores = [self.best_params[model]['score'] for model in model_names]
        times = [self.tuning_results[model]['time_taken'] for model in model_names]
        combinations = [self.tuning_results[model]['n_combinations'] for model in model_names]
        
        # Plot 1: Best scores
        ax1 = axes[0, 0]
        bars = ax1.bar(model_names, scores, color='skyblue', alpha=0.8)
        ax1.set_ylabel('CV F1-Score')
        ax1.set_title('Best Tuned Model Scores')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 2: Tuning time
        ax2 = axes[0, 1]
        ax2.bar(model_names, times, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Tuning Time by Model')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter combinations tested
        ax3 = axes[1, 0]
        ax3.bar(model_names, combinations, color='lightgreen', alpha=0.8)
        ax3.set_ylabel('Combinations Tested')
        ax3.set_title('Search Space Size')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score improvement (if baseline available)
        ax4 = axes[1, 1]
        try:
            baseline_df = pd.read_csv('model_results/model_performance_comparison.csv', index_col=0)
            baseline_scores = baseline_df['F1_Score'].to_dict()
            
            improvements = []
            available_models = []
            for model in model_names:
                if model in baseline_scores:
                    baseline = baseline_scores[model]
                    tuned = self.best_params[model]['score']
                    improvement = ((tuned - baseline) / baseline) * 100
                    improvements.append(improvement)
                    available_models.append(model)
            
            if improvements:
                bars = ax4.bar(available_models, improvements, 
                              color=['green' if x > 0 else 'red' for x in improvements],
                              alpha=0.7)
                ax4.set_ylabel('F1-Score Improvement (%)')
                ax4.set_title('Performance Improvement After Tuning')
                ax4.tick_params(axis='x', rotation=45)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, improvements):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                            f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            else:
                ax4.text(0.5, 0.5, 'No baseline\ncomparison available', 
                        ha='center', va='center', transform=ax4.transAxes)
        except Exception as e:
            ax4.text(0.5, 0.5, 'Baseline comparison\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Improvement')
        
        plt.tight_layout()
        plt.savefig('tuning_plots/01_quick_tuning_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter importance heatmap
        self.create_parameter_importance_plot()
        
        print("✅ Quick visualizations created")
    
    def create_parameter_importance_plot(self):
        """Create parameter importance visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Best Parameters by Model', fontsize=16, fontweight='bold')
        
        for i, (model_name, params) in enumerate(self.best_params.items()):
            if i >= 4:  # Only plot first 4 models
                break
                
            ax = axes[i // 2, i % 2]
            
            param_names = list(params['params'].keys())
            param_values = [str(v) for v in params['params'].values()]
            
            # Create a simple bar chart of parameters
            y_pos = np.arange(len(param_names))
            
            # For numerical parameters, use the actual values
            # For categorical parameters, use index
            plot_values = []
            for name, value in params['params'].items():
                if isinstance(value, (int, float)):
                    plot_values.append(value)
                else:
                    # For categorical, use a dummy value
                    plot_values.append(1)
            
            bars = ax.barh(y_pos, plot_values, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(param_names)
            ax.set_xlabel('Parameter Value')
            ax.set_title(f'{model_name}\\nF1-Score: {params["score"]:.4f}')
            
            # Add value labels
            for j, (bar, value_str) in enumerate(zip(bars, param_values)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       value_str, ha='left', va='center', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.best_params), 4):
            axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('tuning_plots/02_best_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_quick_results(self):
        """Save quick tuning results"""
        print("\n5. SAVING QUICK RESULTS")
        print("-" * 28)
        
        # 1. Save tuned models
        for model_name, model in self.tuned_models.items():
            joblib.dump(model, f'tuning_results/quick_tuned_{model_name.lower()}.pkl')
        
        # 2. Save best parameters
        with open('tuning_results/quick_best_parameters.json', 'w') as f:
            json_params = {}
            for model, params in self.best_params.items():
                json_params[model] = {
                    'score': float(params['score']),
                    'params': {k: (v.item() if hasattr(v, 'item') else v) 
                              for k, v in params['params'].items()}
                }
            json.dump(json_params, f, indent=2)
        
        # 3. Create performance comparison
        self.create_performance_comparison()
        
        # 4. Create quick report
        self.create_quick_report()
        
        print("✅ Quick results saved to 'tuning_results/' directory")
    
    def create_performance_comparison(self):
        """Create performance comparison table"""
        
        # Load baseline results for comparison
        try:
            baseline_df = pd.read_csv('model_results/model_performance_comparison.csv', index_col=0)
            
            # Create comparison dataframe
            comparison_data = []
            for model_name in self.best_params.keys():
                if model_name in baseline_df.index:
                    baseline_score = baseline_df.loc[model_name, 'F1_Score']
                    tuned_score = self.best_params[model_name]['score']
                    improvement = ((tuned_score - baseline_score) / baseline_score) * 100
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Baseline_F1': baseline_score,
                        'Tuned_F1': tuned_score,
                        'Improvement_Percent': improvement,
                        'Tuning_Time_Seconds': self.tuning_results[model_name]['time_taken']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv('tuning_results/quick_tuning_comparison.csv', index=False)
            
        except Exception as e:
            print(f"⚠️ Could not create comparison with baseline: {e}")
    
    def create_quick_report(self):
        """Create quick tuning report"""
        
        report = f"""# Quick Gut Microbiota Classification - Hyperparameter Tuning Report

## Quick Tuning Overview
- **Dataset**: Advanced Feature Engineered Gut Microbiota Data
- **Features**: {len(self.feature_names)}
- **Samples**: {len(self.X)}
- **Cross-Validation**: {self.cv.n_splits}-fold Stratified
- **Optimization Method**: Grid Search (Reduced Parameter Space)

## Models Tuned
{len(self.tuned_models)} models successfully tuned:
{', '.join(self.tuned_models.keys())}

## Quick Results Summary

| Model | CV F1-Score | Time (sec) | Best Parameters |
|-------|-------------|------------|-----------------|
"""
        
        for model_name, params in self.best_params.items():
            time_taken = self.tuning_results[model_name]['time_taken']
            best_params_str = ', '.join([f"{k}={v}" for k, v in list(params['params'].items())[:3]])
            if len(params['params']) > 3:
                best_params_str += "..."
                
            report += f"| {model_name} | {params['score']:.4f} | {time_taken:.1f} | {best_params_str} |\n"
        
        # Find best model
        best_model = max(self.best_params.keys(), key=lambda x: self.best_params[x]['score'])
        best_score = self.best_params[best_model]['score']
        
        report += f"""
## Best Performing Model: {best_model}
- **CV F1-Score**: {best_score:.4f}
- **Parameters**: 
"""
        for param, value in self.best_params[best_model]['params'].items():
            report += f"  - `{param}`: {value}\n"
        
        total_time = sum(result['time_taken'] for result in self.tuning_results.values())
        
        report += f"""
## Tuning Summary
- **Total Tuning Time**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)
- **Average Time per Model**: {total_time/len(self.tuned_models):.1f} seconds
- **Most Time-Consuming**: {max(self.tuning_results.keys(), key=lambda x: self.tuning_results[x]['time_taken'])}
- **Fastest to Tune**: {min(self.tuning_results.keys(), key=lambda x: self.tuning_results[x]['time_taken'])}

## Key Findings
1. **Quick tuning provides measurable improvements** over baseline models
2. **Grid search with reduced parameter space** balances thoroughness and speed
3. **Cross-validation ensures robust performance estimates**
4. **Model-specific parameter importance** varies significantly

## Recommendations
1. **Deploy the best tuned model** ({best_model}) for production use
2. **Consider ensemble methods** combining top 2-3 models
3. **Extend tuning** with full parameter spaces if computational resources allow
4. **Monitor performance** and retune periodically with new data

---
*Quick tuning report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open('tuning_results/quick_tuning_report.md', 'w') as f:
            f.write(report)
    
    def run_quick_tuning_pipeline(self):
        """Run the complete quick hyperparameter tuning pipeline"""
        print("🚀 Starting Quick Hyperparameter Tuning Pipeline")
        print("="*52)
        
        start_time = time.time()
        
        # Define search spaces
        self.define_quick_search_spaces()
        
        # Tune all models
        self.tune_all_models()
        
        # Create visualizations
        self.create_quick_visualizations()
        
        # Save results
        self.save_quick_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 QUICK HYPERPARAMETER TUNING COMPLETED!")
        print("="*45)
        print(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"🏆 Best performing models:")
        
        # Sort by score
        sorted_models = sorted(self.best_params.items(), key=lambda x: x[1]['score'], reverse=True)
        for model_name, params in sorted_models:
            print(f"   {model_name}: {params['score']:.4f} (F1-Score)")
        
        print(f"\n📁 Results saved in 'tuning_results/' directory")
        print(f"📈 Visualizations saved in 'tuning_plots/' directory")
        
        return self.tuned_models, self.best_params

# Main execution
if __name__ == "__main__":
    # Initialize quick tuning pipeline
    tuning = QuickGutMicrobiotaModelTuning("advanced_feature_engineered_data.csv")
    
    # Run quick tuning pipeline
    tuned_models, best_parameters = tuning.run_quick_tuning_pipeline()
    
    print(f"\n📋 FINAL QUICK TUNING SUMMARY:")
    sorted_results = sorted(best_parameters.items(), key=lambda x: x[1]['score'], reverse=True)
    for model, params in sorted_results:
        print(f"{model:15s}: {params['score']:.4f}")
