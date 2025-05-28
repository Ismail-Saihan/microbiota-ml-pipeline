#!/usr/bin/env python3
"""
Comprehensive Evaluation Report - Summary Visualizations
Generates additional visualizations for the evaluation report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def create_summary_visualizations():
    """Create additional summary visualizations for the evaluation report"""
    
    # Create output directory
    output_dir = Path("evaluation_summary_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load performance data
    df = pd.read_csv("model_results/model_performance_comparison.csv", index_col=0)
    
    # Create comprehensive performance comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Individual metric plots
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Sort models by metric value
        sorted_data = df[metric].sort_values(ascending=True)
        
        bars = ax.barh(range(len(sorted_data)), sorted_data.values, color=colors[i], alpha=0.7)
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data.index, fontsize=10)
        ax.set_xlabel(metric, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(sorted_data.values) * 1.15)
    
    # Overall ranking plot
    ax = axes[1, 2]
    
    # Calculate overall ranking based on F1-Score (most important metric)
    ranking_data = df['F1_Score'].sort_values(ascending=False)
    
    bars = ax.barh(range(len(ranking_data)), ranking_data.values, 
                   color=['#d4af37', '#c0c0c0', '#cd7f32', '#4a4a4a', '#6a6a6a', '#8a8a8a', '#aaaaaa'])
    ax.set_yticks(range(len(ranking_data)))
    ax.set_yticklabels(ranking_data.index, fontsize=10)
    ax.set_xlabel('F1-Score', fontweight='bold')
    ax.set_title('Overall Model Ranking (by F1-Score)', fontweight='bold')
    
    # Add ranking labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'#{i+1}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(ranking_data.values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_comprehensive_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Select top 4 models for radar chart
    top_models = df.nlargest(4, 'F1_Score')
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, model_data) in enumerate(top_models.iterrows()):
        values = [model_data[metric] for metric in metrics_radar]
        values += [values[0]]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors_radar[i])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=12)
    ax.set_ylim(0, 0.6)
    ax.set_title('Top 4 Models - Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.savefig(output_dir / "02_performance_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metric correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[metrics].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=ax, cbar_kws={"shrink": .8})
    
    ax.set_title('Performance Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "03_metrics_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create model category comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define model categories
    model_categories = {
        'Traditional ML': ['Logistic_Regression', 'Random_Forest'],
        'Gradient Boosting': ['XGBoost', 'LightGBM'],
        'Deep Learning': ['Neural_Network'],
        'Transformers': ['TabTransformer', 'LSTM_Transformer']
    }
    
    category_performance = {}
    for category, models in model_categories.items():
        category_models = df.loc[models]
        category_performance[category] = {
            'Mean F1-Score': category_models['F1_Score'].mean(),
            'Max F1-Score': category_models['F1_Score'].max(),
            'Mean Accuracy': category_models['Accuracy'].mean(),
            'Max Accuracy': category_models['Accuracy'].max()
        }
    
    category_df = pd.DataFrame(category_performance).T
    
    x = np.arange(len(category_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, category_df['Mean F1-Score'], width, 
                   label='Mean F1-Score', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, category_df['Max F1-Score'], width, 
                   label='Max F1-Score', alpha=0.8, color='#ff7f0e')
    
    ax.set_xlabel('Model Category', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('Model Performance by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_model_category_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualizations saved to {output_dir}/")
    print("Generated files:")
    print("- 01_comprehensive_performance_analysis.png")
    print("- 02_performance_radar_chart.png") 
    print("- 03_metrics_correlation_heatmap.png")
    print("- 04_model_category_comparison.png")

if __name__ == "__main__":
    create_summary_visualizations()
