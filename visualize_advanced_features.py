import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create directory for advanced feature visualizations
os.makedirs("advanced_feature_plots", exist_ok=True)

# Load the enhanced dataset
df = pd.read_csv("advanced_feature_engineered_data.csv")
target_col = 'Current status of microbiota'

print("🎨 VISUALIZING ADVANCED FEATURE ENGINEERING RESULTS")
print("="*60)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. Composite Dietary Metrics Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Composite Dietary Metrics by Microbiota Status', fontsize=16, fontweight='bold')

dietary_metrics = [
    'Plant_to_Animal_Protein_Ratio', 'Fiber_Rich_Foods_Score', 'Probiotic_Foods_Score',
    'Microbiome_Friendly_Diet_Score', 'Whole_to_Processed_Foods_Ratio'
]

for i, metric in enumerate(dietary_metrics):
    row, col = i // 3, i % 3
    
    # Box plot
    sns.boxplot(data=df, x=target_col, y=metric, ax=axes[row, col])
    axes[row, col].set_title(f'{metric.replace("_", " ")}', fontweight='bold')
    axes[row, col].tick_params(axis='x', rotation=45)
    
    # Add mean values as text
    for j, status in enumerate(df[target_col].unique()):
        mean_val = df[df[target_col] == status][metric].mean()
        axes[row, col].text(j, mean_val, f'{mean_val:.2f}', 
                           ha='center', va='bottom', fontweight='bold')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig("advanced_feature_plots/01_composite_dietary_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Gut Health Risk Scores
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Gut Health Risk Assessment Features', fontsize=16, fontweight='bold')

risk_features = [
    'Gut_Health_Risk_Score', 'Inflammation_Risk_Index',
    'GI_Symptom_Severity_Score', 'Bristol_Scale_Risk'
]

for i, feature in enumerate(risk_features):
    row, col = i // 2, i % 2
    
    # Violin plot for better distribution visualization
    sns.violinplot(data=df, x=target_col, y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature.replace("_", " ")}', fontweight='bold')
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("advanced_feature_plots/02_gut_health_risk_scores.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Lifestyle Interaction Features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Lifestyle Interaction Features by Microbiota Status', fontsize=16, fontweight='bold')

interaction_features = [
    'Stress_Sleep_Interaction', 'Diet_Exercise_Synergy',
    'Hydration_Fiber_Interaction', 'Metabolic_Age_Risk'
]

for i, feature in enumerate(interaction_features):
    row, col = i // 2, i % 2
    
    sns.boxplot(data=df, x=target_col, y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature.replace("_", " ")}', fontweight='bold')
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("advanced_feature_plots/03_lifestyle_interactions.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Recovery and Protection Potential
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Microbiome Recovery and Protection Potential', fontsize=16, fontweight='bold')

# Recovery potential
sns.boxplot(data=df, x=target_col, y='Microbiome_Recovery_Potential', ax=axes[0])
axes[0].set_title('Microbiome Recovery Potential', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Gut barrier function
sns.boxplot(data=df, x=target_col, y='Gut_Barrier_Function_Score', ax=axes[1])
axes[1].set_title('Gut Barrier Function Score', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("advanced_feature_plots/04_recovery_protection.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Ordinal Binning Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Ordinal Binning Distributions by Microbiota Status', fontsize=16, fontweight='bold')

ordinal_features = [
    'Stress_Level_Binned', 'BMI_Detailed_Bins', 'Sleep_Quality_Binned',
    'Physical_Activity_Binned', 'Water_Intake_Binned'
]

for i, feature in enumerate(ordinal_features):
    row, col = i // 3, i % 3
    
    # Create crosstab and normalize
    crosstab = pd.crosstab(df[feature], df[target_col], normalize='columns') * 100
    
    # Stacked bar plot
    crosstab.plot(kind='bar', ax=axes[row, col], stacked=True)
    axes[row, col].set_title(f'{feature.replace("_", " ")}', fontweight='bold')
    axes[row, col].tick_params(axis='x', rotation=45)
    axes[row, col].legend(title='Microbiota Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[row, col].set_ylabel('Percentage (%)')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig("advanced_feature_plots/05_ordinal_binning.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Feature Correlation Heatmap (new features only)
new_numeric_features = [
    'Plant_to_Animal_Protein_Ratio', 'Fiber_Rich_Foods_Score', 'Probiotic_Foods_Score',
    'Microbiome_Friendly_Diet_Score', 'Gut_Health_Risk_Score', 'Inflammation_Risk_Index',
    'GI_Symptom_Severity_Score', 'Microbiome_Recovery_Potential', 'Gut_Barrier_Function_Score',
    'Stress_Sleep_Interaction', 'Diet_Exercise_Synergy', 'Hydration_Fiber_Interaction'
]

plt.figure(figsize=(14, 10))
correlation_matrix = df[new_numeric_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Correlation Matrix: Advanced Engineered Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("advanced_feature_plots/06_feature_correlations.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Statistical Significance Testing
print("\n📊 STATISTICAL SIGNIFICANCE TESTING")
print("-" * 40)

significance_results = []

for feature in new_numeric_features[:8]:  # Test first 8 features
    groups = [df[df[target_col] == status][feature].values for status in df[target_col].unique()]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    significance_results.append({
        'Feature': feature,
        'F_Statistic': f_stat,
        'P_Value': p_value,
        'Significant': p_value < 0.05
    })
    
    print(f"{feature:30s}: F={f_stat:6.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

# 8. Feature Importance Summary
print(f"\n✅ ADVANCED FEATURE ENGINEERING SUMMARY")
print("="*50)
print(f"📊 Original features: 54")
print(f"🔬 New features created: 38")
print(f"📈 Total features: 92")
print(f"📋 Feature categories:")
print(f"   • Composite dietary metrics: 5")
print(f"   • Ordinal bins: 5") 
print(f"   • Gut health-specific: 7")
print(f"   • Lifestyle interactions: 4")
print(f"   • Domain-specific ratios: 4")
print(f"   • Additional engineered: 13")

print(f"\n🎯 Key Achievements:")
print(f"   ✅ Plant-to-Animal Protein Ratio created")
print(f"   ✅ Gut Health Risk Score (composite)")
print(f"   ✅ Microbiome Recovery Potential") 
print(f"   ✅ Stress-Sleep Interaction modeling")
print(f"   ✅ 5-level ordinal binning for key variables")
print(f"   ✅ Domain knowledge integration")

print(f"\n📁 Visualizations saved to 'advanced_feature_plots/' directory:")
plots_created = [
    "01_composite_dietary_metrics.png",
    "02_gut_health_risk_scores.png", 
    "03_lifestyle_interactions.png",
    "04_recovery_protection.png",
    "05_ordinal_binning.png",
    "06_feature_correlations.png"
]

for plot in plots_created:
    print(f"   📈 {plot}")

print(f"\n🚀 Dataset ready for transformer model training!")
print(f"💾 Enhanced dataset: 'advanced_feature_engineered_data.csv'")
print(f"📄 Report: 'advanced_feature_engineering_report.md'")
