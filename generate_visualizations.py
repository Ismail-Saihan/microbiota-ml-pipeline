import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Create output directory for plots
output_dir = "eda_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data
df = pd.read_csv("health_data_10000_chunk - health_data_10000_chunk.csv")
target_column = 'Current status of microbiota'

print("🎨 Generating EDA Visualizations...")

# 1. Target Variable Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
target_counts = df[target_column].value_counts()

# Count plot
target_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0].set_title('Target Variable Distribution (Count)', fontsize=14)
axes[0].set_xlabel('Microbiota Status')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = axes[1].pie(target_counts.values, 
                                      labels=target_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
axes[1].set_title('Target Variable Distribution (Proportion)', fontsize=14)

plt.tight_layout()
plt.savefig(f"{output_dir}/01_target_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Demographic Analysis
demographic_cols = ['Height (cm)', 'Weight (kg)', 'BMI']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, col in enumerate(demographic_cols):
    # Distribution plot
    axes[0, i].hist(df[col], bins=30, alpha=0.7, color=f'C{i}')
    axes[0, i].set_title(f'{col} Distribution')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('Frequency')
    
    # Box plot by target
    sns.boxplot(data=df, x=target_column, y=col, ax=axes[1, i])
    axes[1, i].set_title(f'{col} by Microbiota Status')
    axes[1, i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/02_demographic_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Lifestyle Analysis
numeric_lifestyle = [
    'Weekly frequency of physical activity (per week)',
    'Average sleep hours (hours)',
    'Stress level (1-10 scale)'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(numeric_lifestyle):
    # Distribution by target
    for status in df[target_column].unique():
        subset = df[df[target_column] == status]
        axes[i].hist(subset[col], alpha=0.6, label=status, bins=20)
    axes[i].set_title(f'{col} by Microbiota Status')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/03_lifestyle_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Dietary Analysis - Correlation Matrix
dietary_cols = [
    'Weekly consumption of vegetables (portions)',
    'Weekly consumption of fruits (portions)',
    'Weekly consumption of whole grains (portions)',
    'Weekly consumption of animal proteins (portions)',
    'Weekly consumption of plant proteins (portions)',
    'Weekly consumption of dairy products (portions)',
    'Weekly consumption of fermented foods (portions)',
    'Daily water intake (liters)'
]

dietary_corr = df[dietary_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(dietary_corr, dtype=bool))
sns.heatmap(dietary_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
           square=True, fmt='.2f')
plt.title('Dietary Factors Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/04_dietary_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Dietary Patterns by Microbiota Status
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(dietary_cols):
    sns.boxplot(data=df, x=target_column, y=col, ax=axes[i])
    axes[i].set_title(col.replace('Weekly consumption of ', '').replace('Daily ', ''))
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/05_dietary_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Gastrointestinal Health Analysis
gi_numeric = ['Frequency of bowel movements', 'Stool consistency (Bristol scale)']
gi_boolean = ['Presence of bloating', 'Presence of gas', 'Presence of abdominal pain', 'Difficult digestion']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Numeric GI by target
for i, col in enumerate(gi_numeric):
    sns.boxplot(data=df, x=target_column, y=col, ax=axes[i])
    axes[i].set_title(f'{col} by Microbiota Status')
    axes[i].tick_params(axis='x', rotation=45)

# Boolean GI by target
for i, col in enumerate(gi_boolean):
    if i+2 < len(axes):
        # Create contingency table
        ct = pd.crosstab(df[col], df[target_column], normalize='columns') * 100
        ct.plot(kind='bar', ax=axes[i+2], stacked=True)
        axes[i+2].set_title(f'{col} by Microbiota Status (%)')
        axes[i+2].tick_params(axis='x', rotation=45)
        axes[i+2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f"{output_dir}/06_gastrointestinal_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Overall Feature Correlation Matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Meal times']
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
           square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/07_feature_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. Feature Importance Visualization
# Load feature importance from the saved CSV
feature_importance = pd.read_csv("feature_importance.csv")

plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Mutual_Information'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Mutual Information Score')
plt.title('Top 15 Feature Importance (Mutual Information)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/08_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. Class Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Class counts by different categories
# By BMI categories
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, float('inf')], 
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
ct1 = pd.crosstab(df['BMI_Category'], df[target_column], normalize='index') * 100
ct1.plot(kind='bar', ax=axes[0,0], stacked=True)
axes[0,0].set_title('Microbiota Status by BMI Category (%)')
axes[0,0].tick_params(axis='x', rotation=45)

# By stress level categories
df['Stress_Category'] = pd.cut(df['Stress level (1-10 scale)'], bins=[0, 3, 7, 10], 
                              labels=['Low', 'Medium', 'High'])
ct2 = pd.crosstab(df['Stress_Category'], df[target_column], normalize='index') * 100
ct2.plot(kind='bar', ax=axes[0,1], stacked=True)
axes[0,1].set_title('Microbiota Status by Stress Level (%)')
axes[0,1].tick_params(axis='x', rotation=45)

# By sleep hours
df['Sleep_Category'] = pd.cut(df['Average sleep hours (hours)'], bins=[0, 6, 8, 12], 
                             labels=['Poor', 'Adequate', 'Good'])
ct3 = pd.crosstab(df['Sleep_Category'], df[target_column], normalize='index') * 100
ct3.plot(kind='bar', ax=axes[1,0], stacked=True)
axes[1,0].set_title('Microbiota Status by Sleep Quality (%)')
axes[1,0].tick_params(axis='x', rotation=45)

# By exercise frequency
df['Exercise_Category'] = pd.cut(df['Weekly frequency of physical activity (per week)'], 
                                bins=[0, 2, 5, 7], labels=['Low', 'Moderate', 'High'])
ct4 = pd.crosstab(df['Exercise_Category'], df[target_column], normalize='index') * 100
ct4.plot(kind='bar', ax=axes[1,1], stacked=True)
axes[1,1].set_title('Microbiota Status by Exercise Frequency (%)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/09_class_distribution_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All visualizations saved to '{output_dir}' directory")
print("\n📊 Generated Plots:")
print("1. Target Variable Distribution")
print("2. Demographic Analysis (Height, Weight, BMI)")
print("3. Lifestyle Analysis (Exercise, Sleep, Stress)")
print("4. Dietary Correlation Matrix")
print("5. Dietary Patterns by Microbiota Status")
print("6. Gastrointestinal Health Analysis")
print("7. Overall Feature Correlation Matrix")
print("8. Feature Importance (Top 15)")
print("9. Class Distribution Analysis by Categories")
