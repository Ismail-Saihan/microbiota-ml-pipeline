import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("health_data_10000_chunk - health_data_10000_chunk.csv")
target_column = 'Current status of microbiota'

print("🔍 ADDITIONAL INSIGHTS FROM EDA")
print("="*60)

# 1. Medical Conditions Analysis
print("\n1. MEDICAL CONDITIONS ANALYSIS")
print("-" * 40)

# Parse medical conditions
medical_conditions = []
for conditions in df['Medical conditions'].dropna():
    if isinstance(conditions, str):
        # Split by comma and clean
        conds = [c.strip() for c in conditions.split(',')]
        medical_conditions.extend(conds)

from collections import Counter
condition_counts = Counter(medical_conditions)
print("Most common medical conditions:")
for condition, count in condition_counts.most_common(10):
    percentage = (count / len(df)) * 100
    print(f"  {condition}: {count} patients ({percentage:.1f}%)")

# 2. Dietary Pattern Analysis
print("\n2. DIETARY PATTERN ANALYSIS")
print("-" * 40)

dietary_cols = [
    'Weekly consumption of vegetables (portions)',
    'Weekly consumption of fruits (portions)',
    'Weekly consumption of whole grains (portions)',
    'Weekly consumption of animal proteins (portions)',
    'Weekly consumption of plant proteins (portions)',
    'Weekly consumption of dairy products (portions)',
    'Weekly consumption of fermented foods (portions)'
]

# Calculate total dietary diversity score
df['Dietary_Diversity_Score'] = df[dietary_cols].sum(axis=1)

print("Dietary diversity by microbiota status:")
for status in df[target_column].unique():
    subset = df[df[target_column] == status]
    mean_score = subset['Dietary_Diversity_Score'].mean()
    std_score = subset['Dietary_Diversity_Score'].std()
    print(f"  {status}: {mean_score:.1f} ± {std_score:.1f} portions/week")

# 3. Lifestyle Risk Factors
print("\n3. LIFESTYLE RISK FACTORS")
print("-" * 40)

# Define risk categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def categorize_stress(stress):
    if stress <= 3:
        return "Low"
    elif stress <= 7:
        return "Medium"
    else:
        return "High"

def categorize_sleep(sleep):
    if sleep < 6:
        return "Poor"
    elif sleep <= 8:
        return "Adequate"
    else:
        return "Excessive"

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
df['Stress_Category'] = df['Stress level (1-10 scale)'].apply(categorize_stress)
df['Sleep_Category'] = df['Average sleep hours (hours)'].apply(categorize_sleep)

print("Risk factor distribution by microbiota status:")

# BMI categories
print("\nBMI Categories:")
bmi_cross = pd.crosstab(df['BMI_Category'], df[target_column], normalize='columns') * 100
for status in df[target_column].unique():
    print(f"\n{status}:")
    for bmi_cat in bmi_cross.index:
        pct = bmi_cross.loc[bmi_cat, status]
        print(f"  {bmi_cat}: {pct:.1f}%")

# 4. Gastrointestinal Symptom Clustering
print("\n4. GASTROINTESTINAL SYMPTOM ANALYSIS")
print("-" * 40)

gi_symptoms = ['Presence of bloating', 'Presence of gas', 'Presence of abdominal pain', 'Difficult digestion']

# Count total symptoms per patient
df['Total_GI_Symptoms'] = df[gi_symptoms].sum(axis=1)

print("GI symptom burden by microbiota status:")
for status in df[target_column].unique():
    subset = df[df[target_column] == status]
    mean_symptoms = subset['Total_GI_Symptoms'].mean()
    print(f"  {status}: {mean_symptoms:.2f} symptoms on average")

# Symptom combinations
print("\nSymptom burden distribution:")
symptom_counts = df['Total_GI_Symptoms'].value_counts().sort_index()
for symptoms, count in symptom_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {symptoms} symptoms: {count} patients ({percentage:.1f}%)")

# 5. Physical Activity and Health Correlation
print("\n5. PHYSICAL ACTIVITY INSIGHTS")
print("-" * 40)

def categorize_activity(freq):
    if freq == 0:
        return "Sedentary"
    elif freq <= 2:
        return "Low Active"
    elif freq <= 4:
        return "Moderately Active"
    else:
        return "Highly Active"

df['Activity_Category'] = df['Weekly frequency of physical activity (per week)'].apply(categorize_activity)

print("Physical activity distribution by microbiota status:")
activity_cross = pd.crosstab(df['Activity_Category'], df[target_column], normalize='columns') * 100
for status in df[target_column].unique():
    print(f"\n{status}:")
    for activity_cat in ['Sedentary', 'Low Active', 'Moderately Active', 'Highly Active']:
        if activity_cat in activity_cross.index:
            pct = activity_cross.loc[activity_cat, status]
            print(f"  {activity_cat}: {pct:.1f}%")

# 6. Combined Risk Score
print("\n6. COMBINED RISK ASSESSMENT")
print("-" * 40)

# Create a simple risk score
risk_score = 0

# BMI risk (1 point for overweight, 2 for obese)
df['BMI_Risk'] = df['BMI_Category'].map({'Underweight': 1, 'Normal': 0, 'Overweight': 1, 'Obese': 2})

# Stress risk (1 point for high stress)
df['Stress_Risk'] = (df['Stress level (1-10 scale)'] > 7).astype(int)

# Sleep risk (1 point for poor sleep)
df['Sleep_Risk'] = (df['Average sleep hours (hours)'] < 6).astype(int)

# Activity risk (1 point for sedentary)
df['Activity_Risk'] = (df['Weekly frequency of physical activity (per week)'] == 0).astype(int)

# GI symptom risk (1 point for 3+ symptoms)
df['GI_Risk'] = (df['Total_GI_Symptoms'] >= 3).astype(int)

# Total risk score
df['Total_Risk_Score'] = (df['BMI_Risk'] + df['Stress_Risk'] + 
                         df['Sleep_Risk'] + df['Activity_Risk'] + df['GI_Risk'])

print("Risk score distribution by microbiota status:")
for status in df[target_column].unique():
    subset = df[df[target_column] == status]
    mean_risk = subset['Total_Risk_Score'].mean()
    print(f"  {status}: {mean_risk:.2f} average risk score")

print("\nRisk score breakdown (0-7 scale):")
risk_cross = pd.crosstab(df['Total_Risk_Score'], df[target_column], normalize='columns') * 100
for score in sorted(df['Total_Risk_Score'].unique()):
    print(f"\nRisk Score {score}:")
    for status in df[target_column].unique():
        if score in risk_cross.index:
            pct = risk_cross.loc[score, status]
            print(f"  {status}: {pct:.1f}%")

# Save enhanced dataset with new features
df.to_csv("enhanced_microbiota_data.csv", index=False)
print(f"\n💾 Enhanced dataset saved with {len(df.columns)} features (added {len(df.columns) - 54} new features)")

print("\n✅ Additional insights analysis completed!")
print("\nNew Features Created:")
print("- Dietary_Diversity_Score")
print("- BMI_Category, Stress_Category, Sleep_Category, Activity_Category")
print("- Total_GI_Symptoms")
print("- Individual risk scores (BMI_Risk, Stress_Risk, Sleep_Risk, Activity_Risk, GI_Risk)")
print("- Total_Risk_Score (composite risk assessment)")
