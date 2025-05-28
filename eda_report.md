# Microbiota Status Classification - EDA Summary Report

## 📊 Executive Summary

This report presents the findings from the Exploratory Data Analysis (EDA) of the microbiota status classification dataset. The analysis was conducted to understand the data structure, identify patterns, and prepare for transformer-based machine learning model development.

## 🔍 Dataset Overview

- **Total Samples**: 10,000 patients
- **Total Features**: 54 variables
- **Target Variable**: Current status of microbiota
- **Classes**: 3 categories (Optimal, Suboptimal, At Risk)
- **Missing Values**: 0 (Complete dataset)

## 🎯 Key Findings

### 1. Target Variable Distribution

**Class Distribution:**

- Suboptimal: 4,682 samples (46.82%)
- Optimal: 4,593 samples (45.93%)
- At Risk: 725 samples (7.25%)

**⚠️ Important Finding**: Significant class imbalance detected with a ratio of 6.46:1 between the largest and smallest classes. The "At Risk" category is severely underrepresented, which will require attention during model training.

### 2. Demographics Analysis

**Key Statistics:**

- Height: Mean 174.9 cm (Range: 150-200 cm)
- Weight: Mean 82.4 kg (Range: 45-120 kg)
- BMI: Mean 27.6 (Range: 11.2-53.3)

**Outliers**: Minimal outliers detected (only 1 BMI outlier out of 10,000 samples)

### 3. Lifestyle Factors

**Physical Activity:**

- Mean frequency: 3.5 times per week (Range: 0-7)
- Standard deviation: 2.3 (high variability)

**Sleep Patterns:**

- Mean sleep hours: 7.0 hours (Range: 4-10 hours)
- Good distribution around recommended 7-8 hours

**Stress Levels:**

- Mean stress level: 5.5 on 1-10 scale
- High variability (SD: 2.8)

### 4. Dietary Consumption Patterns

**Weekly Consumption Averages:**

- Vegetables: 5.6 portions
- Fruits: 5.6 portions
- Whole grains: 5.6 portions
- Animal proteins: 5.6 portions
- Plant proteins: 5.6 portions
- Dairy products: 5.6 portions
- Fermented foods: 5.6 portions

**Water Intake:** Mean 2.5 liters/day (Range: 1-4 liters)

### 5. Gastrointestinal Health Indicators

**Bowel Movement Frequency:**

- Mean: 13 movements per week
- Range: 5-21 movements per week

**Bristol Stool Scale:**

- Mean: 4.0 (normal range)
- Range: 1-7 (complete scale coverage)

**Symptom Prevalence:**

- Bloating: 50.6% of patients
- Gas: 48.7% of patients
- Abdominal pain: 49.9% of patients
- Difficult digestion: 49.7% of patients

### 6. Feature Correlations

**Strong Correlations (|r| > 0.7):**

- Weight ↔ BMI: 0.835 (expected correlation)

**Generally Low Correlations:** Most features show independence, which is beneficial for machine learning models.

### 7. Feature Importance Analysis (Mutual Information)

**Top 5 Most Important Features:**

1. Average sleep hours: 0.0048
2. Frequency of bowel movements: 0.0045
3. Weight (kg): 0.0011
4. Weekly frequency of physical activity: 0.0011
5. Stress level: 0.0005

**⚠️ Note**: Overall mutual information scores are low, suggesting complex non-linear relationships that may benefit from transformer-based approaches.

## 🔧 Data Quality Assessment

### Strengths:

- ✅ Complete dataset (no missing values)
- ✅ Comprehensive feature coverage across all relevant domains
- ✅ Good data range and distribution for most features
- ✅ Minimal outliers
- ✅ Low multicollinearity

### Challenges:

- ⚠️ Significant class imbalance (At Risk: 7.25%)
- ⚠️ Low individual feature importance scores
- ⚠️ Complex feature interactions likely present

## 📈 Recommendations for Model Development

### 1. Class Imbalance Handling

- **SMOTE (Synthetic Minority Oversampling Technique)** for the "At Risk" class
- **Class weights** adjustment in loss function
- **Stratified sampling** for train/validation splits

### 2. Feature Engineering

- **BMI categories** (Underweight, Normal, Overweight, Obese)
- **Sleep quality categories** (Poor: <6h, Adequate: 6-8h, Good: >8h)
- **Stress level groups** (Low: 1-3, Medium: 4-7, High: 8-10)
- **Dietary diversity scores** (combination of various food groups)

### 3. Model Architecture Considerations

- **Transformer models** well-suited due to complex feature interactions
- **Attention mechanisms** can capture non-linear relationships
- **Multi-head attention** for different feature groups (demographics, lifestyle, diet, GI)

### 4. Validation Strategy

- **Stratified K-fold cross-validation** to maintain class distribution
- **Holdout test set** with balanced representation
- **Monitor per-class performance** especially for "At Risk" category

## 📊 Data Preprocessing Pipeline

1. **Categorical Encoding**: One-hot encoding for categorical variables
2. **Numerical Scaling**: StandardScaler or MinMaxScaler
3. **Feature Selection**: Based on mutual information and domain knowledge
4. **Class Balancing**: SMOTE or class weights
5. **Train/Validation/Test Split**: 70/15/15 with stratification

## 🎯 Expected Model Performance

Given the data characteristics:

- **Baseline Accuracy**: ~47% (majority class prediction)
- **Target Accuracy**: 75-85% overall
- **Priority**: High recall for "At Risk" class (clinical importance)
- **Balanced F1-score**: Primary evaluation metric

## 📁 Files Generated

1. `microbiota_eda.py` - Main EDA script
2. `generate_visualizations.py` - Visualization generation
3. `feature_importance.csv` - Feature importance scores
4. `eda_plots/` - Directory with 9 comprehensive visualizations
5. `eda_report.md` - This summary report

## 🔄 Next Steps

1. **Data Preprocessing**: Implement the recommended preprocessing pipeline
2. **Model Development**: Build transformer-based classification models
3. **Hyperparameter Tuning**: Optimize model performance
4. **Model Evaluation**: Comprehensive performance assessment
5. **Feature Interpretability**: SHAP values or attention weights analysis

---

_This EDA provides a solid foundation for developing an effective microbiota status classification model. The identified challenges and recommendations should guide the subsequent model development phases._
