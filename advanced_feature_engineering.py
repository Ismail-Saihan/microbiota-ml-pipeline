import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for gut microbiota classification
    with domain knowledge-driven composite metrics and ordinal binning
    """
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        print("🔬 ADVANCED FEATURE ENGINEERING FOR GUT MICROBIOTA")
        print("="*65)
        
        # Load enhanced dataset if it exists, otherwise load original
        try:
            self.df = pd.read_csv("enhanced_microbiota_data.csv")
            print("✅ Loading enhanced dataset...")
        except:
            self.df = pd.read_csv(data_path)
            print("✅ Loading original dataset...")
            
        self.target_column = 'Current status of microbiota'
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"🎯 Target classes: {self.df[self.target_column].value_counts().to_dict()}")
        
    def create_composite_dietary_metrics(self):
        """Create gut health-focused composite dietary metrics"""
        print("\n1. CREATING COMPOSITE DIETARY METRICS")
        print("-" * 50)
        
        # Plant-to-Animal Protein Ratio (key for gut microbiome diversity)
        plant_protein = self.df['Weekly consumption of plant proteins (portions)']
        animal_protein = self.df['Weekly consumption of animal proteins (portions)']
        
        # Handle division by zero
        self.df['Plant_to_Animal_Protein_Ratio'] = np.where(
            animal_protein == 0, 
            plant_protein * 10,  # High ratio when no animal protein
            plant_protein / animal_protein
        )
        
        # Fiber-Rich Foods Score (vegetables + fruits + whole grains)
        self.df['Fiber_Rich_Foods_Score'] = (
            self.df['Weekly consumption of vegetables (portions)'] +
            self.df['Weekly consumption of fruits (portions)'] +
            self.df['Weekly consumption of whole grains (portions)']
        )
        
        # Probiotic Foods Score (fermented foods + dairy)
        self.df['Probiotic_Foods_Score'] = (
            self.df['Weekly consumption of fermented foods (portions)'] +
            self.df['Weekly consumption of dairy products (portions)'] * 0.5  # Weight dairy less
        )
        
        # Microbiome-Friendly Diet Score (combines fiber and probiotics)
        self.df['Microbiome_Friendly_Diet_Score'] = (
            self.df['Fiber_Rich_Foods_Score'] * 0.6 +
            self.df['Probiotic_Foods_Score'] * 0.4
        )
        
        # Processed vs Whole Foods Ratio
        whole_foods = (self.df['Weekly consumption of vegetables (portions)'] +
                      self.df['Weekly consumption of fruits (portions)'] +
                      self.df['Weekly consumption of whole grains (portions)'])
        
        processed_foods = (self.df['Weekly consumption of animal proteins (portions)'] +
                          self.df['Weekly consumption of dairy products (portions)'])
        
        self.df['Whole_to_Processed_Foods_Ratio'] = np.where(
            processed_foods == 0,
            whole_foods * 5,
            whole_foods / processed_foods
        )
        
        print("✅ Created 5 composite dietary metrics:")
        print("   - Plant_to_Animal_Protein_Ratio")
        print("   - Fiber_Rich_Foods_Score") 
        print("   - Probiotic_Foods_Score")
        print("   - Microbiome_Friendly_Diet_Score")
        print("   - Whole_to_Processed_Foods_Ratio")
        
    def create_ordinal_bins(self):
        """Convert ordinal data to meaningful bins using domain knowledge"""
        print("\n2. CREATING ORDINAL BINS")
        print("-" * 30)
        
        # Stress Level Bins (gut-brain axis is crucial for microbiome)
        def categorize_stress_detailed(stress):
            if stress <= 2:
                return "Very_Low"
            elif stress <= 4:
                return "Low"
            elif stress <= 6:
                return "Moderate"
            elif stress <= 8:
                return "High"
            else:
                return "Very_High"
                
        self.df['Stress_Level_Binned'] = self.df['Stress level (1-10 scale)'].apply(categorize_stress_detailed)
        
        # BMI Bins (detailed for gut health impact)
        def categorize_bmi_detailed(bmi):
            if bmi < 18.5:
                return "Underweight"
            elif bmi < 23:
                return "Normal_Low"
            elif bmi < 25:
                return "Normal_High"
            elif bmi < 27:
                return "Overweight_Mild"
            elif bmi < 30:
                return "Overweight_Moderate"
            else:
                return "Obese"
                
        self.df['BMI_Detailed_Bins'] = self.df['BMI'].apply(categorize_bmi_detailed)
        
        # Sleep Quality Bins (sleep affects gut microbiome)
        def categorize_sleep_detailed(sleep):
            if sleep < 5:
                return "Very_Poor"
            elif sleep < 6:
                return "Poor"
            elif sleep < 7:
                return "Suboptimal"
            elif sleep <= 8:
                return "Optimal"
            elif sleep <= 9:
                return "Good_Long"
            else:
                return "Excessive"
                
        self.df['Sleep_Quality_Binned'] = self.df['Average sleep hours (hours)'].apply(categorize_sleep_detailed)
        
        # Physical Activity Bins (exercise modulates gut microbiome)
        def categorize_activity_detailed(freq):
            if freq == 0:
                return "Sedentary"
            elif freq <= 1:
                return "Minimal"
            elif freq <= 2:
                return "Low"
            elif freq <= 4:
                return "Moderate"
            elif freq <= 6:
                return "High"
            else:
                return "Very_High"
                
        self.df['Physical_Activity_Binned'] = self.df['Weekly frequency of physical activity (per week)'].apply(categorize_activity_detailed)
        
        # Water Intake Bins (hydration affects gut health)
        def categorize_water_intake(water):
            if water < 1.5:
                return "Low"
            elif water < 2.5:
                return "Adequate"
            elif water < 3.5:
                return "Good"
            else:
                return "High"
                
        self.df['Water_Intake_Binned'] = self.df['Daily water intake (liters)'].apply(categorize_water_intake)
        
        print("✅ Created 5 ordinal bins:")
        print("   - Stress_Level_Binned (5 levels)")
        print("   - BMI_Detailed_Bins (6 levels)")
        print("   - Sleep_Quality_Binned (6 levels)")
        print("   - Physical_Activity_Binned (6 levels)")
        print("   - Water_Intake_Binned (4 levels)")
        
    def create_gut_health_features(self):
        """Create gut health-specific features using domain knowledge"""
        print("\n3. CREATING GUT HEALTH-FOCUSED FEATURES")
        print("-" * 45)
        
        # Bristol Stool Scale Risk (1-2 constipated, 6-7 diarrhea, 3-5 optimal)
        def bristol_risk_score(bristol):
            if bristol in [1, 2]:
                return 2  # Constipation risk
            elif bristol in [6, 7]:
                return 2  # Diarrhea risk
            elif bristol in [3, 4, 5]:
                return 0  # Optimal
            else:
                return 1  # Unknown/missing
                
        self.df['Bristol_Scale_Risk'] = self.df['Stool consistency (Bristol scale)'].apply(bristol_risk_score)
        
        # Bowel Movement Frequency Risk
        def bowel_frequency_risk(freq):
            if freq < 3:
                return 2  # Too infrequent
            elif freq > 7:
                return 1  # Too frequent
            else:
                return 0  # Normal
                
        self.df['Bowel_Frequency_Risk'] = self.df['Frequency of bowel movements'].apply(bowel_frequency_risk)
        
        # Gastrointestinal Symptom Severity Score
        gi_symptoms = ['Presence of bloating', 'Presence of gas', 'Presence of abdominal pain', 'Difficult digestion']
        
        # Weight symptoms by severity impact on gut health
        symptom_weights = {
            'Presence of bloating': 1.0,
            'Presence of gas': 0.8,
            'Presence of abdominal pain': 1.5,
            'Difficult digestion': 1.2
        }
        
        self.df['GI_Symptom_Severity_Score'] = sum(
            self.df[symptom] * weight for symptom, weight in symptom_weights.items()
        )
        
        # Antibiotic Impact Score (recent antibiotics disrupt microbiome)
        self.df['Antibiotic_Impact_Score'] = self.df['Recent use of antibiotics'].astype(int) * 2
          # Supplement Support Score (probiotics, prebiotics, etc.)
        supplement_cols = ['Probiotics', 'Prebiotics', 'Vitamins', 'Minerals']
        
        # Convert "Other supplements" to boolean (True if not empty/NaN)
        self.df['Other_Supplements_Bool'] = self.df['Other supplements'].notna() & (self.df['Other supplements'] != '')
        
        # Calculate supplement support score
        self.df['Supplement_Support_Score'] = (
            self.df[supplement_cols].sum(axis=1) + 
            self.df['Other_Supplements_Bool'].astype(int)
        )
        
        # Gut Health Risk Composite Score
        self.df['Gut_Health_Risk_Score'] = (
            self.df['Bristol_Scale_Risk'] * 0.25 +
            self.df['Bowel_Frequency_Risk'] * 0.20 +
            self.df['GI_Symptom_Severity_Score'] * 0.30 +
            self.df['Antibiotic_Impact_Score'] * 0.15 +
            (5 - self.df['Supplement_Support_Score']) * 0.10  # Inverted - less supplements = higher risk
        )
        
        # Previous GI Issues Impact
        self.df['Previous_GI_Impact'] = self.df['Previous gastrointestinal issues'].astype(int) * 1.5
        
        print("✅ Created 7 gut health-specific features:")
        print("   - Bristol_Scale_Risk")
        print("   - Bowel_Frequency_Risk") 
        print("   - GI_Symptom_Severity_Score")
        print("   - Antibiotic_Impact_Score")
        print("   - Supplement_Support_Score")
        print("   - Gut_Health_Risk_Score (composite)")
        print("   - Previous_GI_Impact")
        
    def create_lifestyle_interaction_features(self):
        """Create interaction features between lifestyle factors"""
        print("\n4. CREATING LIFESTYLE INTERACTION FEATURES")
        print("-" * 48)
        
        # Stress-Sleep Interaction (both affect gut-brain axis)
        self.df['Stress_Sleep_Interaction'] = (
            self.df['Stress level (1-10 scale)'] * 
            (10 - self.df['Average sleep hours (hours)'])  # Higher when sleep is poor
        )
        
        # Diet-Exercise Synergy Score
        self.df['Diet_Exercise_Synergy'] = (
            self.df['Microbiome_Friendly_Diet_Score'] * 
            np.log1p(self.df['Weekly frequency of physical activity (per week)'])
        )
        
        # Hydration-Fiber Interaction (both important for gut transit)
        self.df['Hydration_Fiber_Interaction'] = (
            self.df['Daily water intake (liters)'] * 
            self.df['Fiber_Rich_Foods_Score']
        )
        
        # Age-Risk Interaction (if age data available)
        # Note: Age not in current dataset, using BMI as proxy for metabolic age
        self.df['Metabolic_Age_Risk'] = self.df['BMI'] * self.df['Gut_Health_Risk_Score']
        
        print("✅ Created 4 lifestyle interaction features:")
        print("   - Stress_Sleep_Interaction")
        print("   - Diet_Exercise_Synergy")
        print("   - Hydration_Fiber_Interaction")
        print("   - Metabolic_Age_Risk")
        
    def create_domain_specific_ratios(self):
        """Create additional domain-specific ratios and indices"""
        print("\n5. CREATING DOMAIN-SPECIFIC RATIOS")
        print("-" * 40)
        
        # Microbiome Diversity Index (higher diversity = better health)
        dietary_diversity = ['Weekly consumption of vegetables (portions)',
                           'Weekly consumption of fruits (portions)',
                           'Weekly consumption of whole grains (portions)',
                           'Weekly consumption of plant proteins (portions)',
                           'Weekly consumption of fermented foods (portions)']
        
        # Count non-zero dietary components (diversity indicator)
        self.df['Dietary_Diversity_Count'] = (self.df[dietary_diversity] > 0).sum(axis=1)
        
        # Inflammation Risk Index (combining multiple factors)
        self.df['Inflammation_Risk_Index'] = (
            (self.df['BMI'] > 25).astype(int) * 0.3 +
            (self.df['Stress level (1-10 scale)'] > 6).astype(int) * 0.3 +
            (self.df['Average sleep hours (hours)'] < 7).astype(int) * 0.2 +
            (self.df['Weekly frequency of physical activity (per week)'] == 0).astype(int) * 0.2
        )
        
        # Gut Barrier Function Score (factors affecting intestinal permeability)
        self.df['Gut_Barrier_Function_Score'] = (
            self.df['Probiotic_Foods_Score'] * 0.4 +
            (self.df['Daily water intake (liters)'] > 2).astype(int) * 0.3 +
            (self.df['Stress level (1-10 scale)'] <= 5).astype(int) * 0.3
        )
        
        # Microbiome Recovery Potential (ability to bounce back)
        self.df['Microbiome_Recovery_Potential'] = (
            self.df['Supplement_Support_Score'] * 0.3 +
            self.df['Diet_Exercise_Synergy'] * 0.4 +
            (5 - self.df['Gut_Health_Risk_Score']) * 0.3  # Inverted risk
        )
        
        print("✅ Created 4 domain-specific ratios:")
        print("   - Dietary_Diversity_Count")
        print("   - Inflammation_Risk_Index")
        print("   - Gut_Barrier_Function_Score")
        print("   - Microbiome_Recovery_Potential")
        
    def analyze_feature_impact(self):
        """Analyze the impact of new features on target classes"""
        print("\n6. ANALYZING FEATURE IMPACT ON TARGET CLASSES")
        print("-" * 52)
        
        # Get all new features created
        new_features = [
            'Plant_to_Animal_Protein_Ratio', 'Fiber_Rich_Foods_Score', 'Probiotic_Foods_Score',
            'Microbiome_Friendly_Diet_Score', 'Whole_to_Processed_Foods_Ratio',
            'Stress_Level_Binned', 'BMI_Detailed_Bins', 'Sleep_Quality_Binned',
            'Physical_Activity_Binned', 'Water_Intake_Binned',
            'Bristol_Scale_Risk', 'Bowel_Frequency_Risk', 'GI_Symptom_Severity_Score',
            'Antibiotic_Impact_Score', 'Supplement_Support_Score', 'Gut_Health_Risk_Score',
            'Previous_GI_Impact', 'Stress_Sleep_Interaction', 'Diet_Exercise_Synergy',
            'Hydration_Fiber_Interaction', 'Metabolic_Age_Risk', 'Dietary_Diversity_Count',
            'Inflammation_Risk_Index', 'Gut_Barrier_Function_Score', 'Microbiome_Recovery_Potential'
        ]
        
        # Analyze numeric features by target class
        numeric_features = [f for f in new_features if self.df[f].dtype in ['int64', 'float64']]
        
        print("\nFeature means by microbiota status:")
        print("-" * 40)
        
        for feature in numeric_features[:10]:  # Show first 10 for brevity
            print(f"\n{feature}:")
            for status in sorted(self.df[self.target_column].unique()):
                mean_val = self.df[self.df[self.target_column] == status][feature].mean()
                std_val = self.df[self.df[self.target_column] == status][feature].std()
                print(f"  {status:12}: {mean_val:6.2f} ± {std_val:5.2f}")
        
        return new_features
        
    def save_enhanced_dataset(self):
        """Save the dataset with all new engineered features"""
        print("\n7. SAVING ENHANCED DATASET")
        print("-" * 32)
        
        output_file = "advanced_feature_engineered_data.csv"
        self.df.to_csv(output_file, index=False)
        
        original_features = 54
        total_features = len(self.df.columns)
        new_features_count = total_features - original_features
        
        print(f"✅ Enhanced dataset saved as '{output_file}'")
        print(f"📊 Total features: {total_features}")
        print(f"🔬 New features added: {new_features_count}")
        print(f"📈 Feature increase: {(new_features_count/original_features)*100:.1f}%")
        
        return output_file
        
    def generate_feature_summary_report(self, new_features):
        """Generate a comprehensive feature engineering report"""
        print("\n8. GENERATING FEATURE ENGINEERING REPORT")
        print("-" * 45)
        
        report_content = f"""# Advanced Feature Engineering Report
## Gut Microbiota Classification Project

### Overview
- **Original Features**: 54
- **Total Features After Engineering**: {len(self.df.columns)}
- **New Features Created**: {len(new_features)}
- **Dataset Size**: {len(self.df):,} samples

### Feature Categories Created

#### 1. Composite Dietary Metrics (5 features)
- **Plant_to_Animal_Protein_Ratio**: Critical for microbiome diversity
- **Fiber_Rich_Foods_Score**: Essential for beneficial bacteria growth
- **Probiotic_Foods_Score**: Direct microbiome support
- **Microbiome_Friendly_Diet_Score**: Overall gut health diet assessment
- **Whole_to_Processed_Foods_Ratio**: Food quality indicator

#### 2. Ordinal Binning (5 features)
- **Stress_Level_Binned**: 5-level stress categorization (gut-brain axis)
- **BMI_Detailed_Bins**: 6-level BMI classification (metabolic health)
- **Sleep_Quality_Binned**: 6-level sleep categorization (circadian impact)
- **Physical_Activity_Binned**: 6-level activity classification (exercise benefits)
- **Water_Intake_Binned**: 4-level hydration assessment (gut transit)

#### 3. Gut Health-Specific Features (7 features)
- **Bristol_Scale_Risk**: Stool consistency risk assessment
- **Bowel_Frequency_Risk**: Movement frequency evaluation
- **GI_Symptom_Severity_Score**: Weighted symptom burden
- **Antibiotic_Impact_Score**: Microbiome disruption indicator
- **Supplement_Support_Score**: Gut health support assessment
- **Gut_Health_Risk_Score**: Composite gut health risk
- **Previous_GI_Impact**: Historical gastrointestinal issues

#### 4. Lifestyle Interaction Features (4 features)
- **Stress_Sleep_Interaction**: Combined stress-sleep impact
- **Diet_Exercise_Synergy**: Nutrition-activity synergy
- **Hydration_Fiber_Interaction**: Gut transit optimization
- **Metabolic_Age_Risk**: Age-related metabolic risk

#### 5. Domain-Specific Ratios (4 features)
- **Dietary_Diversity_Count**: Number of diverse food categories
- **Inflammation_Risk_Index**: Multi-factor inflammation assessment
- **Gut_Barrier_Function_Score**: Intestinal barrier health
- **Microbiome_Recovery_Potential**: Recovery capacity assessment

### Key Insights by Target Class

"""
        
        # Add class-specific insights for key features
        key_features = ['Gut_Health_Risk_Score', 'Microbiome_Friendly_Diet_Score', 
                       'Inflammation_Risk_Index', 'Microbiome_Recovery_Potential']
        
        for feature in key_features:
            report_content += f"\n#### {feature}\n"
            for status in sorted(self.df[self.target_column].unique()):
                mean_val = self.df[self.df[self.target_column] == status][feature].mean()
                report_content += f"- **{status}**: {mean_val:.2f}\n"
        
        report_content += f"""
### Feature Engineering Methodology

1. **Domain Knowledge Integration**: Features designed based on gut microbiome research
2. **Clinical Relevance**: All features have biological significance for gut health
3. **Interaction Modeling**: Captured synergistic effects between lifestyle factors
4. **Risk Stratification**: Multiple risk scores for different aspects of gut health
5. **Composite Scoring**: Combined multiple indicators for robust assessment

### Next Steps
1. Apply these features to transformer-based classification model
2. Analyze feature importance and attention weights
3. Validate clinical relevance of engineered features
4. Use interpretability analysis to understand model decisions

---
*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open("advanced_feature_engineering_report.md", "w", encoding='utf-8') as f:
            f.write(report_content)
            
        print("✅ Feature engineering report saved as 'advanced_feature_engineering_report.md'")
        
    def run_complete_feature_engineering(self):
        """Run the complete advanced feature engineering pipeline"""
        print("\n🚀 RUNNING COMPLETE ADVANCED FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Execute all feature engineering steps
        self.create_composite_dietary_metrics()
        self.create_ordinal_bins()
        self.create_gut_health_features()
        self.create_lifestyle_interaction_features()
        self.create_domain_specific_ratios()
        
        new_features = self.analyze_feature_impact()
        output_file = self.save_enhanced_dataset()
        self.generate_feature_summary_report(new_features)
        
        print(f"\n🎉 ADVANCED FEATURE ENGINEERING COMPLETED!")
        print("="*50)
        print(f"✅ Dataset enhanced: {output_file}")
        print(f"✅ Report generated: advanced_feature_engineering_report.md")
        print(f"📊 Ready for transformer model training!")
        
        return output_file, new_features

# Main execution
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer("health_data_10000_chunk - health_data_10000_chunk.csv")
    
    # Run complete pipeline
    output_file, new_features = engineer.run_complete_feature_engineering()
    
    print(f"\n📋 NEW FEATURES CREATED ({len(new_features)} total):")
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
