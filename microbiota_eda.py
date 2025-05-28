import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MicrobiotaEDA:
    def __init__(self, data_path):
        """Initialize the EDA class and load the dataset."""
        self.data_path = data_path
        self.df = None
        self.target_column = 'Current status of microbiota'
        self.load_data()
        
    def load_data(self):
        """Load and perform initial data inspection."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        
    def basic_info(self):
        """Display basic information about the dataset."""
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
            
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        
        print(f"\nFirst few rows:")
        print(self.df.head())
        
    def missing_values_analysis(self):
        """Analyze missing values in the dataset."""
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        
        print("Missing values summary:")
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        missing_cols = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_cols) > 0:
            plt.subplot(1, 2, 1)
            missing_cols['Missing Count'].plot(kind='bar')
            plt.title('Missing Values Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.subplot(1, 2, 2)
            missing_cols['Missing Percentage'].plot(kind='bar')
            plt.title('Missing Values Percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found!', 
                    ha='center', va='center', fontsize=16)
            
        plt.suptitle('Missing Values Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
    def target_variable_analysis(self):
        """Analyze the target variable distribution."""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target_counts = self.df[self.target_column].value_counts()
        target_props = self.df[self.target_column].value_counts(normalize=True) * 100
        
        print("Target variable distribution:")
        for category, count in target_counts.items():
            percentage = target_props[category]
            print(f"{category}: {count} ({percentage:.2f}%)")
        
        # Visualize target distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
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
        plt.show()
        
        # Check for class imbalance
        print(f"\nClass Imbalance Analysis:")
        min_class = target_counts.min()
        max_class = target_counts.max()
        imbalance_ratio = max_class / min_class
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print("⚠️  Significant class imbalance detected! Consider balancing techniques.")
        else:
            print("✅ Classes are relatively balanced.")
            
    def demographic_analysis(self):
        """Analyze demographic features (Height, Weight, BMI)."""
        print("\n" + "="*60)
        print("DEMOGRAPHIC ANALYSIS")
        print("="*60)
        
        demographic_cols = ['Height (cm)', 'Weight (kg)', 'BMI']
        
        # Basic statistics
        print("Demographic statistics:")
        print(self.df[demographic_cols].describe())
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, col in enumerate(demographic_cols):
            # Distribution plot
            axes[0, i].hist(self.df[col], bins=30, alpha=0.7, color=f'C{i}')
            axes[0, i].set_title(f'{col} Distribution')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('Frequency')
            
            # Box plot by target
            sns.boxplot(data=self.df, x=self.target_column, y=col, ax=axes[1, i])
            axes[1, i].set_title(f'{col} by Microbiota Status')
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Outlier detection using IQR method
        print("\nOutlier Detection (IQR method):")
        for col in demographic_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
    
    def lifestyle_analysis(self):
        """Analyze lifestyle factors."""
        print("\n" + "="*60)
        print("LIFESTYLE ANALYSIS")
        print("="*60)
        
        lifestyle_cols = [
            'Physical activity type',
            'Weekly frequency of physical activity (per week)',
            'Average sleep hours (hours)',
            'Stress level (1-10 scale)',
            'Smoking status (Yes/No, quantity)',
            'Alcohol consumption'
        ]
        
        numeric_lifestyle = [
            'Weekly frequency of physical activity (per week)',
            'Average sleep hours (hours)',
            'Stress level (1-10 scale)'
        ]
        
        # Numeric lifestyle factors
        print("Numeric lifestyle statistics:")
        print(self.df[numeric_lifestyle].describe())
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_lifestyle):
            # Distribution by target
            for status in self.df[self.target_column].unique():
                subset = self.df[self.df[self.target_column] == status]
                axes[i].hist(subset[col], alpha=0.6, label=status, bins=20)
            axes[i].set_title(f'{col} by Microbiota Status')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        # Categorical lifestyle factors
        categorical_lifestyle = ['Physical activity type', 'Smoking status (Yes/No, quantity)', 'Alcohol consumption']
        
        for i, col in enumerate(categorical_lifestyle, 3):
            if i < len(axes):
                # Count plot
                value_counts = self.df[col].value_counts()
                if len(value_counts) <= 10:  # Only plot if not too many categories
                    value_counts.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, f'Too many categories\n({len(value_counts)})', 
                               ha='center', va='center')
                    axes[i].set_title(f'{col} - {len(value_counts)} categories')
        
        plt.tight_layout()
        plt.show()
        
    def dietary_analysis(self):
        """Analyze dietary consumption patterns."""
        print("\n" + "="*60)
        print("DIETARY ANALYSIS")
        print("="*60)
        
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
        
        print("Dietary consumption statistics:")
        print(self.df[dietary_cols].describe())
        
        # Correlation matrix for dietary factors
        dietary_corr = self.df[dietary_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(dietary_corr, dtype=bool))
        sns.heatmap(dietary_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Dietary Factors Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Dietary patterns by microbiota status
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(dietary_cols):
            sns.boxplot(data=self.df, x=self.target_column, y=col, ax=axes[i])
            axes[i].set_title(col.replace('Weekly consumption of ', '').replace('Daily ', ''))
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def gastrointestinal_analysis(self):
        """Analyze gastrointestinal health indicators."""
        print("\n" + "="*60)
        print("GASTROINTESTINAL HEALTH ANALYSIS")
        print("="*60)
        
        gi_cols = [
            'Frequency of bowel movements',
            'Stool consistency (Bristol scale)',
            'Presence of bloating',
            'Presence of gas',
            'Presence of abdominal pain',
            'Difficult digestion'
        ]
        
        numeric_gi = ['Frequency of bowel movements', 'Stool consistency (Bristol scale)']
        boolean_gi = ['Presence of bloating', 'Presence of gas', 'Presence of abdominal pain', 'Difficult digestion']
        
        # Numeric GI factors
        print("Numeric GI statistics:")
        print(self.df[numeric_gi].describe())
        
        # Boolean GI factors
        print("\nBoolean GI factors distribution:")
        for col in boolean_gi:
            counts = self.df[col].value_counts()
            print(f"{col}:")
            for value, count in counts.items():
                print(f"  {value}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Numeric GI by target
        for i, col in enumerate(numeric_gi):
            sns.boxplot(data=self.df, x=self.target_column, y=col, ax=axes[i])
            axes[i].set_title(f'{col} by Microbiota Status')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Boolean GI by target
        for i, col in enumerate(boolean_gi, 2):
            if i < len(axes):
                # Create contingency table
                ct = pd.crosstab(self.df[col], self.df[self.target_column], normalize='columns') * 100
                ct.plot(kind='bar', ax=axes[i], stacked=True)
                axes[i].set_title(f'{col} by Microbiota Status (%)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def correlation_analysis(self):
        """Perform comprehensive correlation analysis."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns that might not be useful for correlation
        exclude_cols = ['Meal times']  # Add any other columns to exclude
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"Analyzing correlations for {len(numeric_cols)} numeric features")
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Find high correlations
        print("\nHigh correlations (|r| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"{col1} ↔ {col2}: {corr_val:.3f}")
            
    def feature_importance_analysis(self):
        """Analyze feature importance using mutual information."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Prepare data for feature importance
        # Select numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it's numeric
        if self.target_column in numeric_features:
            numeric_features.remove(self.target_column)
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.df[self.target_column])
        
        # Calculate mutual information
        X_numeric = self.df[numeric_features].fillna(self.df[numeric_features].median())
        mi_scores = mutual_info_classif(X_numeric, y_encoded, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        
        print("Top 20 most important features (Mutual Information):")
        print(feature_importance.head(20))
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['Mutual_Information'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mutual Information Score')
        plt.title('Top 20 Feature Importance (Mutual Information)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance
        
    def run_complete_eda(self):
        """Run the complete EDA pipeline."""
        print("🔬 MICROBIOTA STATUS CLASSIFICATION - EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        self.basic_info()
        self.missing_values_analysis()
        self.target_variable_analysis()
        self.demographic_analysis()
        self.lifestyle_analysis()
        self.dietary_analysis()
        self.gastrointestinal_analysis()
        self.correlation_analysis()
        feature_importance = self.feature_importance_analysis()
        
        print("\n" + "="*80)
        print("📊 EDA SUMMARY")
        print("="*80)
        print(f"✅ Dataset analyzed: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        print(f"✅ Target variable: {self.target_column}")
        print(f"✅ Classes: {', '.join(self.df[self.target_column].unique())}")
        print(f"✅ Missing values: {self.df.isnull().sum().sum()} total")
        print(f"✅ Feature importance analysis completed")
        print("\n🎯 Ready for model development!")
        
        return feature_importance

# Main execution
if __name__ == "__main__":
    # Initialize EDA
    data_path = r"c:\Users\Saihan\SM tech task\health_data_10000_chunk - health_data_10000_chunk.csv"
    eda = MicrobiotaEDA(data_path)
    
    # Run complete EDA
    feature_importance = eda.run_complete_eda()
    
    # Save feature importance for later use
    feature_importance.to_csv(r"c:\Users\Saihan\SM tech task\feature_importance.csv", index=False)
    print(f"\n💾 Feature importance saved to: feature_importance.csv")
