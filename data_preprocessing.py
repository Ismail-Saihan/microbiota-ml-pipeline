import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class MicrobiotaDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for microbiota classification.
    Handles missing values, encoding, scaling, and class imbalance.
    """
    
    def __init__(self, data_path, target_column='Current status of microbiota', random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.smote = None
        self.feature_names = None
        self.class_mapping = None
        
        # Load and prepare data
        self.load_data()
        
    def load_data(self):
        """Load and perform initial data preparation."""
        print("🔄 Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Use enhanced dataset if available
        enhanced_path = self.data_path.replace('.csv', '').replace('health_data_10000_chunk - health_data_10000_chunk', 'enhanced_microbiota_data.csv')
        try:
            self.df = pd.read_csv(enhanced_path)
            print(f"✅ Using enhanced dataset: {self.df.shape}")
        except FileNotFoundError:
            print("ℹ️ Using original dataset")
            
    def identify_feature_types(self):
        """Identify different types of features for appropriate preprocessing."""
        print("\n🔍 Identifying feature types...")
        
        # Exclude target and non-predictive columns
        exclude_cols = [
            self.target_column,
            'Residential Address',  # Not predictive
            'Supplement Plan - Recommended products',
            'Supplement Plan - Dosages and timings', 
            'Supplement Plan - Intake tracking',
            'Supplement Plan - Automatic reminders',
            'Meal Plan - Recommended/avoided foods',
            'Weekly menu',
            'Personalized recipes',
            'Intestinal health indicators',  # Likely derived from target
            'Comparison with optimal values'  # Likely derived from target
        ]
        
        # Get feature columns
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Categorize features
        self.numeric_features = []
        self.categorical_features = []
        self.boolean_features = []
        
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                if self.df[col].nunique() == 2:  # Binary numeric (0/1)
                    self.boolean_features.append(col)
                else:
                    self.numeric_features.append(col)
            elif self.df[col].dtype == 'bool':
                self.boolean_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"✅ Numeric features: {len(self.numeric_features)}")
        print(f"✅ Categorical features: {len(self.categorical_features)}")
        print(f"✅ Boolean features: {len(self.boolean_features)}")
        
        return feature_cols
    
    def handle_missing_values(self):
        """Analyze and handle missing values."""
        print("\n🔧 Handling missing values...")
        
        missing_counts = self.df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) == 0:
            print("✅ No missing values found!")
            return
        
        print(f"Found missing values in {len(missing_features)} features:")
        for feature, count in missing_features.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {feature}: {count} ({percentage:.2f}%)")
            
        # Strategy: Remove features with >50% missing, impute others
        high_missing = missing_features[missing_features > len(self.df) * 0.5]
        if len(high_missing) > 0:
            print(f"Removing features with >50% missing: {list(high_missing.index)}")
            self.df = self.df.drop(columns=high_missing.index)
            
        # Update feature lists
        self.identify_feature_types()
    
    def encode_categorical_features(self):
        """Handle categorical feature encoding."""
        print("\n🏷️ Processing categorical features...")
        
        # Analyze categorical features
        categorical_info = {}
        for col in self.categorical_features:
            unique_vals = self.df[col].nunique()
            categorical_info[col] = {
                'unique_count': unique_vals,
                'sample_values': list(self.df[col].unique()[:5])
            }
            print(f"  {col}: {unique_vals} unique values")
            
        # Decide encoding strategy
        self.high_cardinality_features = []
        self.low_cardinality_features = []
        
        for col, info in categorical_info.items():
            if info['unique_count'] > 10:  # High cardinality
                self.high_cardinality_features.append(col)
            else:
                self.low_cardinality_features.append(col)
        
        print(f"✅ High cardinality features (Label Encoding): {len(self.high_cardinality_features)}")
        print(f"✅ Low cardinality features (One-Hot Encoding): {len(self.low_cardinality_features)}")
    
    def create_preprocessing_pipeline(self):
        """Create comprehensive preprocessing pipeline."""
        print("\n🔨 Creating preprocessing pipeline...")
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Boolean pipeline (convert to int and scale)
        boolean_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])
        
        # Low cardinality categorical pipeline (One-Hot Encoding)
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # High cardinality categorical pipeline (Label Encoding)
        # Note: We'll handle this separately since ColumnTransformer doesn't support LabelEncoder directly
        
        # Combine pipelines
        preprocessor_steps = []
        
        if self.numeric_features:
            preprocessor_steps.append(('num', numeric_pipeline, self.numeric_features))
            
        if self.boolean_features:
            preprocessor_steps.append(('bool', boolean_pipeline, self.boolean_features))
            
        if self.low_cardinality_features:
            preprocessor_steps.append(('cat_low', categorical_pipeline, self.low_cardinality_features))
        
        if preprocessor_steps:
            self.preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='drop'  # Drop remaining columns
            )
        
        print("✅ Preprocessing pipeline created")
    
    def prepare_target_variable(self):
        """Encode target variable and analyze class distribution."""
        print("\n🎯 Preparing target variable...")
        
        # Check current class distribution
        class_counts = self.df[self.target_column].value_counts()
        print("Original class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(self.df[self.target_column])
        
        # Create class mapping for interpretability
        self.class_mapping = dict(zip(
            self.label_encoder.classes_,
            range(len(self.label_encoder.classes_))
        ))
        
        print("Class mapping:")
        for class_name, encoded_value in self.class_mapping.items():
            print(f"  {class_name} → {encoded_value}")
        
        return y_encoded
    
    def handle_high_cardinality_features(self, X_df):
        """Handle high cardinality categorical features with Label Encoding."""
        if not self.high_cardinality_features:
            return X_df
        
        print(f"📊 Processing {len(self.high_cardinality_features)} high cardinality features...")
        
        # Store label encoders for each feature
        self.label_encoders = {}
        
        for col in self.high_cardinality_features:
            print(f"  Encoding {col} ({self.df[col].nunique()} unique values)")
            
            # Handle missing values first
            X_df[col] = X_df[col].fillna('Unknown')
            
            # Label encode
            le = LabelEncoder()
            X_df[col + '_encoded'] = le.fit_transform(X_df[col].astype(str))
            self.label_encoders[col] = le
            
            # Drop original column
            X_df = X_df.drop(columns=[col])
        
        return X_df
    
    def split_data(self, test_size=0.2, val_size=0.15):
        """Split data into train, validation, and test sets with stratification."""
        print(f"\n📊 Splitting data (train: {1-test_size-val_size:.0%}, val: {val_size:.0%}, test: {test_size:.0%})...")
        
        # Prepare features and target
        feature_cols = self.identify_feature_types()
        X = self.df[feature_cols].copy()
        y = self.prepare_target_variable()
        
        # Handle high cardinality features before preprocessing
        X = self.handle_high_cardinality_features(X)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"✅ Train set: {X_train.shape[0]} samples")
        print(f"✅ Validation set: {X_val.shape[0]} samples") 
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Show class distribution in each set
        for set_name, y_set in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
            unique, counts = np.unique(y_set, return_counts=True)
            print(f"{set_name} class distribution:")
            for class_idx, count in zip(unique, counts):
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                percentage = (count / len(y_set)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_preprocessing(self, X_train, X_val, X_test):
        """Apply preprocessing pipeline to all datasets."""
        print("\n⚙️ Applying preprocessing pipeline...")
        
        # Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline()
        
        if self.preprocessor is None:
            print("⚠️ No preprocessing pipeline created")
            return X_train, X_val, X_test
        
        # Fit on training data and transform all sets
        print("Fitting preprocessor on training data...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names for interpretability
        self.get_feature_names()
        
        print(f"✅ Processed feature shape: {X_train_processed.shape[1]} features")
        print(f"✅ Train set processed: {X_train_processed.shape}")
        print(f"✅ Validation set processed: {X_val_processed.shape}")
        print(f"✅ Test set processed: {X_test_processed.shape}")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def get_feature_names(self):
        """Extract feature names after preprocessing."""
        try:
            # Get feature names from each transformer
            feature_names = []
            
            # Process each transformer
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend([f"num_{col}" for col in columns])
                elif name == 'bool':
                    feature_names.extend([f"bool_{col}" for col in columns])
                elif name == 'cat_low':
                    # Get feature names from OneHotEncoder
                    if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                        onehot_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                        feature_names.extend(onehot_features)
                    else:
                        # Fallback for older sklearn versions
                        feature_names.extend([f"cat_{col}_{i}" for col in columns for i in range(10)])  # Approximation
            
            # Add high cardinality encoded features
            if hasattr(self, 'label_encoders'):
                for col in self.label_encoders.keys():
                    feature_names.append(f"high_card_{col}_encoded")
            
            self.feature_names = feature_names
            print(f"✅ Feature names extracted: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"⚠️ Could not extract feature names: {e}")
            self.feature_names = [f"feature_{i}" for i in range(self.preprocessor.n_features_in_)]
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using SMOTE or other techniques."""
        print(f"\n⚖️ Handling class imbalance using {method.upper()}...")
        
        # Check current class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("Before balancing:")
        for class_idx, count in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if method == 'smote':
            # Use SMOTE for oversampling
            self.smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            
            try:
                X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
                
                # Check new distribution
                unique, counts = np.unique(y_train_balanced, return_counts=True)
                print("After SMOTE balancing:")
                for class_idx, count in zip(unique, counts):
                    class_name = self.label_encoder.inverse_transform([class_idx])[0]
                    percentage = (count / len(y_train_balanced)) * 100
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
                
                print(f"✅ Balanced dataset: {X_train_balanced.shape[0]} samples (+{X_train_balanced.shape[0] - X_train.shape[0]})")
                
                return X_train_balanced, y_train_balanced
                
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}")
                print("Continuing without balancing...")
                return X_train, y_train
        
        return X_train, y_train
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for handling imbalance in loss function."""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        
        # Create class weight dictionary
        class_weight_dict = dict(zip(classes, class_weights))
        
        print("\n📊 Class weights calculated:")
        for class_idx, weight in class_weight_dict.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"  {class_name}: {weight:.3f}")
        
        return class_weight_dict
    
    def save_preprocessor(self, filepath="microbiota_preprocessor.pkl"):
        """Save the complete preprocessing pipeline."""
        preprocessing_components = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'label_encoders': getattr(self, 'label_encoders', {}),
            'smote': self.smote,
            'feature_names': self.feature_names,
            'class_mapping': self.class_mapping,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'boolean_features': self.boolean_features,
            'high_cardinality_features': getattr(self, 'high_cardinality_features', []),
            'low_cardinality_features': getattr(self, 'low_cardinality_features', [])
        }
        
        joblib.dump(preprocessing_components, filepath)
        print(f"💾 Preprocessing pipeline saved to: {filepath}")
    
    def run_complete_preprocessing(self, apply_smote=True, save_pipeline=True):
        """Run the complete preprocessing pipeline."""
        print("🚀 MICROBIOTA DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Identify feature types
        self.identify_feature_types()
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Analyze categorical features
        self.encode_categorical_features()
        
        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        
        # Step 5: Apply preprocessing
        X_train_processed, X_val_processed, X_test_processed = self.apply_preprocessing(
            X_train, X_val, X_test
        )
        
        # Step 6: Handle class imbalance
        if apply_smote:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train_processed, y_train
            )
        else:
            X_train_balanced, y_train_balanced = X_train_processed, y_train
        
        # Step 7: Calculate class weights (useful for model training)
        class_weights = self.calculate_class_weights(y_train)
        
        # Step 8: Save preprocessing pipeline
        if save_pipeline:
            self.save_preprocessor()
        
        # Summary
        print("\n" + "="*80)
        print("📊 PREPROCESSING SUMMARY")
        print("="*80)
        print(f"✅ Original dataset: {self.df.shape}")
        print(f"✅ Features after preprocessing: {X_train_processed.shape[1]}")
        print(f"✅ Train set: {X_train_balanced.shape}")
        print(f"✅ Validation set: {X_val_processed.shape}")
        print(f"✅ Test set: {X_test_processed.shape}")
        print(f"✅ Class imbalance handled: {'Yes (SMOTE)' if apply_smote else 'No'}")
        print(f"✅ Class weights calculated: Yes")
        print(f"✅ Pipeline saved: {'Yes' if save_pipeline else 'No'}")
        
        return {
            'X_train': X_train_balanced,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': y_train_balanced,
            'y_val': y_val,
            'y_test': y_test,
            'class_weights': class_weights,
            'feature_names': self.feature_names,
            'class_mapping': self.class_mapping
        }

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    data_path = "health_data_10000_chunk - health_data_10000_chunk.csv"
    preprocessor = MicrobiotaDataPreprocessor(data_path)
    
    # Run complete preprocessing
    processed_data = preprocessor.run_complete_preprocessing(
        apply_smote=True,
        save_pipeline=True
    )
    
    # Save processed data
    np.savez('processed_microbiota_data.npz', **processed_data)
    print("\n💾 Processed data saved to: processed_microbiota_data.npz")
    
    print("\n🎯 Data ready for transformer model training!")
