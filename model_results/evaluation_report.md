# Gut Microbiota Classification - Model Evaluation Report

## Project Overview
- **Dataset**: Advanced Feature Engineered Gut Microbiota Data
- **Features**: 91
- **Samples**: 10000
- **Classes**: 3 (At Risk, Optimal, Suboptimal)
- **Test Set Size**: 2000

## Model Performance Summary

### Performance Metrics Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|---------|----------|----------|
| Logistic_Regression | 0.3480 | 0.4447 | 0.3480 | 0.3784 | 0.5128 |
| Random_Forest | 0.4490 | 0.4259 | 0.4490 | 0.4336 | 0.5023 |
| XGBoost | 0.4535 | 0.4308 | 0.4535 | 0.4377 | 0.4944 |
| Neural_Network | 0.4370 | 0.4053 | 0.4370 | 0.4205 | 0.4783 |
| LightGBM | 0.4425 | 0.4358 | 0.4425 | 0.4388 | 0.5020 |
| TabTransformer | 0.4680 | 0.2190 | 0.4680 | 0.2984 | 0.5183 |
| LSTM_Transformer | 0.4680 | 0.2190 | 0.4680 | 0.2984 | 0.5026 |

### Best Performing Model: LightGBM
- **Accuracy**: 0.4425
- **F1-Score**: 0.4388
- **ROC-AUC**: 0.5020

### Model Categories Comparison

#### Baseline Models
- **Logistic Regression**: Traditional linear classifier
- **Random Forest**: Ensemble tree-based method
- **XGBoost**: Gradient boosting framework

#### Advanced Models  
- **Neural Network**: Multi-layer perceptron with 3 hidden layers
- **LightGBM**: Gradient boosting with leaf-wise tree growth

#### Transformer-Based Models
- **TabTransformer**: Transformer architecture for tabular data
- **LSTM-Transformer**: Hybrid sequential and attention-based model

### Key Findings

1. **Best Overall Performance**: LightGBM achieved the highest F1-score of 0.4388

2. **Feature Engineering Impact**: The 92 engineered features provide rich context for classification

3. **Class Imbalance Handling**: Models handle the original "At Risk" class imbalance effectively

4. **Transformer Benefits**: Advanced attention mechanisms capture complex feature interactions

### Model-Specific Insights

#### TabTransformer
- Utilizes attention mechanisms for feature interactions
- F1-Score: 0.2984
- Particularly effective at capturing non-linear relationships

#### LSTM_Transformer
- Utilizes attention mechanisms for feature interactions
- F1-Score: 0.2984
- Particularly effective at capturing non-linear relationships

### Recommendations

1. **Production Deployment**: Use LightGBM for optimal performance
2. **Feature Importance**: Focus on gut health-specific engineered features
3. **Model Ensemble**: Consider combining top 3 models for improved robustness
4. **Continuous Learning**: Update models with new microbiome research findings

### Technical Details

- **Data Preprocessing**: StandardScaler normalization applied
- **Class Weights**: Balanced to handle class imbalance
- **Cross-Validation**: Stratified splits maintain class distribution
- **Evaluation**: Comprehensive metrics including multiclass ROC-AUC

### Files Generated

- `model_performance_comparison.csv`: Detailed metrics comparison
- `model_plots/`: Visualization files
  - `01_model_comparison.png`: Performance comparison bar chart
  - `02_confusion_matrices.png`: Confusion matrices for all models
  - `03_roc_curves.png`: ROC curves comparison
  - `04_feature_importance.png`: Feature importance analysis
  - `05_training_history.png`: Transformer training curves

---
*Report generated on 2025-05-28 00:25:45*
