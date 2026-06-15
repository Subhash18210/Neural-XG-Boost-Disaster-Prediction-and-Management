# Neural-XG-Boost-Disaster-Prediction-and-Management
# Neural-XGBoost Disaster Prediction Pipeline

An end-to-end Machine Learning pipeline combining a Deep Neural Network feature extractor with an optimized XGBoost classifier to predict multi-class natural hazards under heavy class-imbalanced conditions.

## Pipeline Architecture
1. **Data Preprocessing**: Handling structural data drops using median/mode imputation and Z-score scaling.
2. **Resampling**: SMOTE interpolation to eliminate majority class structural biases.
3. **Latent Feature Extraction**: 3-layer deep Multilayer Perceptron optimization.
4. **Ensemble Classification**: Regularized gradient-boosted tree execution.
