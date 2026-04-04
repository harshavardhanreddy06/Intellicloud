# Random Forest Energy Efficiency Classifier

The Random Forest (RF) model is the primary classification engine in the IntelliCloud project, responsible for predicting the energy efficiency class of an incoming task.

## 1. Energy Efficiency Class Definition
The RF model is trained to classify tasks into 5 classes based on their energy efficiency:
- **Class 1 (0-20%)**: Lowest Efficiency (High Resource consumption, poor power usage).
- **Class 2 (20-40%)**: Low Efficiency.
- **Class 3 (40-60%)**: Medium Efficiency.
- **Class 4 (60-80%)**: High Efficiency.
- **Class 5 (80-100%)**: Highest Efficiency (Optimized Resource consumption, high power efficiency).

## 2. Feature Vector (12 Features)
The RF model accepts a 12-dimensional feature vector:
1. `input_size_mb`
2. `cpu_usage_cores_absolute`
3. `memory_usage_mb`
4. `execution_time_normalized`
5. `instruction_count`
6. `network_io_mb`
7. `power_consumption_watts`
8. `task_size_category` (Ordinal Encoded)
9. `latent_f1` (Latent feature 1 from Autoencoder)
10. `latent_f2` (Latent feature 2 from Autoencoder)
11. `latent_f3` (Latent feature 3 from Autoencoder)
12. `latent_f4` (Latent feature 4 from Autoencoder)

## 3. Composite Scoring Methodology
During the data preparation phase (`core/random_forest_energy.py`), the ground truth `energy_efficiency_class` is calculated using a **weighted average composite score**:
- **CPU Normalized (30%)**: Normalized between [0, 1] across the entire dataset.
- **Memory Normalized (20%)**: Normalized between [0, 1] across the entire dataset.
- **Power Normalized (50%)**: Normalized between [0, 1] across the entire dataset.

**Efficiency Score = 1 - Composite Score**
**Efficiency Class = Binned Efficiency Percentage (1..5)**

## 4. Model Training
The RF is trained on `dataset/task_profiles_full_12_features.json` (~10k records):
- **Classifier Type**: Scikit-learn `RandomForestClassifier`.
- **Training Parameters**: 100 n_estimators, random_state=42.
- **Validation**: 5-fold cross-validation is used to ensure stability and accuracy.
- **Scaling**: All 12 features are normalized using `MinMaxScaler` prior to training.

## 5. Summary of Model Files
- `models/random_forest_energy_efficiency.pkl`: Scaled and trained RF classifier.
- `models/rf_scaler.pkl`: The MinMaxScaler used for task inputs during RF inference.

## 6. How it's Used in the Pipeline
1. The 12-feature vector is passed to the classifier.
2. The classifier returns the predicted `energy_efficiency_class`.
3. This class is then passed to the **DQN Agent** as part of its state vector for scheduling decisions.
4. The predicted class is also used to generate the **SHAP Explanation** for the user dashboard.
