# Lab 1 Report  
**Data-Centric ML Pipeline, Leakage Control, and Deployment Readiness**  
CSCI-4170 – Spring 2026

---

## 1. Setup

We implemented a reproducible, leakage-safe machine learning pipeline in Python using Jupyter Notebook.

### Environment

The setup uses:
- pandas for data loading and cleaning
- NumPy for numerical operations
- scikit-learn for modeling and preprocessing
- matplotlib for basic visualization
- Fixed random seeds to ensure reproducibility

All notebooks run top-to-bottom without manual edits.

---

### Data Loading and Initial Cleaning

We loaded the dataset directly using pandas and performed initial inspection to:

- View dataset shape and column names
- Check data types
- Identify missing values
- Check for duplicate rows
- Examine the target variable distribution

This ensured that we understood class balance and potential quality issues before modeling.

---

### Leakage Risk Identification

During setup, we explicitly searched for plausible leakage vectors, including:

- ID-like columns that uniquely identify rows
- Post-outcome variables
- Timestamp-related features
- Near-duplicate rows

Any columns identified as potential leakage sources were removed prior to model training.

---

### Train/Test Split

We created a single leakage-safe split:

- Stratified split (for classification) to preserve class distribution
- Fixed random seed for reproducibility
- Training data used exclusively to fit preprocessing and models
- Test data held out for final evaluation

This aligns with the lab requirement of disciplined evaluation.

---

### Preprocessing Pipeline

We constructed a scikit-learn `Pipeline` to ensure preprocessing is applied correctly:

- Feature scaling (e.g., StandardScaler) applied only to numeric features
- Optional ColumnTransformer if mixed feature types were present
- All transformations fit only on the training set

By embedding preprocessing inside the pipeline, we ensured:

- No data leakage
- Reproducible transformations
- Fair comparison between baseline and SVM models

---

### Experiment Logging

We maintained an Experiment Log CSV file containing:

- Model type
- Hyperparameters
- Random seed
- Evaluation metric
- Timestamp

This allowed controlled comparison across runs and ensured reproducibility.

---

Overall, the setup phase focused on disciplined data handling, explicit leakage prevention, reproducibility, and structured experimentation.


---

## 2. Baseline Model — k-Nearest Neighbors (kNN)

As our required non-tree baseline, we implemented a **k-Nearest Neighbors (kNN) classifier** using `KNeighborsClassifier` within a leakage-safe scikit-learn pipeline.

### Features Used

The model was trained using the following predictors:

age, gender, bmi, smoking_status, alcohol_consumption, exercise_level,  
diet_type, sun_exposure, income_level, latitude_region,  
vitamin_a_percent_rda, vitamin_c_percent_rda, vitamin_d_percent_rda,  
vitamin_e_percent_rda, vitamin_b12_percent_rda, folate_percent_rda,  
calcium_percent_rda, iron_percent_rda, hemoglobin_g_dl,  
serum_vitamin_d_ng_ml, serum_vitamin_b12_pg_ml, serum_folate_ng_ml,  
symptoms_count, symptoms_list,  
has_night_blindness, has_fatigue, has_bleeding_gums, has_bone_pain,  
has_muscle_weakness, has_numbness_tingling, has_memory_problems,  
has_pale_skin  

Target: **has_multiple_deficiencies**

Numeric features were scaled using `StandardScaler` inside the pipeline to ensure distance-based learning was not distorted by feature magnitude differences.

---

### Model Configuration

- Model: `KNeighborsClassifier`
- Distance metric: Euclidean (default)
- Preprocessing embedded inside pipeline
- Train/test split performed before fitting
- Fixed random seed for reproducibility

Because kNN is distance-based, scaling was essential to prevent features with larger numeric ranges (e.g., vitamin levels) from dominating similarity calculations.

---

### Test Performance (with 95% Confidence Intervals)

Accuracy: **0.9188**  
95% CI: [0.9000, 0.9363]

F1 Score: **0.9336**  
95% CI: [0.9170, 0.9485]

Confidence intervals were estimated using bootstrap resampling of the test set, providing uncertainty bounds around performance.

---

### Interpretation

- The baseline achieved strong overall classification performance.
- The high F1 score indicates good balance between precision and recall.
- Confidence intervals are relatively tight, suggesting stable generalization.
- Performance reflects strong predictive signal in clinical and symptom-based features.
- However, kNN relies on local neighborhood structure and may struggle in higher-dimensional settings.

This baseline established a strong reference point for comparison with the SVM model in Week 2.

---

## 3. SVM Results

We evaluated three SVM variants using the same train/test split (seed=42), identical preprocessing pipeline, and consistent evaluation metrics. Hyperparameters for each model were selected using search procedures, and the best estimators were evaluated on the held-out test set.

To quantify uncertainty, we computed 95% confidence intervals using bootstrap resampling (1,000 iterations) on the test predictions.

---

### Linear SVM

The Linear SVM learns a maximum-margin linear decision boundary.

**Test Performance (95% CI via bootstrap):**

Accuracy: 0.9329  
95% CI: [0.9162, 0.9487]

F1 Score: 0.9453  
95% CI: [0.9314, 0.9591]

The linear model improved upon the kNN baseline, indicating that the dataset contains strong linearly separable structure. Confidence intervals are relatively tight, suggesting stable performance.

---

### RBF SVM

The RBF (Radial Basis Function) kernel allows nonlinear decision boundaries by mapping features into a higher-dimensional space.

**Test Performance (95% CI via bootstrap):**

Accuracy: 0.9462  
95% CI: [0.9300, 0.9613]

F1 Score: 0.9566  
95% CI: [0.9427, 0.9692]

The RBF SVM achieved the strongest performance across all evaluated models. The improvement over the linear SVM suggests the presence of nonlinear interactions among clinical, nutritional, and symptom features.

This model produced the highest F1 score and the tightest confidence interval, indicating both improved predictive capacity and stable generalization.

---

### Sigmoid SVM

The Sigmoid kernel approximates neural-network-like decision boundaries but is less commonly optimal in practice.

**Test Performance (95% CI via bootstrap):**

Accuracy: 0.9297  
95% CI: [0.9125, 0.9463]

F1 Score: 0.9426  
95% CI: [0.9277, 0.9568]

The sigmoid kernel performed slightly worse than the linear and RBF variants. While still strong, it did not capture structure as effectively as the RBF kernel.

---

### Comparison and Interpretation

Across all three variants, SVM models outperformed the Week 1 kNN baseline. The RBF SVM provided the best overall performance:

- Highest accuracy
- Highest F1 score
- Strong, stable confidence intervals

Because all preprocessing, splits, and evaluation procedures were held constant, performance gains can be attributed to increased model capacity rather than experimental variation.

The results suggest that the classification task benefits from nonlinear decision boundaries, but even a linear margin performs competitively due to strong predictive signal in the feature set.=

---

## 4. Calibration and Threshold Decision

### Probability Calibration (Platt Scaling)

Because SVM decision scores are not inherently calibrated probabilities, we applied probability calibration using **CalibratedClassifierCV** with sigmoid scaling (Platt scaling).

We selected the best-performing model from Week 2 — the **RBF SVM** — and calibrated it as follows:

- Split the original training data into:
  - Sub-training set (75%)
  - Validation set (25%), stratified
- Used `cv="prefit"` to calibrate the already tuned RBF SVM
- Applied sigmoid scaling to map decision scores to calibrated probabilities

This ensured threshold selection was performed on meaningful probability estimates rather than raw margin scores.

---

### Cost-Based Threshold Selection

We defined a simple 2×2 cost matrix reflecting higher penalty for false negatives:

- False Positive (FP) cost = 1  
- False Negative (FN) cost = 5  

This reflects a health screening context where missing an unhealthy individual is significantly more costly than incorrectly flagging a healthy one.

We evaluated thresholds from 0.10 to 0.90 and selected the threshold that minimized expected cost on the validation set.

**Chosen threshold:** 0.15

---

### Test Set Results — Default vs Optimized Threshold

#### Default Threshold (0.5)

Accuracy: 0.9475  
F1 Score: 0.9578  

Confusion matrix:
- 21 false positives  
- 21 false negatives  

At the default threshold, the model achieves strong overall performance with balanced error types.

---

#### Cost-Optimized Threshold (0.15)

Accuracy: 0.9387  
F1 Score: 0.9523  

Confusion matrix:
- 40 false positives  
- 9 false negatives  

Lowering the threshold substantially reduces false negatives (21 → 9), improving detection of unhealthy individuals. As expected, false positives increase (21 → 40), and overall accuracy decreases slightly.

---

### Decision Justification

The optimized threshold aligns the model with domain-specific risk priorities:

- It prioritizes recall for the “Not Healthy” class.
- It significantly reduces costly missed detections.
- It maintains strong overall F1 performance despite a modest drop in accuracy.

In a health-related screening context, the cost of failing to identify an unhealthy individual outweighs the inconvenience of additional follow-up evaluations. Therefore, the threshold of 0.15 provides a more appropriate operating point under the defined cost assumptions.

This section demonstrates explicit alignment between model outputs and decision policy, moving beyond pure accuracy optimization toward deployment-aware modeling.


---

## 5. NVIDIA DLI Lab Progress Summary

For Week 3, we completed the NVIDIA DLI lab on **IQR-based outlier detection** using the Kaggle House Prices dataset.

### IQR Implementation

We manually implemented the Interquartile Range (IQR) method on the `LotArea` feature:

- Computed Q1 (25th percentile) and Q3 (75th percentile)
- Calculated IQR = Q3 − Q1
- Defined outlier thresholds:
  - Lower bound = Q1 − 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR
- Removed observations outside these bounds

**Results:**

- Original observations: 1460  
- Filtered observations: 1391  
- Outliers removed: 69  

---

### Visual Comparison

Side-by-side boxplots show:
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/09620f4d-3059-4bad-8f0c-5a24404cc8c6" />

- The original `LotArea` distribution contained extreme high-value outliers.
- After IQR filtering, the distribution became more compact.
- Extreme values above 100,000 square feet were removed.
- The median and interquartile range became more representative of typical properties.

This confirms that the dataset originally had heavy right-tail skew driven by a small number of very large lots.

---

### Impact on Model Performance

To evaluate the practical effect of outlier removal, we trained a simple linear regression model predicting `SalePrice` from `LotArea`.

#### Raw Data Model

R²: 0.0627  
MSE: 7,189,094,014.83

#### Filtered Data Model (IQR Applied)

R²: 0.1865  
MSE: 4,725,084,841.92  

---

### Interpretation

Removing outliers significantly improved model performance:

- R² increased from 0.0627 → 0.1865  
- MSE decreased substantially  

This indicates that extreme `LotArea` values were disproportionately influencing the regression line and degrading predictive performance.

The experiment demonstrates a key principle from the NVIDIA lab:

> Preprocessing decisions, particularly outlier handling, can materially affect downstream model behavior.

---

### What We Learned

- IQR provides a simple and robust statistical method for detecting extreme values.
- Outliers can distort linear models due to their influence on the least-squares objective.
- Visual validation (boxplots) strengthens preprocessing decisions.
- Even a single-feature regression can show measurable improvement after proper cleaning.

---

## 6. Limitations

While the pipeline is reproducible and leakage-safe, several limitations remain across the full lab.

### Data Limitations

- Potential residual leakage may still exist despite manual auditing.
- The dataset may contain measurement noise or self-reported bias (e.g., symptom-based variables).
- Class imbalance may inflate certain metrics (e.g., accuracy).
- External validity is unknown — performance may not generalize to different populations or collection environments.

---

### Modeling Limitations

- Only light hyperparameter tuning was performed.
- Evaluation relied on a single train/test split rather than nested cross-validation.
- The SVM models, particularly RBF, may overfit subtle nonlinear structure.
- Feature engineering was minimal; no domain-informed transformations were applied.
- Interpretability is limited, especially for kernel-based SVM models.

---

### Calibration and Thresholding Limitations

- Calibration used Platt scaling (sigmoid), which assumes a specific parametric form.
- Threshold selection relied on a simplified cost matrix (FP = 1, FN = 5).
- Real-world cost structures are often more complex and dynamic.
- The selected threshold may not remain optimal under distribution shift.

---

### NVIDIA Lab Limitations

- Outlier removal was performed on a single feature (`LotArea`) only.
- IQR assumes symmetric distribution around quartiles and may not capture all anomalous structure.
- The regression evaluation used only one predictor, limiting model complexity.
- More advanced anomaly detection methods (e.g., Isolation Forest, robust regression) were not explored.

---

### Reproducibility and Deployment Considerations

- The pipeline is reproducible under fixed seeds but has not been stress-tested under distributional shift.
- No fairness or subgroup performance analysis was conducted.
- The system has not been evaluated in a real deployment scenario.

---

Overall, while the lab demonstrates disciplined ML practice — including leakage control, calibration, threshold optimization, and preprocessing validation, further work would be required before deployment in a real-world clinical or financial setting.


## Summary

Over three weeks, we developed a reproducible, leakage-safe ML pipeline including:

- Data documentation and audit  
- Baseline comparison  
- SVM with tuning  
- Calibration and cost-based thresholding  
- NVIDIA lab preprocessing work  

