# Road Accident Risk Prediction and Safety Prioritization

## Project Overview

This project explores whether roadway, environmental, time-related, and historical accident characteristics can be used to estimate accident risk and identify higher-risk road-condition profiles for transportation safety prioritization.

The project is designed as a **proof-of-concept decision-support workflow** for transportation safety planners, traffic engineers, and road-safety analysts. Its purpose is to help prioritize cases for additional safety or engineering review, not to replace professional judgment or prescribe roadway interventions.

**Course:** MSBA 265 – Special Analytics Topics  
**Project Stage:** Final Project – Part 1

---

## Business Problem

Transportation agencies must make safety decisions across many roadway conditions while working with limited staff, time, and funding.

A practical risk-screening workflow could help agencies identify road-condition profiles that appear more risky and therefore deserve closer review before resources are committed.

### Intended Stakeholders

- Transportation safety planners
- Traffic engineers
- Road-safety analysts
- City or state transportation agencies
- Road-safety consulting organizations

### Decision Supported

**Which road-condition observations should receive higher priority for additional safety or engineering review?**

The project is intended to support screening and prioritization. It does not replace professional engineering judgment, field investigation, or formal roadway-safety analysis.

---

## Project Question

> Can road, environmental, time-related, and historical characteristics be used to estimate accident risk and identify high-risk road conditions for transportation safety prioritization?

---

## Dataset

The project uses Kaggle's **Predicting Road Accident Risk** dataset from the 2025 Playground Series.

### Dataset Summary

| Item | Value |
|---|---:|
| Training set | 517,754 rows × 14 columns |
| Kaggle test set | 172,585 rows × 13 columns |
| Target | `accident_risk` |
| Target type | Continuous numeric value from 0 to 1 |
| Missing values | None identified |
| Duplicate rows | None identified |

Each row represents one synthetic road-condition observation containing roadway, environmental, temporal, and historical accident characteristics.

### Key Predictors

- `road_type`
- `num_lanes`
- `curvature`
- `speed_limit`
- `lighting`
- `weather`
- `road_signs_present`
- `public_road`
- `time_of_day`
- `holiday`
- `school_season`
- `num_reported_accidents`

The `id` field is treated as an observation identifier and is excluded from model predictors.

### Dataset Source

Kaggle – Predicting Road Accident Risk  
https://www.kaggle.com/competitions/playground-series-s5e10

---

## Planned Modeling Approach

The project uses two complementary supervised-learning models.

### 1. Multiple Linear Regression

Purpose:

- Estimate the continuous `accident_risk` score.
- Provide an interpretable regression model for understanding how predictors are associated with estimated accident risk.

Baseline:

- Training-set mean accident risk.

Evaluation metrics:

- RMSE
- MAE
- R²

### 2. Logistic Regression

Purpose:

- Identify whether an observation belongs to a higher-risk group.
- Provide a classification-oriented output that can support screening and prioritization.

The proposed high-risk cutoff is the **75th percentile of `accident_risk` calculated from the training data only**.

Baseline:

- Majority-class prediction.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Recall will receive particular attention because a false negative represents a high-risk observation that the screening process failed to flag.

---

## Initial Exploratory Data Analysis

Initial EDA has been completed in:

```text
notebooks/01_eda.ipynb
```

### Data Quality Findings

- No missing values were identified.
- No duplicate rows were identified.
- Training and Kaggle test datasets contain compatible predictor columns.
- Training and test datasets contain compatible category values.
- The `id` column is excluded from modeling.
- An approximately 80/20 training-validation split was created for model development.

### Numeric Relationships

| Predictor | Correlation with `accident_risk` |
|---|---:|
| `curvature` | ≈ 0.544 |
| `speed_limit` | ≈ 0.431 |
| `num_reported_accidents` | ≈ 0.214 |
| `num_lanes` | ≈ -0.006 |

`curvature` shows the strongest positive linear relationship with accident risk among the numeric predictors examined, followed by `speed_limit`.

### Categorical Findings

- Nighttime observations have the highest average accident risk among lighting categories.
- Foggy conditions have the highest average accident risk among weather categories.
- Clear weather has the lowest average risk among the weather groups examined.
- These exploratory relationships do not establish causation.

---

## Proposed Decision-Support Workflow

```text
Road & Environmental Conditions
              ↓
        Preprocessing
              ↓
Multiple Linear Regression + Logistic Regression
              ↓
Risk Estimate + High-Risk Flag / Probability
              ↓
Safety Review Prioritization
              ↓
Transportation Planner / Engineer
              ↓
Further Investigation
```

If model performance is adequate, the final project will translate model outputs into a simple **Road Accident Risk Screening Tool**.

Potential outputs include:

- Estimated accident risk
- High-risk probability
- Priority level
- Suggested action such as additional transportation-safety review

---

## Repository Structure

```text
road-accident-risk-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── docs/
│   └── p1.pdf
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   └── 01_data_split.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The local virtual environment folder should not be committed to the repository.

Recommended `.gitignore` entries:

```text
venv/
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd road-accident-risk-prediction
```

Replace `<repository-url>` with the actual GitHub repository URL.

---

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the EDA Notebook

The exploratory data analysis notebook is located at:

```text
notebooks/01_eda.ipynb
```

### Option 1: Run in VS Code

1. Open the repository folder in VS Code.
2. Make sure the **Python** and **Jupyter** extensions are installed.
3. Open:

```text
notebooks/01_eda.ipynb
```

4. Click **Select Kernel** in the upper-right corner of the notebook.
5. Select the Python interpreter from the project's `venv`.
6. Choose **Run All** or run the notebook cells from top to bottom.

### Option 2: Launch Jupyter from the Terminal

From the repository root:

```bash
python -m jupyter notebook notebooks/01_eda.ipynb
```

Then run all cells from top to bottom.

### Expected EDA Results

A successful notebook run should reproduce:

- Dataset shape and data-type summaries
- Missing-value and duplicate checks
- Target distribution
- Numeric predictor correlations
- Lighting and weather comparisons
- Train/test compatibility checks
- Preliminary preprocessing decisions
- Training-validation split information

The notebook should complete without errors.

---

## Running the Data Split Script

The data split script is located at:

```text
src/01_data_split.py
```

From the repository root, run:

```bash
python src/01_data_split.py
```

The script prepares the training and validation data used for model development.

The Kaggle test dataset is kept separate from model evaluation because its target values are not provided.

---

## Current Project Status

### Completed

- Project framing and stakeholder definition
- Real-world decision-support use case
- Dataset audit
- Missing-value checks
- Duplicate checks
- Initial exploratory data analysis
- Numeric correlation analysis
- Lighting and weather comparisons
- Train/test compatibility checks
- Training-validation split
- Initial preprocessing decisions
- P1 project report

### Next Steps

1. Build the preprocessing pipeline.
2. Establish the regression baseline.
3. Train and evaluate Multiple Linear Regression.
4. Create the high-risk classification target using training data only.
5. Establish the classification baseline.
6. Train and evaluate Logistic Regression.
7. Compare regression and classification performance.
8. Interpret model coefficients and error patterns.
9. Save and reload fitted preprocessing/model artifacts.
10. Develop the Road Accident Risk Screening Tool if model performance is adequate.
11. Complete independent reproducibility testing.

---

## Limitations

This project should be interpreted as a **proof of concept**.

- The Kaggle dataset is synthetic rather than data collected directly from a transportation agency.
- The dataset does not include geographic coordinates or exact roadway locations.
- The project cannot identify a specific real-world intersection or roadway segment.
- Exploratory relationships do not imply causation.
- Model predictions should support further investigation rather than replace professional engineering judgment.
- A real-world deployment would require validation and retraining using local transportation-agency data.

---

## P1 Report

The P1 submission is located at:

```text
docs/p1.pdf
```

The report includes:

- Topic examples
- Supervised machine learning infographic
- Business problem and stakeholder
- Project question and success measures
- Dataset and planned modeling approach
- Initial exploratory data analysis
- Feasibility and next steps
- Proposed Road Accident Risk Screening Tool
- References

---

## References

Federal Highway Administration. *Data-Driven Safety Analysis (DDSA).*  
https://highways.dot.gov/safety/data-analysis-tools/rsdp/data-driven-safety-analysis-ddsa

Kaggle. *Predicting Road Accident Risk: Playground Series – Season 5, Episode 10.*  
https://www.kaggle.com/competitions/playground-series-s5e10

Chen, J., Meng, X., Chen, Z., Yuan, C., Zhao, Y., Zhang, J., et al. (2026). *Machine learning for traffic crash injury severity modelling: A systematic review integrating text mining analysis.* European Transport Research Review, 18, Article 96.  
https://doi.org/10.1186/s12544-026-00856-8

---

## Author

**Anh Ngo**  
University of the Pacific  
MSBA 265 – Special Analytics Topics
