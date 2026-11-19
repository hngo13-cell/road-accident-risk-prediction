# Road Accident Risk Prediction and Safety Prioritization

A proof-of-concept analytics project that uses roadway, environmental, temporal, and historical conditions to estimate accident risk and prioritize higher-risk observations for transportation-safety review.

The project includes:

- **Multiple Linear Regression** for continuous accident-risk estimation
- **Logistic Regression** for high-risk screening
- A **Streamlit application** for single-observation screening and batch CSV prioritization

> **Note:** This project uses a synthetic Kaggle dataset and is intended for analytical demonstration and decision support only.

---

## Repository Structure

```text id="tv7ze4"
road-accident-risk-prediction/
├── app/
│   └── app.py
│
├── artifacts/
│   ├── classification_metadata.joblib
│   ├── classification_pipeline.joblib
│   └── regression_pipeline.joblib
│
├── data/
│   ├── batch_test_sample.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── docs/
│   ├── P1.pdf
│   ├── P2.pdf
│   └── Road_Accident_Risk_Presentation.pptx
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_final_model.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── src/
│   └── data_split.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The repository includes the dataset, fitted model artifacts, analysis notebooks, generated outputs, and project documentation.

No additional dataset download is required.

---

# Getting Started

## 1. Clone the Repository

```bash id="s1ce11"
git clone https://github.com/hngo13-cell/road-accident-risk-prediction.git
cd road-accident-risk-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```powershell id="0t2kja"
python -m venv venv
```

If PowerShell blocks script execution:

```powershell id="02ehtv"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell id="g1u01e"
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash id="e3j0qh"
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash id="mtiq6q"
python -m pip install --upgrade pip
```

Install project dependencies:

```bash id="2subwo"
pip install -r requirements.txt
```

---

# Dataset

The required dataset files are included in:

```text id="g2a5m2"
data/
├── batch_test_sample.csv
├── sample_submission.csv
├── test.csv
└── train.csv
```

The main analysis uses:

```text id="if1n2p"
data/train.csv
data/test.csv
```

`batch_test_sample.csv` is provided for testing the Streamlit batch-prioritization workflow.

---

# VS Code Setup

When using VS Code notebooks, select the project's virtual environment:

```text id="pp55zu"
Select Kernel
→ Python Environments
→ venv
```

Verify the active Python environment with:

```bash id="4ngz5p"
python -c "import sys; print(sys.executable)"
```

---

# Reproducing the Analysis

Run the notebooks in this order:

```text id="rg4hfr"
1. notebooks/01_eda.ipynb
2. notebooks/02_model_development.ipynb
3. notebooks/03_final_model.ipynb
```

From the project root:

```bash id="bdv3zl"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_development.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_final_model.ipynb
```

### Notebook Purpose

**`01_eda.ipynb`**

- Audits the dataset
- Reviews predictor distributions and relationships
- Checks data quality
- Establishes the train-validation split

**`02_model_development.ipynb`**

- Builds regression and classification baselines
- Fits Multiple Linear Regression
- Fits Logistic Regression
- Evaluates model performance
- Evaluates classification thresholds

**`03_final_model.ipynb`**

- Implements the selected final workflow
- Saves fitted model artifacts
- Reloads and verifies the saved artifacts

---

# Generated Outputs

Model outputs are stored in:

```text id="tfv20l"
reports/
├── figures/
└── tables/
```

The repository already includes these outputs for inspection.

Running the notebooks again may regenerate or overwrite them.

---

# Saved Model Artifacts

The final fitted artifacts are stored in:

```text id="sxrfri"
artifacts/
├── regression_pipeline.joblib
├── classification_pipeline.joblib
└── classification_metadata.joblib
```

`classification_metadata.joblib` stores:

```text id="tj1g0j"
High-risk target cutoff: 0.46
Classification probability threshold: 0.40
```

The saved artifacts were reloaded and verified to reproduce the original model predictions.

---

# Running the Streamlit Application

The saved artifacts are already included, so the application can run without retraining.

From the project root:

```bash id="dpyx9a"
streamlit run app/app.py
```

Alternatively:

```bash id="pqts7v"
python -m streamlit run app/app.py
```

The application supports:

- **Single Observation**
- **Batch CSV Prioritization**

---

# Batch CSV Input

A sample batch file is included at:

```text id="77qeay"
data/batch_test_sample.csv
```

Required columns:

```text id="dmbf4m"
road_type
num_lanes
curvature
speed_limit
lighting
weather
road_signs_present
public_road
time_of_day
holiday
school_season
num_reported_accidents
```

The batch output includes:

```text id="6zd8ar"
priority_rank
estimated_accident_risk
high_risk_probability
high_risk_flag
review_priority
```

---

# Review Priority

The Streamlit application uses the following illustrative review-priority categories:

```text id="rxem26"
HIGH      probability >= 0.75
MEDIUM    0.40 <= probability < 0.75
LOW       probability < 0.40
```

The `0.40` cutoff is the selected classification threshold.

The `0.75` HIGH-priority cutoff is an illustrative business rule and is not a validated transportation-engineering threshold.

---

# Project Documents

Project deliverables are stored in:

```text id="5nu2xq"
docs/
```

Including:

```text id="gxjs8j"
P1.pdf
P2.pdf
Road_Accident_Risk_Presentation.pptx
```

---

# Troubleshooting

### `jupyter` is not recognized

Make sure the virtual environment is activated and run:

```bash id="df55xk"
pip install -r requirements.txt
```

### `streamlit` is not recognized

Run:

```bash id="a5m1vc"
python -m streamlit run app/app.py
```

### Model artifact error

Confirm that these files exist:

```text id="jxtd4e"
artifacts/regression_pipeline.joblib
artifacts/classification_pipeline.joblib
artifacts/classification_metadata.joblib
```

If they are missing, rerun:

```bash id="k3r9vv"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_final_model.ipynb
```

---

## Author

**Anh Ngo**  
MSBA 265 – Special Analytics Topics  
University of the Pacific