# Road Accident Risk Prediction and Safety Prioritization

## Project Overview

This project develops a proof-of-concept road-safety screening and prioritization workflow using synthetic road accident data.

The goal is to estimate accident risk from roadway, environmental, temporal, and historical characteristics and identify observations that may deserve higher priority for additional transportation-safety review.

The project includes:

- Exploratory Data Analysis
- Multiple Linear Regression for continuous accident-risk estimation
- Logistic Regression for high-risk classification
- Model evaluation and visualization
- Saved reusable model artifacts
- An interactive Streamlit safety-prioritization application

The models are intended to support screening and prioritization rather than automate transportation-engineering decisions.

---

## Research Question

> Can road, environmental, time-related, and historical characteristics be used to estimate accident risk and identify high-risk road conditions for transportation safety prioritization?

---

## Dataset

The project uses the Kaggle **Predicting Road Accident Risk** dataset from Playground Series S5E10.

The training dataset contains 517,754 observations and the target variable `accident_risk`.

The models use 12 predictors:

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

The `id` column is excluded from modeling.

---

## Project Structure

```text
road-accident-risk-prediction/
│
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
│   └── P1.pdf
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/
│   └── data_split.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Modeling Approach

The project uses two complementary models.

### Multiple Linear Regression

Used to estimate continuous `accident_risk`.

Validation results:

```text
RMSE = 0.0735
MAE  = 0.0583
R²   = 0.8042
```

### Logistic Regression

A high-risk observation is defined using the 75th percentile of the training-set target:

```text
accident_risk >= 0.46
```

Validation results:

```text
Accuracy  = 0.8961
Precision = 0.8202
Recall    = 0.7483
F1        = 0.7826
```

The regression model provides a continuous risk estimate, while the classification model provides a high-risk probability and flag for prioritization.

---

## Saved Model Artifacts

The fitted preprocessing and models are stored as reusable `joblib` artifacts:

```text
artifacts/
├── regression_pipeline.joblib
├── classification_pipeline.joblib
└── classification_metadata.joblib
```

Reload testing confirmed that the saved artifacts reproduce the original model outputs.

---

## Safety Prioritization Web Application

The Streamlit application is located at:

```text
app/app.py
```

It provides two modes.

### Single Observation

Users enter one road-condition observation and receive:

- Estimated accident risk
- High-risk probability
- High-risk flag
- Illustrative review priority

### Batch CSV Prioritization

Users can upload multiple road-condition observations.

The application scores and ranks each row using predicted high-risk probability and generates:

- `estimated_accident_risk`
- `high_risk_probability`
- `high_risk_flag`
- `review_priority`
- `priority_rank`

The prioritized results can also be downloaded as a CSV file.

The HIGH, MEDIUM, and LOW priority categories are illustrative decision-support rules and are not validated transportation-engineering thresholds.

---

## Limitations

Key limitations include:

- The dataset is synthetic.
- The data do not contain geographic coordinates or road-segment identifiers.
- Model relationships should not be interpreted as causal.
- Some actual high-risk observations remain unflagged.
- The web application is a proof-of-concept screening tool rather than a production transportation-safety system.

---

# Installation and Environment Setup

## 1. Clone the Repository

```powershell
git clone https://github.com/hngo13-cell/road-accident-risk-prediction.git
cd road-accident-risk-prediction
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

Create the virtual environment:

```powershell
python -m venv venv
```

If PowerShell prevents activation because script execution is disabled, allow scripts for the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, the terminal should display:

```text
(venv)
```

at the beginning of the command prompt.

`-Scope Process` applies the execution-policy change only to the current PowerShell session.

### macOS / Linux

Create the environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Core dependencies include:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
notebook
streamlit
```

---

# Running the Analysis

## Run the EDA Notebook

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

## Run the Modeling Notebook

```powershell
jupyter notebook notebooks/02_modeling.ipynb
```

The modeling notebook performs the following major steps:

1. Loads the train-validation split.
2. Performs preprocessing.
3. Builds a regression baseline.
4. Fits Multiple Linear Regression.
5. Creates the binary high-risk target.
6. Builds a majority-class classification baseline.
7. Fits Logistic Regression.
8. Evaluates both models.
9. Produces evaluation visualizations.
10. Saves and reloads model artifacts.
11. Documents the data-to-action workflow.
12. Documents the Streamlit safety-prioritization application.

---

## Execute the Notebooks from the Command Line

The EDA notebook can be executed automatically using:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

The modeling notebook can be executed using:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb
```

This executes the notebook cells and saves the generated outputs back into the notebook.

---

# Running the Streamlit Application

Make sure the virtual environment is activated.

From the project root, run:

```powershell
streamlit run app/app.py
```

Streamlit will start a local server and display the application address in the terminal.

The application can then be opened in a web browser.

---

# Batch Application Test

A small sample file is included for testing the batch prioritization workflow:

```text
data/batch_test_sample.csv
```

Open the Streamlit application and select:

```text
Batch CSV Prioritization
```

Upload:

```text
data/batch_test_sample.csv
```

Expected behavior:

- CSV validation passes.
- 20 observations are scored.
- Each observation receives:
  - estimated accident risk
  - high-risk probability
  - high-risk flag
  - review priority
  - priority rank
- Observations are ranked from highest to lowest predicted high-risk probability.
- Prioritized results can be downloaded as a CSV file.

---

# Expected Batch CSV Schema

A batch CSV must contain the following model features:

```text
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

Categorical values should use the same values used by the training dataset:

```text
road_type:
highway, rural, urban

lighting:
daylight, dim, night

weather:
clear, foggy, rainy

time_of_day:
afternoon, evening, morning
```

Boolean variables should use `0/1` or `True/False`.

Additional columns may be present in an uploaded CSV, but all required model features must be included.

---

# Model Artifacts and Reuse

The fitted pipelines are saved in:

```text
artifacts/
```

The regression artifact can be loaded with:

```python
import joblib

regression_model = joblib.load(
    "artifacts/regression_pipeline.joblib"
)
```

The classification artifact can be loaded with:

```python
classification_model = joblib.load(
    "artifacts/classification_pipeline.joblib"
)
```

Classification metadata can be loaded with:

```python
metadata = joblib.load(
    "artifacts/classification_metadata.joblib"
)
```

Because preprocessing is stored inside each fitted pipeline, raw predictor data can be passed directly to the model as long as the required columns and valid category values are provided.

---

# Reproducibility

The project supports reproducibility through:

- Fixed train-validation random state
- Training-only preprocessing
- Training-only calculation of the high-risk target threshold
- Saved preprocessing/model pipelines
- Saved classification metadata
- Dependency tracking through `requirements.txt`
- Executable Jupyter notebooks
- Reload testing for saved model artifacts
- Streamlit scoring using saved artifacts rather than model retraining

A peer should be able to clone the repository, create the environment, install dependencies, execute the notebooks, reload the saved artifacts, and run the Streamlit application without changing the modeling workflow.

---

# Key Results

## Multiple Linear Regression

```text
Validation RMSE = 0.0735
Validation MAE  = 0.0583
Validation R²   = 0.8042
```

## Logistic Regression

```text
Validation Accuracy  = 0.8961
Validation Precision = 0.8202
Validation Recall    = 0.7483
Validation F1        = 0.7826
```

The results indicate that the available predictors contain useful information for estimating accident risk and identifying higher-risk observations within this synthetic dataset.

---

# Future Work

Future development could include:

- Real local transportation and crash data
- Geographic coordinates and road-segment identifiers
- GIS mapping
- Alternative classification thresholds
- Improved high-risk recall
- Additional validated roadway characteristics
- Model monitoring
- Professional review outcomes
- Feedback loops for model validation and retraining

A production implementation would require additional validation using real transportation data and review by transportation-safety professionals.

---

# Intended Use

This project is a **proof-of-concept screening and prioritization tool**.

It is intended to demonstrate how predictive analytics could help transportation professionals identify observations for additional investigation.

It is not intended to:

- Make automated engineering decisions
- Recommend specific roadway interventions
- Replace traffic engineers or transportation-safety professionals
- Serve as a validated production safety system

---

# Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

# Course

**MSBA 265 – Special Analytics Topics**

University of the Pacific

---

# Project Author

**Anh Ngo**

---

# AI Use Disclosure

AI tools were used to support project organization, conceptual clarification, troubleshooting, documentation, and visual-design planning.

All reported analytical results are based on the project dataset and code executed by the student.

AI-generated suggestions were reviewed and validated before being incorporated into the project. No analytical results or dataset observations were fabricated.

Final modeling, interpretation, and project decisions remain the responsibility of the student.