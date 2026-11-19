# Road Accident Risk Prediction and Safety Prioritization

A proof-of-concept analytics project that uses road, environmental, temporal, and historical conditions to estimate accident risk and prioritize observations for additional transportation-safety review.

The project combines:

- **Multiple Linear Regression** for continuous accident-risk estimation
- **Logistic Regression** for high-risk screening
- A **Streamlit application** for single-observation analysis and batch CSV prioritization

> **Note:** This project was developed using a synthetic dataset. The application is intended for screening and decision-support purposes only and should not replace professional transportation-safety or engineering analysis.

---

## Repository Structure

```text
road-accident-risk-prediction/
├── app/
│   └── app.py
│
├── artifacts/
│   ├── regression_pipeline.joblib
│   ├── classification_pipeline.joblib
│   └── classification_metadata.joblib
│
├── data/
│   ├── batch_test_sample.csv
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── docs/
│   └── P1.pdf
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

The dataset files required to reproduce the analysis are included in the `data/` directory. No additional dataset download is required.

---

# Getting Started

## 1. Clone the Repository

Open PowerShell, Command Prompt, Terminal, or the VS Code terminal and run:

```bash
git clone https://github.com/hngo13-cell/road-accident-risk-prediction.git
cd road-accident-risk-prediction
```

You should now be inside the project root directory.

---

## 2. Create a Virtual Environment

### Windows

Create a virtual environment:

```powershell
python -m venv venv
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, the terminal should show the virtual environment name, for example:

```text
(venv)
```

> **Important:** Make sure the `venv` environment is activated before installing dependencies or running any notebook.

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

You can verify the installed version with:

```bash
python -m pip --version
```

---

## 4. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset files required to reproduce the full analysis are already included in the repository under:

```text
data/
├── batch_test_sample.csv
├── train.csv
├── test.csv
└── sample_submission.csv
```

The main files are:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

No additional Kaggle download or API authentication is required.

`batch_test_sample.csv` is included as a small example file for testing the Streamlit application's batch-prioritization workflow.

---

# VS Code Environment Setup

If you open the notebooks in VS Code, make sure they use the project's virtual environment.

In the top-right corner of the notebook, select:

```text
Select Kernel
→ Python Environments
→ venv
```

The selected Python environment should point to the `venv` folder inside this project.

You can also verify the active Python environment from the terminal:

```bash
python -c "import sys; print(sys.executable)"
```

The returned path should include:

```text
road-accident-risk-prediction\venv\
```

---

# Reproducing the Full Analysis

Run the notebooks in the following order:

```text
1. notebooks/01_eda.ipynb
2. notebooks/02_model_development.ipynb
3. notebooks/03_final_model.ipynb
```

The notebooks should be executed in this order because later stages build on the analysis and modeling decisions made earlier in the workflow.

> Before running the commands below, confirm that `(venv)` appears in the terminal and that the dataset files are present in the `data/` directory.

---

## 5. Run `01_eda.ipynb`

From the project root, run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

This notebook explores and summarizes the road-accident dataset.

Generated outputs are stored in:

```text
reports/figures/
reports/tables/
```

---

## 6. Run `02_model_development.ipynb`

After the EDA notebook finishes successfully, run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_development.ipynb
```

This notebook develops and evaluates the regression and classification models and evaluates classification probability thresholds.

The project uses:

- **Multiple Linear Regression** for continuous `accident_risk` estimation
- **Logistic Regression** for high-risk classification

Supporting figures and tables are saved in:

```text
reports/figures/
reports/tables/
```

---

## 7. Run `03_final_model.ipynb`

After model development is complete, run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_final_model.ipynb
```

This notebook implements the selected final workflow and saves the reusable model artifacts used by the Streamlit application.

The expected files are:

```text
artifacts/regression_pipeline.joblib
artifacts/classification_pipeline.joblib
artifacts/classification_metadata.joblib
```

---

## Run All Notebooks

Alternatively, after activating the `venv`, run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_development.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_final_model.ipynb
```

Each command should complete successfully before the next notebook is executed.

---

# Saved Model Artifacts

```text
regression_pipeline.joblib
```

Contains the preprocessing workflow and fitted Multiple Linear Regression model.

```text
classification_pipeline.joblib
```

Contains the preprocessing workflow and fitted Logistic Regression model.

```text
classification_metadata.joblib
```

Stores information required by the final classification workflow, including the training-derived high-risk definition and selected classification probability threshold.

The Streamlit application loads these saved artifacts directly.

---

# Running the Application

Because the fitted model artifacts are included in the repository, the Streamlit application can be run without retraining the models.

From the project root:

```bash
streamlit run app/app.py
```

Alternatively:

```bash
python -m streamlit run app/app.py
```

The application provides two modes:

- **Single Observation** – enter one road-condition profile and generate risk predictions.
- **Batch CSV Prioritization** – upload multiple observations and rank them by predicted high-risk probability.

> The full dataset is included in the repository for reproducibility, while the Streamlit application itself loads the saved model artifacts directly.

---

# Using the Application

## Single Observation

Enter the road, environmental, temporal, and historical characteristics of one observation and click **Analyze Risk**.

The application returns:

- **Estimated Accident Risk**
- **High-Risk Probability**
- **High-Risk Flag**
- **Review Priority**

The high-risk classification threshold is `0.40`. The `0.75` cutoff for the **HIGH** review-priority category is an illustrative business rule rather than a transportation-engineering threshold.

---

## Batch CSV Prioritization

A sample upload file is included at:

```text
data/batch_test_sample.csv
```

The uploaded CSV should contain:

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

After upload, the application validates the data, scores all observations, and ranks them from highest to lowest predicted high-risk probability.

The output includes:

```text
priority_rank
estimated_accident_risk
high_risk_probability
high_risk_flag
review_priority
```

The complete prioritized results can be downloaded as:

```text
road_safety_prioritized_results.csv
```

---

# How to Interpret the Results

The application is designed to support **screening and prioritization**, not automatic decision-making.

A recommended workflow is:

```text
Road-condition data
        ↓
Model scoring
        ↓
High-risk probability
        ↓
Priority ranking
        ↓
Professional transportation-safety review
        ↓
Engineering investigation or decision
```

In batch analysis, observations with higher predicted high-risk probability should generally be reviewed first.

The regression estimate can provide additional risk context, while the classification probability serves as the primary prioritization signal.

---

# Troubleshooting

### Dataset files not found

Confirm that the repository contains:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

If any of these files are missing, make sure the repository was cloned completely and that the files were not removed locally.

Do not rename the files because the notebooks expect the original filenames and paths.

---

### Notebook uses the wrong Python environment

Make sure the virtual environment is activated:

```powershell
.\venv\Scripts\Activate.ps1
```

Then verify:

```bash
python -c "import sys; print(sys.executable)"
```

If running the notebook interactively in VS Code, select:

```text
Select Kernel
→ Python Environments
→ venv
```

---

### `jupyter` is not recognized

Make sure the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

Then use:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

---

### `streamlit` is not recognized

Make sure the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run app/app.py
```

---

### PowerShell blocks virtual environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

### Batch CSV cannot be scored

Check that:

- All required columns are present.
- Required fields do not contain missing values.
- Categorical values match the accepted values.
- Boolean columns use `0`, `1`, `True`, or `False`.

You can compare the input file with:

```text
data/batch_test_sample.csv
```

---

### Model artifact error

Confirm that these files exist:

```text
artifacts/regression_pipeline.joblib
artifacts/classification_pipeline.joblib
artifacts/classification_metadata.joblib
```

If they are missing, rerun:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_final_model.ipynb
```

The required dataset files are already included in the `data/` directory.

---

# Project Scope

This project demonstrates a reproducible road-safety risk-screening workflow using synthetic data.

It is appropriate for:

- Risk screening
- Prioritization
- Analytical demonstration
- Decision-support prototyping

It should **not** be used to:

- Automatically determine transportation-safety actions
- Prescribe engineering countermeasures
- Replace professional engineering or safety analysis

Real-world deployment would require local roadway and crash data, additional validation, and professional review.

---

## Author

**Anh Ngo**  
MSBA 265 – Special Analytics Topics  
University of the Pacific