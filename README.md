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
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
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
├── README.md
└── requirements.txt
```

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

---

## 3. Upgrade pip

Before installing the project dependencies, upgrade `pip`:

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

# Reproducing the Full Analysis

To reproduce the analytical workflow from the beginning, run the project notebooks in the following order:

```text
1. notebooks/01_eda.ipynb
2. notebooks/02_model_development.ipynb
3. notebooks/03_final_model.ipynb
```

The notebooks should be run in this order because later stages build on the analysis and modeling decisions made earlier in the workflow.

---

## 5. Run `01_eda.ipynb`

Open:

```text
notebooks/01_eda.ipynb
```

This notebook explores and summarizes the road-accident dataset.

It includes exploratory analysis of the road, environmental, temporal, and historical variables used in the project.

Generated outputs are stored in:

```text
reports/figures/
reports/tables/
```

In VS Code, make sure the notebook is using the project's `venv` Python environment as its kernel.

Then run all cells from top to bottom.

---

## 6. Run `02_model_development.ipynb`

Next, open:

```text
notebooks/02_model_development.ipynb
```

Run all cells from top to bottom.

This notebook develops and evaluates the regression and classification models and evaluates classification probability thresholds.

The project uses:

- **Multiple Linear Regression** for continuous `accident_risk` estimation
- **Logistic Regression** for high-risk classification

The notebook also evaluates model performance and produces supporting figures and tables in:

```text
reports/figures/
reports/tables/
```

---

## 7. Run `03_final_model.ipynb`

Finally, open:

```text
notebooks/03_final_model.ipynb
```

Run all cells from top to bottom.

This notebook implements the selected final workflow and saves the reusable model artifacts used by the Streamlit application.

The final artifacts are stored in:

```text
artifacts/
```

The expected files are:

```text
regression_pipeline.joblib
classification_pipeline.joblib
classification_metadata.joblib
```

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

After the required dependencies are installed and the model artifacts are available in `artifacts/`, start the Streamlit application from the project root:

```bash
streamlit run app/app.py
```

Alternatively:

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

It provides two modes:

- **Single Observation** – enter one road-condition profile and generate risk predictions.
- **Batch CSV Prioritization** – upload multiple observations and rank them by predicted high-risk probability.

---

# Using the Application

## Single Observation

Enter the road, environmental, temporal, and historical characteristics of one observation, then click:

```text
Analyze Risk
```

The application returns:

- **Estimated Accident Risk** – continuous prediction from the regression model.
- **High-Risk Probability** – probability generated by the Logistic Regression model.
- **High-Risk Flag** – whether the probability meets the selected `0.40` classification threshold.
- **Review Priority** – illustrative `LOW`, `MEDIUM`, or `HIGH` review category.

The `0.75` cutoff used for the **HIGH** review-priority category is an illustrative business rule rather than a transportation-engineering threshold.

---

## Batch CSV Prioritization

Use the included sample file to test the batch workflow:

```text
data/batch_test_sample.csv
```

The uploaded CSV should contain these model features:

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

The highest-risk observations appear first in the prioritized results.

You can download the complete scored dataset using:

```text
Download Prioritized Results
```

The downloaded file is:

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

### `streamlit` is not recognized

Make sure the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app/app.py
```

### PowerShell blocks virtual environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

### Batch CSV cannot be scored

Check that:

- All required columns are present.
- Required fields do not contain missing values.
- Categorical values match the accepted values exactly.
- Boolean columns use `0`, `1`, `True`, or `False`.

You can compare your file with:

```text
data/batch_test_sample.csv
```

### Model artifact error

Confirm that these files exist:

```text
artifacts/regression_pipeline.joblib
artifacts/classification_pipeline.joblib
artifacts/classification_metadata.joblib
```

The Streamlit application requires these saved artifacts.

If they are missing, run:

```text
notebooks/03_final_model.ipynb
```

to regenerate them.

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