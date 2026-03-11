# ExamWise  
### Question Difficulty Analysis System

ExamWise is a Machine Learning–powered web application that analyzes programming questions and predicts their difficulty level based on textual features and answer statistics.

The system processes raw question data, performs feature engineering, trains a classification model, and provides an interactive Streamlit interface for analysis and visualization.

---

## Features

- Upload Questions, Answers, and Tags datasets
- Automated data reduction and preprocessing
- Feature engineering and difficulty scoring
- ML-based difficulty prediction
- Model performance metrics display
- Interactive visualizations
- Clean Streamlit UI

---

## Project Structure

```bash
ExamWise
│
├── app.py                        # Streamlit application entry point
│
├── scripts/                      # Data processing & pipeline logic
│   ├── data_reduction.py         # Dataset filtering and reduction
│   └── feature_engineering.py    # Feature creation & difficulty scoring
│
├── models/                       # Trained ML artifacts
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.txt
│
├── data/                         # Data directories (ignored in Git)
│   ├── raw/                      # Uploaded datasets
│   ├── reduced/                  # Reduced datasets
│   └── processed/                # Processed datasets
│
├── outputs/                      # Generated visualizations & analysis
│
├── docs/                         # Project documentation
│
├── requirements.txt              # Project dependencies
└── README.md


---

## Tech Stack

- Python 3.x
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

---

## How It Works

1. User uploads:
   - `Questions.csv`
   - `Answers.csv`
   - `Tags.csv`

2. Pipeline performs:
   - Data reduction
   - Answer aggregation
   - Difficulty score computation
   - Difficulty categorization (easy / medium / hard)

3. NLP model:
   - Vectorizes question text using TF-IDF
   - Predicts difficulty class using Logistic Regression

4. UI displays:
   - True vs Predicted difficulty
   - Performance metrics
   - Distribution visualizations
   - Hardest questions

---

## Local Setup

### 1️ Clone Repository

```bash
git clone https://github.com/ynakoo/ExamWise.git
cd ExamWise
2️ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
3️ Install Dependencies
pip install -r requirements.txt
4️ Run Application
streamlit run app.py

Then open:

http://localhost:8501
 Required Input Files

The application expects three CSV files:

Questions.csv

Answers.csv

Tags.csv

These files are uploaded via the UI and are not stored in the repository.

 Model Output

The system generates:

Predicted difficulty levels

Classification metrics

Difficulty distribution plots

Hardest question analysis

 Difficulty Labels

The model categorizes questions into:

Easy

Medium

Hard

Based on engagement metrics and textual complexity.

 Git Hygiene

The following directories are ignored:

data/raw/
data/reduced/
data/processed/
venv/
__pycache__/

Only source code and essential artifacts are version controlled.

 Future Improvements

Real-time model retraining

Advanced NLP models (BERT-based classification)

Question similarity clustering

User performance analytics

Cloud deployment (Streamlit Cloud / Render)
