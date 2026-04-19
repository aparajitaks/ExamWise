# 📘 ExamWise  
### AI-Driven Educational Analytics & Assessment Design Assistant

ExamWise is a Machine Learning–powered web application that analyzes programming questions, predicts their difficulty level, and extends into an **Agentic AI assistant** that reasons about assessment quality and generates structured recommendations for improving exam design.

---

## 🚀 Features

### Milestone 1 — ML-Based Exam Question Analytics
- Upload Questions, Answers, and Tags datasets
- Automated data reduction and preprocessing
- Feature engineering and difficulty scoring
- ML-based difficulty prediction (Logistic Regression + TF-IDF)
- Model performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Student performance patterns per question
- Interactive visualizations (difficulty distribution, accuracy charts)

### Milestone 2 — Agentic AI Assessment Design Assistant
- AI-driven assessment quality analysis via Google Gemini (free tier)
- Structured report generation with 6 required sections:
  - Assessment quality summary
  - Question difficulty distribution
  - Identified learning gaps
  - Recommended assessment improvements
  - Supporting pedagogical references
  - Educational and ethical disclaimers
- Synthetic RAG with built-in pedagogical best practices
- Agentic workflow with explicit state management
- Graceful handling of incomplete or noisy data
- **Extension:** Automated Question Generation for weak topics

---

## 🏗️ Project Structure

```bash
ExamWise
│
├── app.py                        # Streamlit application entry point
│
├── scripts/                      # Data processing & pipeline logic
│   ├── data_reduction.py         # Dataset filtering and reduction
│   ├── feature_engineering.py    # Feature creation & difficulty scoring
│   └── assessment_agent.py       # Agentic AI assistant (Milestone 2)
│
├── models/                       # Trained ML artifacts
│   ├── model.joblib              # Logistic Regression classifier
│   ├── vectorizer.joblib         # TF-IDF vectorizer
│   └── metrics.txt               # Classification performance metrics
│
├── data/                         # Data directories (ignored in Git)
│   ├── raw/                      # Uploaded datasets
│   ├── reduced/                  # Reduced datasets
│   └── processed/                # Processed datasets
│
├── outputs/                      # Generated visualizations & analysis
│
├── docs/                         # Project documentation
│   ├── problem_statement.md
│   ├── input_output_specification.md
│   ├── system_architecture.md
│   ├── MODEL_CARD.md
│   ├── app_use.md
│   └── dataSet_usage.md
│
├── notebooks/                    # Jupyter notebooks
│   └── model_training.ipynb      # Model training and EDA
│
├── report/                       # LaTeX report and figures
│
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment variable template
└── README.md
```

---

## ⚙️ Tech Stack

- **Python 3.x**
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models and evaluation
- **Matplotlib** — Visualizations
- **Streamlit** — Web UI
- **Joblib** — Model serialization
- **Google Generative AI** — Gemini API (free tier) for agentic assistant
- **python-dotenv** — Environment variable management

---

## 📊 How It Works

### Milestone 1 Pipeline
1. **Upload:** User uploads `Questions.csv`, `Answers.csv`, `Tags.csv`
2. **Preprocessing:** Data reduction → Answer aggregation → Feature engineering → Difficulty scoring
3. **Prediction:** TF-IDF vectorization → Logistic Regression → Easy/Medium/Hard classification
4. **Display:** True vs Predicted difficulty table, confusion matrix, performance metrics, charts

### Milestone 2 Agent
5. **Analysis:** Agent reads processed data, detects quality issues, aggregates statistics
6. **RAG Retrieval:** Agent retrieves pedagogical guidelines from built-in knowledge base
7. **Report:** Gemini generates structured 6-section assessment design report
8. **Extension:** Automated question generation for identified weak topics

---

## 🖥️ Local Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ynakoo/ExamWise.git
cd ExamWise
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API Key (for Milestone 2)

```bash
cp .env.example .env
# Edit .env and add your free-tier Gemini API key
```

Or paste the key directly in the Streamlit sidebar at runtime.

### 5️⃣ Run Application

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📂 Required Input Files

The application expects three CSV files (from the [StackOverflow Kaggle dataset](https://www.kaggle.com/datasets/stackoverflow/stackoverflow)):

- `Questions.csv`
- `Answers.csv`
- `Tags.csv`

These files are uploaded via the UI and are not stored in the repository.

---

## 📈 Model Output

The system generates:
- Predicted difficulty levels (Easy, Medium, Hard)
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion matrix
- Difficulty distribution plots
- Hardest question analysis
- AI-generated assessment design reports

---

## 🧠 Difficulty Labels

| Label  | Score Range | Description |
|--------|------------|-------------|
| Easy   | < 0.33     | Lower engagement complexity |
| Medium | 0.33–0.66  | Moderate difficulty |
| Hard   | ≥ 0.66     | High difficulty, sparse quality answers |

---

## 🔒 Git Hygiene

The following are ignored via `.gitignore`:
- `data/raw/`, `data/reduced/`, `data/processed/`
- `venv/`, `__pycache__/`, `.env`

Only source code and essential artifacts are version controlled.

---

## 🎯 Future Improvements

- Real-time model retraining
- Advanced NLP models (BERT-based classification)
- Question similarity clustering
- User performance analytics
- Multi-exam comparative analysis
- PDF export of assessment reports