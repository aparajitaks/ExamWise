# System Architecture — ExamWise

The system follows a modular ML/NLP and Agentic AI pipeline designed for exam question
analytics, difficulty prediction, and AI-assisted assessment design.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Streamlit)                  │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ CSV Upload  │  │ ML Pipeline  │  │ Agent Report│  │ Question  │ │
│  │ (Q, A, T)  │  │   Trigger    │  │  Generator  │  │ Generator │ │
│  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘  └─────┬─────┘ │
└────────┼────────────────┼────────────────┼──────────────┼──────────┘
         │                │                │              │
         ▼                ▼                ▼              ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────────────────────┐
│  Data Layer │  │  ML Pipeline   │  │  Agentic AI Layer (M2)      │
│             │  │                │  │                             │
│ data/raw/   │  │ 1. Reduction   │  │ ┌─────────────────────────┐ │
│ data/reduced│◄─┤ 2. Feature Eng │  │ │  AssessmentAgent        │ │
│ data/proc/  │  │ 3. TF-IDF Vec  │  │ │  - State Management     │ │
│             │  │ 4. Log. Reg.   │  │ │  - Synthetic RAG (KB)   │ │
│             │  │ 5. Predict     │  │ │  - Gemini API (Free)    │ │
└─────────────┘  │ 6. Evaluate    │  │ │  - Report Generation    │ │
                 └────────────────┘  │ │  - Question Generation  │ │
                                     │ └─────────────────────────┘ │
                                     └─────────────────────────────┘
```

---

## Milestone 1 — ML/NLP Pipeline

### 1. Data Layer
- Raw Stack Overflow question, answer, and tag data (CSV uploads)
- Reduced dataset (sampled to 15,000 rows via `data_reduction.py`)
- Processed dataset with engineered features (`feature_engineering.py`)

### 2. Text Preprocessing & Feature Extraction
- HTML tag removal and text normalization
- TF-IDF vectorization of question text (Title + Body)
- Engineered features: `max_answer_score`, `avg_answer_score`, `answer_score_variance`, `answer_count`, `ratio`

### 3. Machine Learning
- Supervised classifier: **Logistic Regression** (multi-class)
- Target: difficulty label (easy / medium / hard) derived from `difficulty_score`
- Train/test split: 70/30

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance breakdown

### 5. User Interface
- CSV upload for Questions, Answers, Tags
- Predicted vs True difficulty table
- Model performance metrics display
- Difficulty distribution and accuracy visualizations

---

## Milestone 2 — Agentic AI Assessment Design Assistant

### 6. Agentic AI Layer
- **LLM**: Google Gemini API (free tier, `gemini-2.5-flash`)
- **Synthetic RAG**: Built-in pedagogical best practices knowledge base retrieved at inference time
- **Explicit State Management**: Agent tracks configuration status, report count, and identified gaps
- **Prompt Engineering**: Constrained prompts prevent unsupported educational claims

### 7. Structured Report Generation
- Assessment quality summary
- Question difficulty distribution analysis
- Identified learning gaps
- Recommended assessment improvements
- Supporting pedagogical references
- Educational and ethical disclaimers

### 8. Extension — Automated Question Generation
- Users specify a weak topic and desired difficulty
- Agent generates a complete question with title, description, sample I/O, and pedagogical justification
