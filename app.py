import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from scripts.data_reduction import reduce_datasets
from scripts.feature_engineering import build_model_data
from scripts.assessment_agent import AssessmentAgent
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="ExamWise", layout="wide")

with st.sidebar:
    st.header("🔑 AI Settings")
    api_key_input = st.text_input("Gemini API Key", type="password", help="Leave blank if set in environment variable GEMINI_API_KEY")
    
agent = AssessmentAgent(api_key=api_key_input)

st.title("ExamWise - Question Difficulty Analysis System")


# ---------------------------------
# 1. Upload Dataset Files
# ---------------------------------
st.header("1. Upload Dataset Files")

questions_file = st.file_uploader("Upload Questions.csv", type=["csv"])
answers_file = st.file_uploader("Upload Answers.csv", type=["csv"])
tags_file = st.file_uploader("Upload Tags.csv", type=["csv"])

if not questions_file or not answers_file or not tags_file:
    st.info("Please upload Questions.csv, Answers.csv, and Tags.csv to proceed.")
    st.stop()

# Save uploaded files
os.makedirs("data/raw", exist_ok=True)

with open("data/raw/Questions.csv", "wb") as f:
    f.write(questions_file.getbuffer())

with open("data/raw/Answers.csv", "wb") as f:
    f.write(answers_file.getbuffer())

with open("data/raw/Tags.csv", "wb") as f:
    f.write(tags_file.getbuffer())

st.success("All files uploaded successfully.")


# ---------------------------------
# 2. Run Pipeline
# ---------------------------------
st.header("2. Run Full Analysis")

if st.button("Run ML Pipeline"):

    with st.spinner("Running preprocessing and prediction..."):

        # Step 1: Reduce dataset
        reduce_datasets(limit=15000)

        # Step 2: Feature Engineering
        build_model_data()

        # Step 3: Load processed data
        df = pd.read_csv("data/processed/processed_questions.csv")

        # Combine title and body for NLP model input
        df["question_text"] = (
            df["Title"].fillna("") + " " + df["question_body"].fillna("")
        )

        # Step 4: Load Model
        model = joblib.load("models/model.joblib")
        vectorizer = joblib.load("models/vectorizer.joblib")

        X = vectorizer.transform(df["question_text"])
        df["predicted_difficulty"] = model.predict(X)

    st.success("Pipeline completed successfully.")


    # ---------------------------------
    # 3. Predicted Difficulty Table
    # ---------------------------------
    st.header("3. Predicted Difficulty Levels")

    display_df = df[[
        "Title",
        "difficulty_label",
        "predicted_difficulty"
    ]].rename(columns={
        "Title": "Question Title",
        "difficulty_label": "True Difficulty",
        "predicted_difficulty": "Predicted Difficulty"
    })

    st.dataframe(display_df.head(50), use_container_width=True)


    # ---------------------------------
    # 4. Model Performance Metrics
    # ---------------------------------
    st.header("4. Model Performance")

    if os.path.exists("models/metrics.txt"):
        with open("models/metrics.txt", "r") as f:
            st.text(f.read())
    else:
        st.warning("metrics.txt not found in models directory.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    labels = ["easy", "medium", "hard"]
    cm = confusion_matrix(df["difficulty_label"], df["predicted_difficulty"], labels=labels)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix: True vs Predicted Difficulty")
    st.pyplot(fig_cm)

    # ---------------------------------
    # 4.5 Student Performance Patterns
    # ---------------------------------
    st.header("4.5 Student Performance Patterns Per Question")
    if os.path.exists("outputs/accuracy_per_question.csv"):
        perf_df = pd.read_csv("outputs/accuracy_per_question.csv")
        st.dataframe(perf_df.head(30), use_container_width=True)
    else:
        st.info("No per-question performance data available yet.")


    # ---------------------------------
    # 5. Analytics and Visualizations
    # ---------------------------------
    st.header("5. Analytics and Visualizations")

    if os.path.exists("outputs/difficulty_distribution.png"):
        st.image(
            "outputs/difficulty_distribution.png",
            caption="Difficulty Distribution",
            use_container_width=True
        )

    if os.path.exists("outputs/accuracy_distribution.png"):
        st.image(
            "outputs/accuracy_distribution.png",
            caption="Accuracy Distribution",
            use_container_width=True
        )

    if os.path.exists("outputs/hardest_questions.csv"):
        st.subheader("Hardest Questions")
        hardest = pd.read_csv("outputs/hardest_questions.csv")
        st.dataframe(hardest.head(20), use_container_width=True)

# ---------------------------------
# 6. Agentic Assessment Assistant
# ---------------------------------
if os.path.exists("data/processed/processed_questions.csv"):
    st.header("6. Agentic Assessment Assistant")
    df_processed = pd.read_csv("data/processed/processed_questions.csv")

    # Data quality warnings for noisy/incomplete data
    total = len(df_processed)
    missing_titles = df_processed["Title"].isna().sum()
    missing_bodies = df_processed["question_body"].isna().sum()
    if missing_titles > 0 or missing_bodies > 0:
        st.warning(f"⚠️ Data Quality Notice: {missing_titles} questions have missing titles and {missing_bodies} have missing body text. The agent will handle these gracefully but results may be less precise.")
    if total < 100:
        st.warning(f"⚠️ Small dataset detected ({total} questions). Assessment analysis works best with larger samples.")

    if st.button("Generate Assessment Design Report"):
        if not api_key_input and not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter a Gemini API Key in the sidebar or set GEMINI_API_KEY.")
        else:
            with st.spinner("Agent is analyzing assessment data..."):
                report = agent.generate_report(df_processed)
                st.markdown(report)
                
    st.subheader("Automated Question Generation (Extension)")
    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input("Enter a weak topic (e.g., 'Dynamic Programming' or 'SQL Joins')")
    with col2:
        diff_level = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"], index=1)
        
    if st.button("Generate Sample Question"):
        if not topic_input:
            st.warning("Please enter a topic.")
        elif not api_key_input and not os.environ.get("GEMINI_API_KEY"):
            st.error("Please enter a Gemini API Key in the sidebar or set GEMINI_API_KEY.")
        else:
            with st.spinner(f"Generating {diff_level} question for '{topic_input}'..."):
                question = agent.generate_automated_question(topic_input, diff_level)
                st.markdown(question)