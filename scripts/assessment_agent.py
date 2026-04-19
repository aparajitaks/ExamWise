import os
import google.generativeai as genai
import pandas as pd

# Basic pedagogical best practices acting as our "synthetic RAG" knowledge base
PEDAGOGICAL_BEST_PRACTICES = {
    "difficulty_distribution": "A standard exam should aim for approximately 20% easy, 60% medium, and 20% hard questions to adequately differentiate student ability while maintaining confidence.",
    "learning_gaps": "When students consistently score low on specific topics, it indicates a critical learning gap. Recommend targeted supplementary materials, scaffolding for complex topics, and formative assessment prior to summative exams.",
    "cognitive_load": "Avoid unnecessary complexity (extraneous cognitive load) in question phrasing. Clear, concise language ensures we assess the skill, not reading comprehension.",
    "constructive_alignment": "Ensure exam questions directly align with the stated learning objectives of the course.",
    "disclaimer": "This analysis is AI-generated and should serve as an assistive tool for educators, rather than a definitive authority on student performance or assessment quality."
}

class AssessmentAgent:
    def __init__(self, api_key=None):
        self.state = {
            "is_configured": False,
            "reports_generated": 0,
            "last_identified_gaps": []
        }
        
        # Try to configure with provided key or environment variable
        key_to_use = api_key or os.environ.get("GEMINI_API_KEY")
        if key_to_use:
            genai.configure(api_key=key_to_use)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.state["is_configured"] = True
    
    def format_knowledge_base(self):
        kb_text = "Pedagogical Guidelines to Follow:\n"
        for k, v in PEDAGOGICAL_BEST_PRACTICES.items():
            kb_text += f"- {v}\n"
        return kb_text

    def prepare_data_summary(self, df):
        diff_counts = df['difficulty_label'].value_counts().to_dict()
        total_q = len(df)
        
        # Data quality checks for noisy/incomplete data
        missing_titles = df['Title'].isna().sum()
        missing_bodies = df['question_body'].isna().sum() if 'question_body' in df.columns else 0
        zero_score_pct = (df['difficulty_score'] == 0).sum() / max(total_q, 1) * 100
        
        data_quality_notes = ""
        if missing_titles > 0:
            data_quality_notes += f"- {missing_titles} questions have missing titles.\n"
        if missing_bodies > 0:
            data_quality_notes += f"- {missing_bodies} questions have missing body text.\n"
        if zero_score_pct > 30:
            data_quality_notes += f"- {zero_score_pct:.1f}% of questions have a difficulty score of 0, suggesting sparse answer data.\n"
        if total_q < 100:
            data_quality_notes += f"- Only {total_q} questions in the dataset — a small sample size may limit analysis reliability.\n"
        
        # Approximate the hardest questions
        hardest = df[df['difficulty_label'] == 'hard'].nlargest(10, 'difficulty_score')
        hard_titles = hardest['Title'].fillna("(missing title)").tolist()
        
        summary = f"""
        Total Questions Analyzed: {total_q}
        Difficulty Distribution: {diff_counts}
        """
        
        if data_quality_notes:
            summary += f"""
        Data Quality Notes (handle gracefully):
        {data_quality_notes}
        """
        
        summary += "\n        Sample Hardest Questions (Highest Difficulty Scores) used to infer learning gaps:\n"
        for title in hard_titles:
            summary += f"- {title}\n"
            
        return summary

    def generate_report(self, df):
        if not self.state["is_configured"]:
            return "Error: Please provide a valid Gemini API Key to run the Assessment Agent."
        
        data_summary = self.prepare_data_summary(df)
        kb_context = self.format_knowledge_base()
        
        prompt = f"""
        You are an expert Educational Assessment Designer and AI Agent. 
        Your task is to analyze the following assessment data and generate a structured assessment design report.
        
        {kb_context}
        
        Data Summary:
        {data_summary}
        
        Requirements for the report:
        1. Assessment Quality Summary: Evaluate the overall balance based on the difficulty distribution.
        2. Question Difficulty Distribution: Briefly summarize the numbers provided.
        3. Identified Learning Gaps: Based on the sample hardest questions, infer potential learning gaps or challenging topics.
        4. Recommended Assessment Improvements: Provide actionable advice on how to improve the exam design.
        5. Supporting Pedagogical References: Cite the pedagogical guidelines provided.
        6. Educational and Ethical Disclaimers: Include a disclaimer that this is AI-assisted analysis and educators should hold the final judgment.
        
        Format the output in clean Markdown. Avoid making unsupported educational claims outside of the provided guidelines.
        """
        
        try:
            response = self.model.generate_content(prompt)
            self.state["reports_generated"] += 1
            return response.text
        except Exception as e:
            return f"Error generating report: {str(e)}"
            
    def generate_automated_question(self, topic, difficulty="Medium"):
        if not self.state["is_configured"]:
            return "Error: Please provide a valid Gemini API Key to use this feature."
            
        prompt = f"""
        You are an expert Educational Assessment Designer.
        Generate a {difficulty} level programming question related to the topic: "{topic}".
        Include:
        1. **Question Title**
        2. **Question Description** (with clear requirements and constraints)
        3. **Sample Input / Output**
        4. **Pedagogical Justification** (Briefly explain why this question tests the concept well and how it aligns with best practices).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating question: {str(e)}"
