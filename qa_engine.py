# qa_engine.py

import os
from transformers import pipeline
import google.generativeai as genai

# Configure Gemini API key (set GEMINI_API_KEY in your env)
genai.configure(api_key=os.getenv("AIzaSyDKfHn328xqqYRsFS3CZqiyr1T5ef-1I7w"))

class QASystem:
    def __init__(
        self,
        qa_model: str = "deepset/roberta-base-squad2",
        gemini_model_name: str = "gemini-1.5-flash"
    ):
        # 1) Extractive QA pipeline (unchanged)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_model,
            device=-1
        )

        # 2) Generative Gemini model
        #    Uses Google’s official SDK to call Gemini LLM
        self.gemini = genai.GenerativeModel(model_name=gemini_model_name)

    def answer_extractive(self, question: str, context: str) -> str:
        """Span‑based extractive QA."""
        try:
            out = self.qa_pipeline(question=question, context=context)
            return out.get("answer", "No extractive answer found.")
        except Exception:
            return "Error during extractive QA."

    def generate_with_gemini(self, question: str, context: str) -> str:
        """
        Use Gemini LLM to both infer & generate. We prefix the
        context so the model can perform arbitrary analyses.
        """
        prompt = (
            f"Here is some context:\n{context}\n\n"
            f"Based on the above, {question}\n\nAnswer:"
        )
        try:
            resp = self.gemini.generate_content(prompt)
            # .text holds the generated text
            return resp.text.strip()
        except Exception as e:
            return f"Gemini error: {e}"
