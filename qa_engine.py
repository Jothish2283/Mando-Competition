# qa_engine.py

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class QASystem:
    def __init__(
        self,
        qa_model: str = "deepset/roberta-base-squad2",
        llm_model: str = "google/flan-t5-base"
    ):
        # 1) Extractive QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_model,
            device=-1
        )

        # 2) Generative LLM pipeline (explicitly load full weights to CPU)
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            llm_model,
            low_cpu_mem_usage=False  # disable meta‑tensor loading
        )
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,      # force CPU
            max_length=256,
            do_sample=False
        )

    def answer_extractive(self, question: str, context: str) -> str:
        try:
            out = self.qa_pipeline(question=question, context=context)
            return out.get("answer", "No extractive answer found.")
        except Exception:
            return "Error during extractive QA."

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        try:
            out = self.llm_pipeline(prompt)[0]["generated_text"]
            return out.replace(prompt, "").strip()
        except Exception:
            return "Error during generative LLM answer."
