from transformers import pipeline

class QASystem:
    def __init__(
        self,
        qa_model: str = "deepset/roberta-base-squad2",
        llm_model: str = "google/flan-t5-base"
    ):
        # Extractive QA pipeline
        self.qa_pipeline = pipeline("question-answering", model=qa_model)
        # Generative LLM pipeline
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=llm_model,
            max_length=256,
            do_sample=False
        )

    def answer_extractive(self, question: str, context: str) -> str:
        """
        Span-based extractive QA.
        """
        try:
            res = self.qa_pipeline(question=question, context=context)
            return res.get("answer", "No extractive answer found.")
        except Exception:
            return "Error during extractive QA."

    def answer_multi_extractive(self, question: str, context_chunks: list) -> str:
        """
        Run extractive QA over multiple chunks and pick best.
        """
        best_ans = None
        best_score = -1
        for chunk in context_chunks:
            try:
                res = self.qa_pipeline(question=question, context=chunk)
                if res.get("score", 0) > best_score:
                    best_score = res["score"]
                    best_ans = res["answer"]
            except Exception:
                continue
        return best_ans or "No extractive answer found."

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate a free-form answer using the LLM.
        """
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        try:
            out = self.llm_pipeline(prompt)[0]["generated_text"]
            # Strip repeated prompt if echoed
            return out.replace(prompt, "").strip()
        except Exception:
            return "Error during generative LLM answer."
