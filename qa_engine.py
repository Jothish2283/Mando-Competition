from transformers import pipeline

class QASystem:
    def __init__(self, qa_model="deepset/roberta-base-squad2"):
        # Initialize a question answering pipeline
        self.qa_pipeline = pipeline("question-answering", model=qa_model)
        
    def answer_question(self, question, context):
        """
        Given a question and a context, use the QA model to produce an answer.
        """
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']

    def answer_with_multiple_context(self, question, context_chunks):
        """
        For multiple text chunks, choose the one with the highest confidence score.
        """
        best_answer = None
        best_score = -1
        
        for chunk in context_chunks:
            try:
                ans = self.qa_pipeline(question=question, context=chunk)
                if ans['score'] > best_score:
                    best_score = ans['score']
                    best_answer = ans['answer']
            except Exception as e:
                continue  # Skip on error
        return best_answer if best_answer is not None else "The answer is not available in the provided documents."
