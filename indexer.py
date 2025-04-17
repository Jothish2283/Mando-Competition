import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticIndexer:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.text_chunks = []
        self.index = None
        self.embeddings = None

    def add_document(self, doc_text, chunk_size=200, overlap=50):
        """
        Split the document text into overlapping chunks.
        """
        words = doc_text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            self.text_chunks.append(chunk)
            start += (chunk_size - overlap)

    def build_index(self):
        """Generate embeddings and build a FAISS index."""
        if not self.text_chunks:
            raise ValueError("No text chunks to index.")
        self.embeddings = self.model.encode(
            self.text_chunks, show_progress_bar=False, convert_to_numpy=True
        )
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def query(self, query_text, top_k=5):
        """
        Return the top_k most similar text chunks to the query.
        Returns a list of (chunk, distance).
        """
        if self.index is None:
            raise ValueError("Index has not been built.")
        q_vec = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.text_chunks[idx], float(dist)))
        return results
