import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticIndexer:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.text_chunks = []
        self.embeddings = None

    def add_document(self, doc_text, chunk_size=200, overlap=50):
        """
        Split the document text into chunks with some overlap.
        """
        words = doc_text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += (chunk_size - overlap)
        self.text_chunks.extend(chunks)

    def build_index(self):
        """Generate embeddings and create a FAISS index."""
        if not self.text_chunks:
            return None
        # Generate embeddings (each row is an embedding vector)
        self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=True, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
    
    def query(self, query_text, top_k=5):
        """Perform a semantic search on the indexed chunks."""
        query_vector = self.model.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(query_vector, top_k)
        # Return the matching text chunks along with distances
        results = [(self.text_chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
        return results
