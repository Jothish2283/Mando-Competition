import os
os.environ["STREAMLIT_SERVER_ENABLEFILEWATCHER"] = "false"

import streamlit as st
from document_ingestion import (
    ingest_pdf, ingest_docx, ingest_pptx,
    ingest_csv, ingest_excel, ingest_json,
    ingest_txt, ingest_image
)
from indexer import SemanticIndexer
from qa_engine import QASystem
from web_crawler import crawl_url

st.set_page_config(page_title="Mando Q&A", layout="wide")
st.title("Mando: Document Q&A System")

# 1) File upload & optional URL
files = st.file_uploader(
    "Upload documents (PDF, DOCX, PPTX, CSV, XLSX, JSON, TXT, PNG/JPG)",
    accept_multiple_files=True
)
url = st.text_input("Optional: URL to crawl for extra content")

if st.button("Process & Index"):
    if not files and not url:
        st.error("Please upload at least one file or enter a URL.")
    else:
        aggregated = ""
        for f in files:
            name = f.name.lower()
            data = f.read()
            st.write(f"• Processing `{name}`")
            if name.endswith(".pdf"):
                aggregated += "\n" + ingest_pdf(data)
            elif name.endswith(".docx"):
                aggregated += "\n" + ingest_docx(data)
            elif name.endswith(".pptx"):
                aggregated += "\n" + ingest_pptx(data)
            elif name.endswith(".csv"):
                _, txt = ingest_csv(data)
                aggregated += "\n" + txt
            elif name.endswith((".xls", ".xlsx")):
                _, txt = ingest_excel(data)
                aggregated += "\n" + txt
            elif name.endswith(".json"):
                aggregated += "\n" + ingest_json(data)
            elif name.endswith(".txt"):
                aggregated += "\n" + ingest_txt(data)
            elif name.endswith((".png", ".jpg", ".jpeg")):
                aggregated += "\n" + ingest_image(data)
            else:
                aggregated += f"\n[Unsupported format: {name}]"

        if url.strip():
            st.write(f"• Crawling `{url}`")
            aggregated += "\n" + crawl_url(url)

        indexer = SemanticIndexer()
        indexer.add_document(aggregated)
        indexer.build_index()
        st.session_state['indexer'] = indexer
        st.success("✅ Indexed all content.")

# 2) Ask a question
qa = QASystem()
question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    idx = st.session_state.get('indexer')
    if not idx:
        st.error("No index found. Click ‘Process & Index’ first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        hits = idx.query(question, top_k=5)
        context = "\n".join(chunk for chunk, _ in hits)
        st.subheader("Retrieved Context")
        st.write(context)

        st.subheader("Extractive Answer")
        st.write(qa.answer_extractive(question, context))

        st.subheader("Generative Answer (Gemini LLM)")
        st.write(qa.generate_with_gemini(question, context))
