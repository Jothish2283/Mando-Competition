import os
os.environ["STREAMLIT_SERVER_ENABLEFILEWATCHER"] = "false"

import streamlit as st
import io
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

# File upload
uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, PPTX, CSV, XLSX, JSON, TXT, PNG/JPG):",
    accept_multiple_files=True
)
url_input = st.text_input("Optional URL to crawl:")

if st.button("Process & Index"):
    if not uploaded_files and not url_input:
        st.error("Please upload files or enter a URL.")
    else:
        full_text = ""
        # Ingest each file
        for f in uploaded_files:
            name = f.name.lower()
            data = f.read()
            st.write(f"• Processing `{name}`")
            if name.endswith(".pdf"):
                text = ingest_pdf(data)
            elif name.endswith(".docx"):
                text = ingest_docx(data)
            elif name.endswith(".pptx"):
                text = ingest_pptx(data)
            elif name.endswith(".csv"):
                _, text = ingest_csv(data)
            elif name.endswith((".xls", ".xlsx")):
                _, text = ingest_excel(data)
            elif name.endswith(".json"):
                text = ingest_json(data)
            elif name.endswith(".txt"):
                text = ingest_txt(data)
            elif name.endswith((".png", ".jpg", ".jpeg")):
                text = ingest_image(data)
            else:
                text = f"[Unsupported format: {name}]"
            full_text += "\n" + text

        # Crawl URL if given
        if url_input.strip():
            st.write(f"• Crawling `{url_input}`")
            crawled = crawl_url(url_input)
            full_text += "\n" + crawled

        # Build semantic index
        indexer = SemanticIndexer()
        indexer.add_document(full_text)
        indexer.build_index()
        st.session_state['indexer'] = indexer
        st.success("✅ Indexed all content.")

qa_system = QASystem()

question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if 'indexer' not in st.session_state:
        st.error("No index found. Please process documents first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        idx = st.session_state['indexer']
        hits = idx.query(question, top_k=5)
        context = "\n".join([chunk for chunk, _ in hits])
        st.subheader("Retrieved Context")
        st.write(context)

        # Extractive span-based answer
        ext_ans = qa_system.answer_extractive(question, context)
        st.subheader("Extractive Answer")
        st.write(ext_ans)

        # Generative LLM answer
        gen_ans = qa_system.generate_answer(question, context)
        st.subheader("Generative Answer (LLM)")
        st.write(gen_ans)
