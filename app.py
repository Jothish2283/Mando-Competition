import os
os.environ["STREAMLIT_SERVER_ENABLEFILEWATCHER"] = "false"

import streamlit as st
import io
from document_ingestion import (ingest_pdf, ingest_docx, ingest_pptx, 
                                ingest_csv, ingest_excel, ingest_json, 
                                ingest_txt, ingest_image)
from indexer import SemanticIndexer
from qa_engine import QASystem
from web_crawler import crawl_url
import pandas as pd
import json

st.title("Mando Q&A System")
st.write("Upload your documents and ask a question!")

# Upload multiple files
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
additional_url = st.text_input("Optional: Enter a URL to crawl for additional content")

# Instantiate indexer and QA system
indexer = SemanticIndexer()
qa_system = QASystem()

if st.button("Process Documents"):
    all_text = ""
    # Process each uploaded file by extension
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()
        st.write(f"Processing {file_name} ...")
        
        if file_name.endswith(".pdf"):
            text = ingest_pdf(file_bytes)
        elif file_name.endswith(".docx"):
            text = ingest_docx(file_bytes)
        elif file_name.endswith(".pptx"):
            text = ingest_pptx(file_bytes)
        elif file_name.endswith(".csv"):
            # For CSV, we load the DataFrame and then convert to string
            df, csv_text = ingest_csv(file_bytes)
            text = csv_text
        elif file_name.endswith((".xlsx", ".xls")):
            df, csv_text = ingest_excel(file_bytes)
            text = csv_text
        elif file_name.endswith(".json"):
            text = ingest_json(file_bytes)
        elif file_name.endswith(".txt"):
            text = ingest_txt(file_bytes)
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            text = ingest_image(file_bytes)
        else:
            text = f"Unsupported file format: {file_name}"
        
        all_text += "\n" + text
    
    # Optionally fetch content from URL
    if additional_url:
        st.write(f"Crawling URL: {additional_url}")
        crawled_text = crawl_url(additional_url)
        all_text += "\n" + crawled_text
    
    st.success("Documents processed! Building index...")
    
    # Add the combined text into the indexer
    indexer.add_document(all_text)
    indexer.build_index()
    st.session_state['indexer'] = indexer  # Save indexer in session

    st.success("Index built!")

# Ask a question and get an answer
question = st.text_input("Ask your question:")

if st.button("Answer") and question:
    # Retrieve top matching chunks from the index
    indexer = st.session_state.get('indexer', None)
    if not indexer:
        st.error("No index available. Please process documents first.")
    else:
        results = indexer.query(question, top_k=5)
        # Concatenate the top chunks to form a context
        context = "\n".join([chunk for chunk, score in results])
        st.write("**Retrieved Context:**")
        st.write(context)
        # Get answer from QA system
        answer = qa_system.answer_question(question, context)
        st.write("**Answer:**")
        st.write(answer)
