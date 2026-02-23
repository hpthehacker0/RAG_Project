import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma


st.set_page_config(page_title="Industrial AI Copilot", layout="wide")

st.title("🏭 Industrial AI Copilot (Offline)")
st.write("Upload manuals, SOPs, safety documents, etc.")

# ---------- Embeddings ----------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ---------- Load existing DB if available ----------
if os.path.exists("chroma_db"):
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
else:
    db = None

# ---------- Upload PDFs ----------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    with st.spinner("Processing documents..."):

        for uploaded_file in uploaded_files:

            file_path = f"temp_{uploaded_file.name}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            all_docs.extend(docs)

        # ---------- Chunking ----------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(all_docs)

        # ---------- Create / Update ChromaDB ----------
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        

    st.success("Knowledge base ready ✅")

# ---------- Query Section ----------
if db:

    query = st.text_input("Ask a question")

    if query:

        with st.spinner("Thinking..."):

            retriever = db.as_retriever()

            relevant_docs = retriever.invoke(query)

            context = "\n".join([d.page_content for d in relevant_docs])

            # ---------- Local LLM ----------
            llm = OllamaLLM(model="llama2")

            prompt = f"""
You are an industrial assistant.

Answer ONLY using the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

        # ---------- Output ----------
        st.subheader("📌 Answer")
        st.write(response)

        # ---------- Sources ----------
        st.subheader("📄 Sources")

        for doc in relevant_docs:
            st.write(doc.metadata)