import streamlit as st

from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embedding import load_embedding_model
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.rag_chain import get_rag_response


st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.processed_file_name != uploaded_file.name:
        st.success("✅ PDF uploaded successfully!")

        with st.spinner("Processing PDF..."):
            text, page_documents, paper_title = extract_text_from_pdf(uploaded_file)
            chunks = split_text(page_documents)
            embedding_model = load_embedding_model()
            vector_db = create_vector_store(chunks, embedding_model)
            retriever = get_retriever(vector_db)

            st.session_state.text = text
            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.paper_title = paper_title
            st.session_state.messages = []

        st.success("✅ PDF processed successfully!")

    if st.session_state.text and st.session_state.chunks:
        st.subheader("📊 PDF Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Characters", len(st.session_state.text))

        with col2:
            st.metric("Chunks", len(st.session_state.chunks))

        with col3:
            st.metric("File Name", uploaded_file.name)

        st.markdown(f"**Paper Title:** {st.session_state.paper_title}")
        st.caption("The title is extracted from the first page of the PDF.")

        st.subheader("💬 Chat with your PDF")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the PDF")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, docs = get_rag_response(
                        st.session_state.retriever,
                        question,
                        st.session_state.text
                    )
                    st.markdown(answer)

                    if docs:
                        with st.expander("Source Chunks"):
                            for i, doc in enumerate(docs, start=1):
                                page = doc.metadata.get("page", "Unknown")
                                st.write(f"**Chunk {i} — Page {page}**")
                                st.write(doc.page_content)
                                st.write("---")

            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF to start chatting.")