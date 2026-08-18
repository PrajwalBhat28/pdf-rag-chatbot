from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector store from document chunks.
    """
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_db