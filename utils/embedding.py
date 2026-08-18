from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model():
    """
    Load the HuggingFace embedding model.
    """

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding