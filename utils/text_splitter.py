from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(documents):
    """
    Split page-level Documents into smaller chunks while keeping metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks