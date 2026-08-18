from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[Page {page}] {doc.page_content}")
    return "\n\n".join(formatted)


def is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(
        word in q
        for word in ["summary", "summarize", "summarise", "overview", "brief", "abstract"]
    )


def get_rag_response(retriever, question, full_text=None):
    """
    Answer questions from the PDF.
    For summary questions, use the full extracted text.
    For normal questions, use retrieval from FAISS.
    """
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    if is_summary_question(question) and full_text:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Summarize the uploaded paper clearly and concisely. "
                "Include the goal, method, dataset if mentioned, key results, and conclusion. "
                "Use only the provided text."
            ),
            (
                "human",
                "Paper text:\n{text}\n\nWrite a summary of this paper."
            )
        ])

        messages = prompt.format_messages(text=full_text)
        response = llm.invoke(messages)
        return response.content, []

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer only using the provided context. "
            "If the answer is not in the context, say: "
            "\"I couldn't find the answer in the uploaded PDF.\""
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    return response.content, docs