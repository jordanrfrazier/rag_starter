from typing import Optional
from langchain.embeddings import OpenAIEmbeddings


def open_ai_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Load OpenAI embeddings.
    """

    return OpenAIEmbeddings(model=model) if model is not None else OpenAIEmbeddings()
