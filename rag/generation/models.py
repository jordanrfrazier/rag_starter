from typing import Optional
from langchain.chat_models import ChatOpenAI


def chat_open_ai(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model) if model is not None else ChatOpenAI()
