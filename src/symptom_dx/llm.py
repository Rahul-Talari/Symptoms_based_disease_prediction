"""The Groq chat model used by every step."""

from langchain_groq import ChatGroq

from symptom_dx.config import Settings


def get_chat_model(settings: Settings) -> ChatGroq:
    settings.require("groq_api_key")
    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.2, max_tokens=1024)
