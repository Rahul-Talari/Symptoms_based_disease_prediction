"""Ask the chosen sources about the symptoms, then merge their answers into one report.

Sources: "kg" (knowledge graph), "vector" (vector database), "web" (web search agent).
"""

import logging
from dataclasses import dataclass, field

from symptom_dx.config import Settings
from symptom_dx.data import parse_symptoms
from symptom_dx.prompts import synthesis_messages

SOURCES = ("kg", "vector", "web")

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    symptoms: list[str]
    answers: dict[str, str]  # source name -> that source's answer
    errors: dict[str, str] = field(default_factory=dict)  # source name -> error message
    report: str = ""


def build_retrievers(sources: list[str], settings: Settings, llm, verbose: bool = False) -> list:
    """Create a retriever for each source. Imports happen here so unused sources need nothing installed or running."""
    retrievers = []
    for source in sources:
        if source == "kg":
            from symptom_dx.knowledge_graph import KnowledgeGraphRetriever, connect

            retrievers.append(KnowledgeGraphRetriever(connect(settings), llm, verbose=verbose))
        elif source == "vector":
            from symptom_dx.vector_store import VectorStoreRetriever

            retrievers.append(VectorStoreRetriever(settings))
        elif source == "web":
            from symptom_dx.web_agent import WebSearchRetriever

            retrievers.append(WebSearchRetriever(llm, verbose=verbose))
        else:
            raise ValueError(f"Unknown source {source!r}. Choose from {SOURCES}.")
    return retrievers


def predict(symptoms_text: str, retrievers: list, llm) -> Prediction:
    """Ask every retriever, skip any that fail, and have the LLM merge the answers."""
    symptoms = parse_symptoms(symptoms_text)
    if not symptoms:
        raise ValueError("No symptoms given.")

    prediction = Prediction(symptoms=symptoms, answers={})
    for retriever in retrievers:
        try:
            prediction.answers[retriever.name] = retriever.retrieve(symptoms)
        except Exception as error:
            log.warning("%s failed: %s", retriever.name, error)
            prediction.errors[retriever.name] = str(error)

    if not prediction.answers:
        raise RuntimeError(f"Every source failed: {prediction.errors}")

    prediction.report = llm.invoke(synthesis_messages(symptoms, prediction.answers)).content
    return prediction
