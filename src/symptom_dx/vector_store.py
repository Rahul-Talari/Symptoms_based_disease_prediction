"""Vector index of the dataset records (LlamaIndex + Ollama embeddings)."""

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq

from symptom_dx.config import CORPUS, VECTOR_INDEX_DIR, Settings
from symptom_dx.prompts import question_for


def get_embedding_model(settings: Settings) -> OllamaEmbedding:
    return OllamaEmbedding(model_name=settings.embedding_model, base_url=settings.ollama_base_url)


def build_index(settings: Settings) -> int:
    """Embed each corpus line as one document and save the index. Returns the number of documents."""
    lines = [line for line in Path(CORPUS).read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes = [TextNode(text=line) for line in lines]
    index = VectorStoreIndex(nodes, embed_model=get_embedding_model(settings), show_progress=True)
    index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)
    return len(nodes)


class VectorStoreRetriever:
    """Finds the dataset records closest to the symptoms and lets the LLM answer from them."""

    name = "Vector database"

    def __init__(self, settings: Settings, top_k: int = 5):
        if not Path(VECTOR_INDEX_DIR).exists():
            raise FileNotFoundError(f"No vector index in {VECTOR_INDEX_DIR}. Run: symptom-dx build-index")
        settings.require("groq_api_key")

        storage = StorageContext.from_defaults(persist_dir=VECTOR_INDEX_DIR)
        index = load_index_from_storage(storage, embed_model=get_embedding_model(settings))
        llm = Groq(model=settings.groq_model, api_key=settings.groq_api_key)
        self.query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)

    def retrieve(self, symptoms: list[str]) -> str:
        return str(self.query_engine.query(question_for(symptoms)))
