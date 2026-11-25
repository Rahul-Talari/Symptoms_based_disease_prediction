"""File locations and settings read from environment variables or a `.env` file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

RAW_DATA = "data/raw/symbipredict_2022.csv"
PROCESSED_DATA = "data/processed/preprocessed_data.csv"
CORPUS = "data/processed/symptoms_corpus.txt"
VECTOR_INDEX_DIR = "artifacts/vector_index"
RESULTS_DIR = "results"


class MissingSettingError(RuntimeError):
    """A required environment variable is not set."""


@dataclass
class Settings:
    groq_api_key: str
    groq_model: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    ollama_base_url: str
    embedding_model: str

    def require(self, *names: str) -> None:
        """Raise MissingSettingError if any of the named settings is empty."""
        missing = [name.upper() for name in names if not getattr(self, name)]
        if missing:
            raise MissingSettingError(f"Set {', '.join(missing)} in your environment or in a .env file.")


def load_settings() -> Settings:
    """Read settings from the environment, after loading `.env` from the current folder."""
    load_dotenv(".env")
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        neo4j_uri=os.getenv("NEO4J_URI", ""),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
    )
