"""Neo4j knowledge graph: (:Disease {name})-[:HAS_SYMPTOM]->(:Symptom {name})."""

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from symptom_dx.config import Settings
from symptom_dx.prompts import CYPHER_GENERATION_TEMPLATE, GRAPH_SCHEMA, question_for

# Data is passed as a parameter ($rows), never pasted into the query text.
LOAD_QUERY = """
UNWIND $rows AS row
MERGE (d:Disease {name: row.disease})
WITH d, row
UNWIND row.symptoms AS symptom_name
MERGE (s:Symptom {name: symptom_name})
MERGE (d)-[:HAS_SYMPTOM]->(s)
"""


def connect(settings: Settings) -> Neo4jGraph:
    settings.require("neo4j_uri", "neo4j_password")
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )


def to_graph_rows(df: pd.DataFrame) -> list[dict]:
    """Group the processed dataset into one {disease, symptoms} row per disease."""
    symptoms_by_disease: dict[str, set[str]] = {}
    for row in df.itertuples():
        symptoms_by_disease.setdefault(row.prognosis, set()).update(row.symptoms.split(", "))
    return [
        {"disease": disease, "symptoms": sorted(symptoms)} for disease, symptoms in sorted(symptoms_by_disease.items())
    ]


def build_graph(graph: Neo4jGraph, df: pd.DataFrame, reset: bool = False) -> None:
    """Load the dataset into Neo4j. Safe to re-run; `reset=True` first deletes everything in the database."""
    if reset:
        graph.query("MATCH (n) DETACH DELETE n")
    graph.query("CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE")
    graph.query(LOAD_QUERY, params={"rows": to_graph_rows(df)})


class KnowledgeGraphRetriever:
    """The LLM writes a Cypher query for the question, runs it, and answers from the results."""

    name = "Knowledge graph"

    def __init__(self, graph: Neo4jGraph, llm, verbose: bool = False):
        prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE).partial(schema=GRAPH_SCHEMA)
        self.chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=prompt,
            verbose=verbose,
            # Required by LangChain because the LLM-written query runs on the database.
            # Use a read-only Neo4j user for predictions.
            allow_dangerous_requests=True,
        )

    def retrieve(self, symptoms: list[str]) -> str:
        return self.chain.invoke({"query": question_for(symptoms)})["result"]
