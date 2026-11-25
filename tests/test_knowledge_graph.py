import pandas as pd

from symptom_dx.knowledge_graph import to_graph_rows
from symptom_dx.prompts import CYPHER_GENERATION_TEMPLATE, GRAPH_SCHEMA


def test_to_graph_rows_groups_symptoms_by_disease():
    df = pd.DataFrame(
        {
            "symptoms": ["itching, skin_rash", "skin_rash, nodal_skin_eruptions", "cough"],
            "prognosis": ["Fungal Infection", "Fungal Infection", "Bronchial Asthma"],
        }
    )
    assert to_graph_rows(df) == [
        {"disease": "Bronchial Asthma", "symptoms": ["cough"]},
        {"disease": "Fungal Infection", "symptoms": ["itching", "nodal_skin_eruptions", "skin_rash"]},
    ]


def test_cypher_prompt_uses_the_same_direction_as_the_graph():
    assert "(Disease)-[:HAS_SYMPTOM]->(Symptom)" in GRAPH_SCHEMA
    assert "(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)" in CYPHER_GENERATION_TEMPLATE
    assert "<-[:HAS_SYMPTOM]-" not in CYPHER_GENERATION_TEMPLATE
