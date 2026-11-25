"""All prompts in one place."""

GRAPH_SCHEMA = """\
Nodes:
- Disease: {name: string}
- Symptom: {name: string}
Relationships:
- (Disease)-[:HAS_SYMPTOM]->(Symptom)
"""

# Used by the knowledge graph: the LLM turns a question into a Cypher query.
CYPHER_GENERATION_TEMPLATE = """\
Task: Generate a Cypher statement to query a graph database.

Instructions:
- Use only the node labels, relationship types and properties in the schema.
- Symptom names are lower-case snake_case, e.g. "skin_rash", "high_fever".
- Rank diseases by how many of the given symptoms they match.
- Return only the Cypher statement: no explanations, apologies or markdown.

Schema:
{schema}

Examples:
# What diseases are associated with itching, skin_rash, nodal_skin_eruptions?
MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
WHERE s.name IN ["itching", "skin_rash", "nodal_skin_eruptions"]
RETURN d.name AS disease, count(s) AS matching_symptoms
ORDER BY matching_symptoms DESC
LIMIT 5

# Which symptoms does Malaria have?
MATCH (d:Disease {{name: "Malaria"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name AS symptom

The question is:
{question}"""

# Used by the web search agent.
WEB_AGENT_SYSTEM_PROMPT = (
    "You find diseases that match a set of symptoms. Search the web, then answer with "
    "the top candidate diseases, each with a one-line reason."
)

# Used for the final step that merges the answers from all sources.
SYNTHESIS_SYSTEM_PROMPT = (
    "You are a careful, detail-oriented medical information assistant. You critically "
    "compare evidence from several sources, discard errors and duplicates, and "
    "never present a diagnosis as certain."
)

SYNTHESIS_INSTRUCTIONS = """\
Review the sources above for mistakes, inconsistencies and duplicate diseases, then answer with these sections:

1. **Likely conditions**: grouped as high, medium and low likelihood. List each disease once, and prioritise by how well it matches the symptoms and by severity.
2. **Diet**: specific foods and drinks to favour and to avoid.
3. **Natural techniques and exercises**: safe, simple measures that complement medical treatment.
4. **Who to consult**: the type of doctor or specialist, with a one-line reason each.

Be clear and concise. End with a one-sentence reminder that this is not a diagnosis and a clinician should be consulted."""


def question_for(symptoms: list[str]) -> str:
    """The question every source is asked."""
    return f"What diseases are associated with these symptoms: {', '.join(symptoms)}?"


def synthesis_messages(symptoms: list[str], answers: dict[str, str]) -> list[tuple[str, str]]:
    """Chat messages asking the LLM to merge the answers from every source."""
    sources = "\n\n".join(f"### {name}\n{answer.strip()}" for name, answer in answers.items())
    question = f"Patient-reported symptoms: {', '.join(symptoms)}\n\n{sources}\n\n{SYNTHESIS_INSTRUCTIONS}"
    return [("system", SYNTHESIS_SYSTEM_PROMPT), ("human", question)]
