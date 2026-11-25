"""Turn the raw one-hot dataset into readable symptom lists.

The raw CSV has one 0/1 column per symptom plus a `prognosis` column (the disease).
Each row becomes "symptom_a, symptom_b, ..." plus its disease.
"""

import re
from pathlib import Path

import pandas as pd


def normalize_symptom(name: str) -> str:
    """Lower-case snake_case: "Skin Rash" -> "skin_rash", "dischromic _patches" -> "dischromic_patches"."""
    name = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", name).strip("_")


def parse_symptoms(text: str) -> list[str]:
    """Split comma-separated user input into normalized symptoms, without blanks or duplicates."""
    symptoms = [normalize_symptom(part) for part in text.split(",")]
    return list(dict.fromkeys(s for s in symptoms if s))


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot table -> DataFrame with `symptoms` and `prognosis` columns."""
    raw = raw.drop_duplicates()
    symptom_columns = [column for column in raw.columns if column != "prognosis"]

    rows = []
    for _, row in raw.iterrows():
        present = [normalize_symptom(column) for column in symptom_columns if row[column] == 1]
        if present:
            rows.append({"symptoms": ", ".join(present), "prognosis": row["prognosis"].strip()})
    return pd.DataFrame(rows, columns=["symptoms", "prognosis"])


def to_corpus_lines(df: pd.DataFrame) -> list[str]:
    """One text line per record; these lines are the documents in the vector index."""
    return [f"symptoms : {row.symptoms} -> prognosis : {row.prognosis}" for row in df.itertuples()]


def run_preprocessing(raw_path: str, processed_path: str, corpus_path: str) -> pd.DataFrame:
    """Read the raw CSV, then write the processed CSV and the text corpus."""
    df = preprocess(pd.read_csv(raw_path))

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    Path(corpus_path).write_text("\n".join(to_corpus_lines(df)) + "\n", encoding="utf-8")
    return df
