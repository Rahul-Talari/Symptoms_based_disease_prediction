import pandas as pd
import pytest

from symptom_dx.data import normalize_symptom, parse_symptoms, preprocess, run_preprocessing, to_corpus_lines


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("itching", "itching"),
        ("Skin Rash", "skin_rash"),
        ("dischromic _patches", "dischromic_patches"),
        ("  high-fever ", "high_fever"),
    ],
)
def test_normalize_symptom(raw, expected):
    assert normalize_symptom(raw) == expected


def test_parse_symptoms_removes_blanks_and_duplicates():
    assert parse_symptoms("Itching, skin rash, , itching") == ["itching", "skin_rash"]


RAW = pd.DataFrame(
    {
        "itching": [1, 1, 0, 0],
        "skin_rash": [1, 1, 0, 0],
        "dischromic _patches": [0, 0, 1, 0],
        "prognosis": ["Fungal Infection", "Fungal Infection", "Acne ", "Healthy"],
    }
)


def test_preprocess_drops_duplicates_and_rows_without_symptoms():
    assert preprocess(RAW).to_dict("records") == [
        {"symptoms": "itching, skin_rash", "prognosis": "Fungal Infection"},
        {"symptoms": "dischromic_patches", "prognosis": "Acne"},
    ]


def test_corpus_lines():
    assert to_corpus_lines(preprocess(RAW))[0] == "symptoms : itching, skin_rash -> prognosis : Fungal Infection"


def test_run_preprocessing_writes_csv_and_corpus(tmp_path):
    RAW.to_csv(tmp_path / "raw.csv", index=False)

    run_preprocessing(tmp_path / "raw.csv", tmp_path / "out" / "processed.csv", tmp_path / "out" / "corpus.txt")

    assert len(pd.read_csv(tmp_path / "out" / "processed.csv")) == 2
    assert len((tmp_path / "out" / "corpus.txt").read_text().splitlines()) == 2
