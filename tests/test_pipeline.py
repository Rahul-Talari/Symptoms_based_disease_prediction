from types import SimpleNamespace

import pytest

from symptom_dx.pipeline import build_retrievers, predict


class FakeRetriever:
    def __init__(self, name, answer="", error=None):
        self.name = name
        self.answer = answer
        self.error = error
        self.asked_with = None

    def retrieve(self, symptoms):
        self.asked_with = symptoms
        if self.error:
            raise self.error
        return self.answer


class FakeLLM:
    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="merged report")


def test_predict_asks_every_source_and_merges_answers():
    kg = FakeRetriever("Knowledge graph", "Fungal infection")
    web = FakeRetriever("Web search", "Eczema")
    llm = FakeLLM()

    result = predict("Itching, Skin Rash", [kg, web], llm)

    assert kg.asked_with == web.asked_with == ["itching", "skin_rash"]
    assert result.answers == {"Knowledge graph": "Fungal infection", "Web search": "Eczema"}
    assert result.report == "merged report"
    assert "Fungal infection" in llm.messages[1][1] and "Eczema" in llm.messages[1][1]


def test_predict_skips_a_failing_source():
    working = FakeRetriever("Working", "answer")
    broken = FakeRetriever("Broken", error=ConnectionError("database down"))

    result = predict("fever", [broken, working], FakeLLM())

    assert result.answers == {"Working": "answer"}
    assert result.errors == {"Broken": "database down"}


def test_predict_fails_when_every_source_fails():
    with pytest.raises(RuntimeError):
        predict("fever", [FakeRetriever("Broken", error=ValueError("x"))], FakeLLM())


def test_predict_rejects_empty_input():
    with pytest.raises(ValueError):
        predict(" , ", [FakeRetriever("A", "a")], FakeLLM())


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        build_retrievers(["sql"], settings=None, llm=None)
