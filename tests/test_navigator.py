import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval import navigator


def test_extract_json_array():
    txt = "Here you go: [1, 3, 5] thanks"
    assert navigator._extract_json(txt) == [1, 3, 5]


def test_navigate_batch_validates_indices(monkeypatch):
    outline = [
        {"idx": 0, "heading": "A", "pages": "1-1", "summary": "x"},
        {"idx": 1, "heading": "B", "pages": "2-2", "summary": "y"},
    ]

    def fake_chat(prompt, model=None, host=None, num_ctx=4096):
        return "[0, 99, 1]"

    monkeypatch.setattr(navigator, "_chat", fake_chat)
    res = navigator.navigate_batch(outline, "q", k=5)
    assert res == [0, 1]


def test_navigate_returns_empty_on_failure(monkeypatch):
    outline = [{"idx": 0, "heading": "A", "pages": "1-1", "summary": "x"}]

    def fake_chat(prompt, model=None, host=None, num_ctx=4096):
        raise RuntimeError("no ollama")

    monkeypatch.setattr(navigator, "_chat", fake_chat)
    assert navigator.navigate(outline, "q") == []