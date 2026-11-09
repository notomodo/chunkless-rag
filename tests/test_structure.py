import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.structure import build_tree, assign_ids, to_extracted_doc, section_metadata


def test_build_tree_return_policy():
    pdf = Path(__file__).resolve().parents[1] / "data" / "raw" / "Return_Policy.pdf"
    tree = build_tree(pdf)
    assign_ids(tree)
    assert tree.pages == 2
    assert len(tree.sections) >= 5
    for s in tree.sections:
        assert s.section_id
        assert s.text.strip()
        assert s.page_start >= 1
        assert s.page_end >= s.page_start


def test_to_extracted_doc():
    pdf = Path(__file__).resolve().parents[1] / "data" / "raw" / "Return_Policy.pdf"
    tree = build_tree(pdf)
    assign_ids(tree)
    doc = to_extracted_doc(tree, 400, None)
    assert doc.sha256 == tree.sha256
    assert len(doc.chunks) == len(tree.sections)
    for c in doc.chunks:
        assert c.text.strip()


def test_section_metadata():
    pdf = Path(__file__).resolve().parents[1] / "data" / "raw" / "Privacy_Policy.pdf"
    tree = build_tree(pdf)
    assign_ids(tree)
    metas = section_metadata(tree)
    assert len(metas) == len(tree.sections)
    for m in metas:
        assert "heading" in m
        assert "summary" in m
        assert m["summary"].strip()


def test_all_raw_pdfs_parse():
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    for pdf in sorted(raw.glob("*.pdf")):
        tree = build_tree(pdf)
        assign_ids(tree)
        assert tree.sections, f"no sections for {pdf.name}"