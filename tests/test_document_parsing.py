from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import ingest
from nttv_chatbot import document_parsing
from nttv_chatbot.document_parsing import ParserSettings, StructuredDocument, StructuredDocumentElement


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_parse_file_normalizes_text_sources(tmp_path: Path, suffix: str):
    source = tmp_path / f"sample{suffix}"
    source.write_text("Line one\n\nLine two", encoding="utf-8")

    parsed = document_parsing.parse_file(source)

    assert parsed.parser_name == "plain_text"
    assert parsed.source_path == str(source)
    assert parsed.file_type == suffix.lstrip(".")
    assert len(parsed.elements) == 1

    element = parsed.elements[0]
    assert element.source_file == str(source)
    assert element.file_type == suffix.lstrip(".")
    assert element.parser == "plain_text"
    assert element.page_start is None
    assert element.page_end is None
    assert element.heading_path == []
    assert element.element_type == "document"
    assert element.text == "Line one\n\nLine two"
    assert element.raw_metadata["character_count"] == len("Line one\n\nLine two")


@dataclass
class _FakeProv:
    page_no: int


class _FakeTextItem:
    def __init__(self, text: str, label: str, page_no: int, self_ref: str):
        self.text = text
        self.label = label
        self.prov = [_FakeProv(page_no=page_no)]
        self.self_ref = self_ref


class _FakeTableItem:
    label = "table"

    def __init__(self, page_no: int, self_ref: str):
        self.prov = [_FakeProv(page_no=page_no)]
        self.self_ref = self_ref

    def export_to_markdown(self, doc=None):
        return "| Item | Detail |\n| --- | --- |\n| Ukemi | Rolling |\n"


class _FakeDoclingDocument:
    num_pages = 2

    def iterate_items(self):
        yield _FakeTextItem("NTTV Guide", "title", 1, "#/texts/1"), 1
        yield _FakeTextItem("8th Kyu", "section_header", 1, "#/texts/2"), 2
        yield _FakeTextItem("Rolling basics", "text", 1, "#/texts/3"), 3
        yield _FakeTableItem(2, "#/tables/1"), 3


class _FakeConversionResult:
    def __init__(self):
        self.document = _FakeDoclingDocument()
        self.status = "SUCCESS"


class _FakeDoclingConverter:
    def convert(self, source, **kwargs):
        assert str(source).endswith(".pdf")
        assert kwargs["max_num_pages"] == 3
        return _FakeConversionResult()


def test_docling_pdf_parser_normalizes_mocked_output(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    monkeypatch.setattr(document_parsing, "_build_docling_converter", lambda: _FakeDoclingConverter())

    settings = ParserSettings(
        enable_pdf_ingest=True,
        pdf_parser="docling",
        pdf_parse_max_pages=3,
        pdf_fail_open=True,
        docling_available=True,
    )

    parsed = document_parsing.parse_file(pdf_path, settings=settings)

    assert parsed.parser_name == "docling"
    assert parsed.file_type == "pdf"
    assert parsed.raw_metadata["page_count"] == 2
    assert parsed.raw_metadata["conversion_status"] == "SUCCESS"
    assert len(parsed.elements) == 4

    paragraph = parsed.elements[2]
    assert paragraph.page_start == 1
    assert paragraph.page_end == 1
    assert paragraph.heading_path == ["NTTV Guide", "8th Kyu"]
    assert paragraph.element_type == "paragraph"
    assert paragraph.text == "Rolling basics"

    table = parsed.elements[3]
    assert table.page_start == 2
    assert table.page_end == 2
    assert table.heading_path == ["NTTV Guide", "8th Kyu"]
    assert table.element_type == "table"
    assert "Ukemi" in table.text


def test_ingest_continues_when_pdf_parse_fails(monkeypatch, tmp_path: Path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Rank notes\n\nBasics", encoding="utf-8")

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 broken")

    def fake_parse_file(path, settings=None):
        file_path = Path(path)
        if file_path.suffix.lower() == ".pdf":
            raise RuntimeError("Docling parse exploded")
        return StructuredDocument(
            parser_name="plain_text",
            source_path=str(file_path),
            file_type="txt",
            elements=[
                StructuredDocumentElement(
                    source_file=str(file_path),
                    file_type="txt",
                    parser="plain_text",
                    page_start=None,
                    page_end=None,
                    heading_path=[],
                    element_type="document",
                    text="Rank notes\n\nBasics",
                    raw_metadata={"character_count": 18},
                )
            ],
            raw_metadata={"character_count": 18},
        )

    monkeypatch.setattr(ingest, "parse_file", fake_parse_file)

    settings = ParserSettings(
        enable_pdf_ingest=True,
        pdf_parser="docling",
        pdf_parse_max_pages=None,
        pdf_fail_open=True,
        docling_available=True,
    )

    chunks, ingested_files, skipped_files = ingest.parse_and_chunk_files(
        [text_path, pdf_path],
        root=tmp_path,
        parser_settings=settings,
    )

    assert ingested_files == ["notes.txt"]
    assert len(chunks) == 1
    assert chunks[0]["source"] == "notes.txt"
    assert chunks[0]["meta"]["parser"] == "plain_text"
    assert skipped_files == [{"path": "broken.pdf", "reason": "Docling parse exploded"}]
