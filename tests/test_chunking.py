from __future__ import annotations

from nttv_chatbot.chunking import ChunkingSettings, build_chunks
from nttv_chatbot.document_parsing import StructuredDocument, StructuredDocumentElement


def _element(
    *,
    text: str,
    element_type: str = "paragraph",
    heading_path: list[str] | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    file_type: str = "txt",
    parser: str = "plain_text",
) -> StructuredDocumentElement:
    return StructuredDocumentElement(
        source_file="data/source.txt",
        file_type=file_type,
        parser=parser,
        page_start=page_start,
        page_end=page_end,
        heading_path=heading_path or [],
        element_type=element_type,
        text=text,
        raw_metadata={"origin": "test"},
    )


def test_section_aware_chunking_keeps_heading_with_section_content():
    document = StructuredDocument(
        parser_name="plain_text",
        source_path="data/nttv rank requirements.txt",
        file_type="txt",
        elements=[
            _element(text="8th Kyu", element_type="heading", heading_path=["8th Kyu"]),
            _element(
                text="Ukemi drills and striking combinations for 8th kyu.",
                heading_path=["8th Kyu"],
            ),
            _element(text="9th Kyu", element_type="heading", heading_path=["9th Kyu"]),
            _element(
                text="Kamae review and foundational movement for 9th kyu.",
                heading_path=["9th Kyu"],
            ),
        ],
    )

    chunks = build_chunks(
        document,
        settings=ChunkingSettings(target_tokens=80, max_tokens=120, overlap_tokens=10, min_tokens=20),
    )

    assert len(chunks) == 2
    assert chunks[0]["section_title"] == "8th Kyu"
    assert chunks[0]["heading_path"] == ["8th Kyu"]
    assert chunks[0]["text"].startswith("8th Kyu")
    assert "9th Kyu" not in chunks[0]["text"]
    assert chunks[0]["rank_tag"] == "8th Kyu"


def test_overlap_behavior_repeats_tail_context_in_next_chunk():
    paragraph_one = "Paragraph one marker " + " ".join(["alpha"] * 24)
    paragraph_two = "Paragraph two marker " + " ".join(["beta"] * 24)
    paragraph_three = "Paragraph three marker " + " ".join(["gamma"] * 24)

    document = StructuredDocument(
        parser_name="plain_text",
        source_path="data/notes.txt",
        file_type="txt",
        elements=[
            _element(
                text=f"{paragraph_one}\n\n{paragraph_two}\n\n{paragraph_three}",
                heading_path=[],
            )
        ],
    )

    chunks = build_chunks(
        document,
        settings=ChunkingSettings(target_tokens=35, max_tokens=55, overlap_tokens=10, min_tokens=10),
    )

    assert len(chunks) >= 2
    assert "Paragraph one marker" in chunks[0]["text"]
    assert "Paragraph one marker" in chunks[1]["text"]


def test_metadata_propagation_from_parser_output():
    document = StructuredDocument(
        parser_name="docling",
        source_path="data/NTTV Weapons Reference.pdf",
        file_type="pdf",
        elements=[
            _element(
                text="Weapons",
                element_type="heading",
                heading_path=["Weapons"],
                page_start=2,
                page_end=2,
                file_type="pdf",
                parser="docling",
            ),
            _element(
                text="Hanbo profile, history, and usage notes.",
                heading_path=["Weapons", "Hanbo"],
                page_start=2,
                page_end=3,
                file_type="pdf",
                parser="docling",
            ),
        ],
    )

    chunks = build_chunks(
        document,
        source_file="data/NTTV Weapons Reference.pdf",
        settings=ChunkingSettings(target_tokens=80, max_tokens=120, overlap_tokens=10, min_tokens=20),
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    meta = chunk["meta"]

    assert chunk["source_file"] == "data/NTTV Weapons Reference.pdf"
    assert meta["source_file"] == "data/NTTV Weapons Reference.pdf"
    assert meta["file_type"] == "pdf"
    assert meta["parser"] == "docling"
    assert meta["page_start"] == 2
    assert meta["page_end"] == 3
    assert meta["heading_path"] == ["Weapons", "Hanbo"]
    assert meta["section_title"] == "Hanbo"
    assert meta["weapon_tag"] == "Hanbo"
    assert meta["content_type"] == "weapon_profile"
    assert meta["priority_bucket"] == "p3"


def test_no_empty_chunks_are_emitted():
    document = StructuredDocument(
        parser_name="plain_text",
        source_path="data/blank.txt",
        file_type="txt",
        elements=[
            _element(text="   \n\n   "),
            _element(text="Useful material lives here."),
        ],
    )

    chunks = build_chunks(
        document,
        settings=ChunkingSettings(target_tokens=80, max_tokens=120, overlap_tokens=0, min_tokens=10),
    )

    assert chunks
    assert all(chunk["text"].strip() for chunk in chunks)


def test_chunk_ids_are_stable_for_same_input():
    document = StructuredDocument(
        parser_name="plain_text",
        source_path="data/Technique Descriptions.md",
        file_type="md",
        elements=[
            _element(text="Omote Gyaku", element_type="heading", heading_path=["Omote Gyaku"]),
            _element(
                text="Translation: Outside reverse.\n\nDefinition: Wrist reversal.",
                heading_path=["Omote Gyaku"],
            ),
        ],
    )

    settings = ChunkingSettings(target_tokens=80, max_tokens=120, overlap_tokens=10, min_tokens=10)

    first = build_chunks(document, settings=settings)
    second = build_chunks(document, settings=settings)

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)
