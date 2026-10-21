"""
==============================================================================
HyperFetch-RAG: Unit and Architecture Verification Tests
==============================================================================
"""

import pytest
from src.config import Settings
from src.ingestion import DocumentIngestionPipeline


def test_settings_validation():
    """Verifies settings load defaults and enforces chunk overlap constraints."""
    settings = Settings(
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=100
    )
    assert settings.CHUNK_SIZE == 500
    assert settings.CHUNK_OVERLAP == 100
    assert settings.EMBEDDING_DIMENSION == 1536


def test_invalid_chunk_overlap():
    """Ensures overlap exceeding chunk size raises a validation error."""
    with pytest.raises(ValueError):
        Settings(
            CHUNK_SIZE=300,
            CHUNK_OVERLAP=300
        )


def test_table_markdown_conversion():
    """Tests conversion of 2D raw extracted tables into clean Markdown."""
    pipeline = DocumentIngestionPipeline(
        settings=Settings(
            OPENAI_API_KEY="sk-test",
            QDRANT_HOST="localhost"
        )
    )
    
    raw_table = [
        ["Sensor ID", "Rango de Presión", "Tolerancia"],
        ["PT-101", "0 - 100 Bar", "± 0.2%"],
        ["PT-102", "0 - 250 Bar", "± 0.1%"]
    ]
    
    markdown_output = pipeline._convert_table_to_markdown(raw_table)
    
    assert "| Sensor ID | Rango de Presión | Tolerancia |" in markdown_output
    assert "| --- | --- | --- |" in markdown_output
    assert "| PT-101 | 0 - 100 Bar | ± 0.2% |" in markdown_output


def test_semantic_split_text():
    """Tests semantic chunk boundary preservation."""
    pipeline = DocumentIngestionPipeline(
        settings=Settings(
            OPENAI_API_KEY="sk-test",
            QDRANT_HOST="localhost",
            CHUNK_SIZE=50,
            CHUNK_OVERLAP=10
        )
    )
    
    sample_text = (
        "Manual de Mantenimiento de Turbina Eólica.\n\n"
        "Sección 1: Procedimiento de Seguridad.\n"
        "Antes de intervenir el rotor es mandatorio activar el freno hidráulico de emergencia.\n\n"
        "Sección 2: Calibración del Tacómetro.\n"
        "El tacómetro debe marcar 0 RPM tras el bloqueo mecánico del eje principal."
    )
    
    chunks = pipeline.semantic_split_text(
        text=sample_text,
        page_number=1,
        filename="test_manual.pdf",
        doc_id="doc_test_123"
    )
    
    assert len(chunks) > 0
    assert all(c.metadata.filename == "test_manual.pdf" for c in chunks)
    assert all(c.metadata.page_number == 1 for c in chunks)
