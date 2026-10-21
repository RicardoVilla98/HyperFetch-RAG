"""
==============================================================================
HyperFetch-RAG: Document Ingestion Pipeline / Pipeline de Ingesta Documental
==============================================================================
Specialized in complex technical documents (PDFs with tables, figures, and manuals).
Implements semantic chunking, Markdown table conversion, OpenAI batch embeddings,
and Qdrant vector database ingestion with payload indexing.

Especializado en documentos técnicos complejos (PDFs con tablas, figuras y manuales).
Implementa segmentación semántica, conversión de tablas a Markdown, generación
por lotes de embeddings de OpenAI e inserción en la base vectorial Qdrant.
"""

import hashlib
import io
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pdfplumber
import pypdf
import tiktoken
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import Settings, get_settings


class ChunkMetadata(BaseModel):
    """
    Detailed metadata attached to each semantic chunk for citation and tracking.
    Metadatos detallados adjuntos a cada fragmento semántico para citación y trazabilidad.
    """
    chunk_id: str = Field(..., description="Unique UUID identifier for this chunk.")
    document_id: str = Field(..., description="Hash or UUID of the parent document.")
    filename: str = Field(..., description="Original filename of the ingested file.")
    page_number: int = Field(..., description="1-based page number where chunk originates.")
    chunk_index: int = Field(..., description="0-based sequential index of chunk within document.")
    has_table: bool = Field(default=False, description="Flag indicating if chunk contains tabular data.")
    has_image: bool = Field(default=False, description="Flag indicating presence of figures/images on page.")
    token_count: int = Field(..., description="Number of tokens calculated using tiktoken.")
    char_count: int = Field(..., description="Character length of chunk text.")
    content_hash: str = Field(..., description="SHA-256 hash of the chunk text content.")


class ProcessedChunk(BaseModel):
    """
    Complete chunk entity including text content and structured metadata.
    Entidad completa de fragmento incluyendo texto y metadatos estructurados.
    """
    id: str
    text: str
    metadata: ChunkMetadata
    dense_vector: Optional[List[float]] = None


class IngestionSummary(BaseModel):
    """
    Execution summary returned after successfully processing a document.
    Resumen de ejecución devuelto tras procesar exitosamente un documento.
    """
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    tables_detected: int
    images_detected: int
    processing_time_seconds: float
    status: str = "success"


class DocumentIngestionPipeline:
    """
    Production-grade pipeline for parsing technical PDFs, semantic chunking,
    embedding generation, and vector indexing into Qdrant.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or get_settings()
        
        # Initialize OpenAI Client
        self.openai_client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY.get_secret_value()
        )
        
        # Initialize Tokenizer (cl100k_base used by text-embedding-3-* and gpt-4o)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            logger.warning(f"Could not load cl100k_base directly, falling back to gpt-4o: {exc}")
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(
            host=self.settings.QDRANT_HOST,
            port=self.settings.QDRANT_PORT,
            api_key=self.settings.QDRANT_API_KEY.get_secret_value() if self.settings.QDRANT_API_KEY else None,
            timeout=30.0
        )
        
        # Ensure collection exists and is configured
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self) -> None:
        """
        Ensures the target Qdrant collection exists with Cosine distance and correct dimensions.
        Creates payload indexes for high-speed metadata filtering.
        """
        collection_name = self.settings.QDRANT_COLLECTION_NAME
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.settings.EMBEDDING_DIMENSION,
                        distance=qmodels.Distance.COSINE
                    )
                )
                
                # Create payload indexes on critical fields for filtering
                for field in ["document_id", "filename", "page_number", "has_table"]:
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD if field != "page_number" else qmodels.PayloadSchemaType.INTEGER
                    )
                logger.info(f"Collection {collection_name} created successfully with indexes.")
            else:
                logger.info(f"Qdrant collection {collection_name} already exists.")
        except Exception as exc:
            logger.error(f"Failed to connect or create Qdrant collection: {exc}")
            # Non-blocking during initialization in container orchestration
            pass

    def _convert_table_to_markdown(self, table_rows: List[List[Optional[str]]]) -> str:
        """
        Converts a 2D extracted table matrix into a clean, markdown-formatted table.
        Preserves row-column relationships for dense and sparse retrieval.
        Convierte una matriz de tabla extraída en una tabla formateada en Markdown.
        """
        if not table_rows or len(table_rows) < 1:
            return ""

        cleaned_rows = []
        for row in table_rows:
            cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cell != "" for cell in cleaned_row):
                cleaned_rows.append(cleaned_row)

        if not cleaned_rows:
            return ""

        # Normalize column counts
        max_cols = max(len(row) for row in cleaned_rows)
        for row in cleaned_rows:
            while len(row) < max_cols:
                row.append("")

        header = cleaned_rows[0]
        header_line = "| " + " | ".join(header) + " |"
        separator_line = "| " + " | ".join(["---"] * max_cols) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in cleaned_rows[1:]]

        return "\n".join([header_line, separator_line] + body_lines)

    def extract_pdf_content(self, file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Extracts structured text, tables (converted to Markdown), and image metadata
        page-by-page from technical PDFs using pdfplumber and pypdf.
        
        Extrae texto estructurado, tablas y metadatos de imágenes página por página.
        """
        pages_content: List[Dict[str, Any]] = []
        total_tables = 0
        total_images = 0

        # Primary extraction using pdfplumber for high-precision table detection
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_tables_md: List[str] = []
                has_page_table = False
                
                # Extract tables
                try:
                    extracted_tables = page.extract_tables()
                    if extracted_tables:
                        for table in extracted_tables:
                            table_md = self._convert_table_to_markdown(table)
                            if table_md:
                                page_tables_md.append(table_md)
                                total_tables += 1
                                has_page_table = True
                except Exception as table_err:
                    logger.debug(f"Page {page_idx} table extraction warning: {table_err}")

                # Detect embedded images / figures on page
                images_on_page = len(page.images) if hasattr(page, "images") else 0
                if images_on_page > 0:
                    total_images += images_on_page

                # Extract textual body
                raw_text = page.extract_text(layout=True) or ""
                
                # Append tables to page text body if present
                combined_page_text = raw_text
                if page_tables_md:
                    combined_page_text += "\n\n### Tablas / Datos Estructurados:\n" + "\n\n".join(page_tables_md)

                pages_content.append({
                    "page_number": page_idx,
                    "text": combined_page_text.strip(),
                    "has_table": has_page_table,
                    "has_image": images_on_page > 0,
                    "tables_count": len(page_tables_md),
                    "images_count": images_on_page
                })

        return pages_content, total_tables, total_images

    def semantic_split_text(self, text: str, page_number: int, filename: str, doc_id: str) -> List[ProcessedChunk]:
        """
        Splits raw page content into semantically coherent token chunks with overlap.
        Respects paragraph, header, and sentence boundaries to avoid splitting table cells.
        
        Divide el contenido en fragmentos coherentes respetando límites semánticos y tokens.
        """
        if not text.strip():
            return []

        chunks: List[ProcessedChunk] = []
        
        # Primary semantic boundaries: Headings, Markdown Tables, Double Newlines, Sentences
        paragraphs = text.split("\n\n")
        current_chunk_tokens: List[int] = []
        current_chunk_text = ""
        chunk_idx = 0

        for paragraph in paragraphs:
            para = paragraph.strip()
            if not para:
                continue

            para_tokens = self.tokenizer.encode(para)
            
            # If a single paragraph/table exceeds CHUNK_SIZE, sub-split it by sentences
            if len(para_tokens) > self.settings.CHUNK_SIZE:
                sentences = para.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    sent = sentence.strip()
                    if not sent:
                        continue
                    sent_tokens = self.tokenizer.encode(sent)
                    
                    if len(current_chunk_tokens) + len(sent_tokens) <= self.settings.CHUNK_SIZE:
                        current_chunk_tokens.extend(sent_tokens)
                        current_chunk_text += (" " if current_chunk_text else "") + sent
                    else:
                        if current_chunk_text:
                            # Finalize chunk
                            chunk_id = str(uuid.uuid4())
                            meta = ChunkMetadata(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                filename=filename,
                                page_number=page_number,
                                chunk_index=chunk_idx,
                                has_table="| " in current_chunk_text and "---" in current_chunk_text,
                                has_image=False,
                                token_count=len(current_chunk_tokens),
                                char_count=len(current_chunk_text),
                                content_hash=hashlib.sha256(current_chunk_text.encode("utf-8")).hexdigest()
                            )
                            chunks.append(ProcessedChunk(id=chunk_id, text=current_chunk_text, metadata=meta))
                            chunk_idx += 1

                        # Preserve overlap tokens
                        overlap_tokens = current_chunk_tokens[-self.settings.CHUNK_OVERLAP:] if current_chunk_tokens else []
                        overlap_text = self.tokenizer.decode(overlap_tokens) if overlap_tokens else ""
                        
                        current_chunk_tokens = overlap_tokens + sent_tokens
                        current_chunk_text = (overlap_text + " " + sent).strip()
            else:
                # Normal paragraph accumulation
                if len(current_chunk_tokens) + len(para_tokens) <= self.settings.CHUNK_SIZE:
                    current_chunk_tokens.extend(para_tokens)
                    current_chunk_text += ("\n\n" if current_chunk_text else "") + para
                else:
                    if current_chunk_text:
                        chunk_id = str(uuid.uuid4())
                        meta = ChunkMetadata(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            filename=filename,
                            page_number=page_number,
                            chunk_index=chunk_idx,
                            has_table="| " in current_chunk_text and "---" in current_chunk_text,
                            has_image=False,
                            token_count=len(current_chunk_tokens),
                            char_count=len(current_chunk_text),
                            content_hash=hashlib.sha256(current_chunk_text.encode("utf-8")).hexdigest()
                        )
                        chunks.append(ProcessedChunk(id=chunk_id, text=current_chunk_text, metadata=meta))
                        chunk_idx += 1

                    overlap_tokens = current_chunk_tokens[-self.settings.CHUNK_OVERLAP:] if current_chunk_tokens else []
                    overlap_text = self.tokenizer.decode(overlap_tokens) if overlap_tokens else ""
                    
                    current_chunk_tokens = overlap_tokens + para_tokens
                    current_chunk_text = (overlap_text + "\n\n" + para).strip()

        # Flush trailing chunk
        if current_chunk_text.strip():
            chunk_id = str(uuid.uuid4())
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=filename,
                page_number=page_number,
                chunk_index=chunk_idx,
                has_table="| " in current_chunk_text and "---" in current_chunk_text,
                has_image=False,
                token_count=len(self.tokenizer.encode(current_chunk_text)),
                char_count=len(current_chunk_text),
                content_hash=hashlib.sha256(current_chunk_text.encode("utf-8")).hexdigest()
            )
            chunks.append(ProcessedChunk(id=chunk_id, text=current_chunk_text, metadata=meta))

        return chunks

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generates dense vector embeddings using OpenAI API in optimized batches.
        Genera vectores de embeddings densos utilizando la API de OpenAI en lotes.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Sanitize newlines as recommended by OpenAI embedding guidelines
            clean_batch = [text.replace("\n", " ").strip() for text in batch]
            
            try:
                response = self.openai_client.embeddings.create(
                    input=clean_batch,
                    model=self.settings.OPENAI_EMBEDDING_MODEL
                )
                batch_vectors = [item.embedding for item in response.data]
                all_embeddings.extend(batch_vectors)
            except Exception as exc:
                logger.error(f"Failed to generate embeddings for batch {i // batch_size}: {exc}")
                raise exc

        return all_embeddings

    def ingest_document(self, file_bytes: bytes, filename: str) -> IngestionSummary:
        """
        End-to-end ingestion pipeline:
        1. Generates cryptographic Document ID.
        2. Parses PDF or text, extracting tables and structure.
        3. Executes semantic token chunking.
        4. Computes batch dense embeddings.
        5. Upserts points into Qdrant vector database.
        
        Pipeline de ingesta integral: extracción, partición, embedding y carga en Qdrant.
        """
        start_time = time.time()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        doc_id = f"doc_{file_hash[:16]}"
        logger.info(f"Initiating ingestion for {filename} (ID: {doc_id})")

        # Determine file type
        file_suffix = Path(filename).suffix.lower()
        
        if file_suffix == ".pdf":
            pages_data, total_tables, total_images = self.extract_pdf_content(file_bytes, filename)
        else:
            # Plain text or Markdown document
            raw_content = file_bytes.decode("utf-8", errors="replace")
            pages_data = [{
                "page_number": 1,
                "text": raw_content,
                "has_table": "| " in raw_content and "---" in raw_content,
                "has_image": False,
                "tables_count": 0,
                "images_count": 0
            }]
            total_tables = 1 if pages_data[0]["has_table"] else 0
            total_images = 0

        # Perform semantic chunking across all pages
        all_chunks: List[ProcessedChunk] = []
        for page_info in pages_data:
            page_chunks = self.semantic_split_text(
                text=page_info["text"],
                page_number=page_info["page_number"],
                filename=filename,
                doc_id=doc_id
            )
            # Update image metadata flag from page level if applicable
            for chunk in page_chunks:
                if page_info["has_image"]:
                    chunk.metadata.has_image = True
            all_chunks.extend(page_chunks)

        if not all_chunks:
            logger.warning(f"No extractable text chunks found in document: {filename}")
            return IngestionSummary(
                document_id=doc_id,
                filename=filename,
                total_pages=len(pages_data),
                total_chunks=0,
                tables_detected=total_tables,
                images_detected=total_images,
                processing_time_seconds=round(time.time() - start_time, 3),
                status="empty_document"
            )

        logger.info(f"Generated {len(all_chunks)} semantic chunks for {filename}. Generating embeddings...")

        # Generate Dense Embeddings
        chunk_texts = [c.text for c in all_chunks]
        embeddings = self.generate_embeddings_batch(chunk_texts, batch_size=64)

        for chunk, vector in zip(all_chunks, embeddings):
            chunk.dense_vector = vector

        # Build Qdrant PointStruct instances
        qdrant_points: List[qmodels.PointStruct] = []
        for chunk in all_chunks:
            point = qmodels.PointStruct(
                id=chunk.id,
                vector=chunk.dense_vector,
                payload={
                    "text": chunk.text,
                    "document_id": chunk.metadata.document_id,
                    "filename": chunk.metadata.filename,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "has_table": chunk.metadata.has_table,
                    "has_image": chunk.metadata.has_image,
                    "token_count": chunk.metadata.token_count,
                    "char_count": chunk.metadata.char_count,
                    "content_hash": chunk.metadata.content_hash,
                }
            )
            qdrant_points.append(point)

        # Batch upsert to Qdrant
        logger.info(f"Upserting {len(qdrant_points)} points to Qdrant collection: {self.settings.QDRANT_COLLECTION_NAME}")
        self.qdrant_client.upsert(
            collection_name=self.settings.QDRANT_COLLECTION_NAME,
            points=qdrant_points,
            wait=True
        )

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Ingestion completed for {filename} in {elapsed}s.")

        return IngestionSummary(
            document_id=doc_id,
            filename=filename,
            total_pages=len(pages_data),
            total_chunks=len(all_chunks),
            tables_detected=total_tables,
            images_detected=total_images,
            processing_time_seconds=elapsed,
            status="success"
        )
