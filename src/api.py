"""
==============================================================================
HyperFetch-RAG: FastAPI REST Service / Servicio REST de API de Alta Disponibilidad
==============================================================================
Provides high-throughput endpoints for document ingestion, multimodal technical
parsing, hybrid retrieval, and grounded Q&A with microsecond latency metrics.

Proporciona endpoints de alto rendimiento para ingesta de documentos, análisis
técnico multimodal, recuperación híbrida y preguntas y respuestas fundamentadas.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.ingestion import DocumentIngestionPipeline, IngestionSummary
from src.rag_engine import HybridRAGEngine, RAGQueryResponse

# Global singletons
ingestion_pipeline: Optional[DocumentIngestionPipeline] = None
rag_engine: Optional[HybridRAGEngine] = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager: initializes models, Qdrant schemas,
    and warmups on startup; frees resources on shutdown.
    """
    global ingestion_pipeline, rag_engine, start_time
    start_time = time.time()
    settings = get_settings()
    logger.info(f"Initializing {settings.APP_NAME} v{settings.APP_VERSION} in {settings.APP_ENV} mode...")

    try:
        ingestion_pipeline = DocumentIngestionPipeline(settings=settings)
        rag_engine = HybridRAGEngine(settings=settings)
        # Pre-warm sparse index
        rag_engine.sync_sparse_index(force_refresh=True)
        logger.info("HyperFetch-RAG engine and ingestion services initialized successfully.")
    except Exception as exc:
        logger.error(f"Startup initialization warning (services will retry on demand): {exc}")

    yield

    logger.info("Gracefully shutting down HyperFetch-RAG services...")


# Initialize FastAPI instance
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Multimodal Knowledge Engine with Hybrid Dense-Sparse Retrieval, "
        "Cross-Encoder Re-ranking, and Anti-Hallucination Grounding."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend and enterprise portal integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Models ---

class QueryRequest(BaseModel):
    """
    Payload for /query endpoint.
    Cuerpo de la petición para realizar consultas al motor RAG.
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User question regarding technical documents or manuals.",
        examples=["¿Cuál es el procedimiento de calibración del sensor de presión en la página 14?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of final cited passages to return (defaults to configuration TOP_K_FINAL)."
    )
    use_reranker: Optional[bool] = Field(
        default=True,
        description="Flag to toggle neural Cross-Encoder re-ranking stage."
    )
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Hybrid balance weight (1.0 = Pure Dense, 0.0 = Pure BM25, 0.5 = Balanced)."
    )


class HealthCheckResponse(BaseModel):
    """
    Payload for /health endpoint.
    Estado de salud y conectividad de la plataforma.
    """
    status: str
    app_name: str
    version: str
    uptime_seconds: float
    qdrant_connected: bool
    reranker_active: bool
    openai_configured: bool


# --- API Routes ---

@app.get("/health", response_model=HealthCheckResponse, tags=["Observability"])
async def health_check() -> HealthCheckResponse:
    """
    Verifies system health, Qdrant cluster connectivity, and AI model readiness.
    Comprueba el estado del sistema, la conexión a Qdrant y los modelos de IA.
    """
    global rag_engine, start_time
    current_settings = get_settings()

    qdrant_ok = False
    if rag_engine and rag_engine.qdrant_client:
        try:
            rag_engine.qdrant_client.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False

    reranker_ready = bool(rag_engine and rag_engine.reranker is not None)
    openai_ok = bool(current_settings.OPENAI_API_KEY.get_secret_value())

    overall_status = "healthy" if (qdrant_ok and openai_ok) else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        app_name=current_settings.APP_NAME,
        version=current_settings.APP_VERSION,
        uptime_seconds=round(time.time() - start_time, 2),
        qdrant_connected=qdrant_ok,
        reranker_active=reranker_ready,
        openai_configured=openai_ok
    )


@app.post(
    "/upload-doc",
    response_model=IngestionSummary,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"]
)
async def upload_document(
    file: UploadFile = File(..., description="Technical document file (PDF, TXT, or Markdown).")
) -> IngestionSummary:
    """
    Uploads, parses, chunks, and indexes a complex technical document into Qdrant.
    Extracts tabular data into Markdown and detects page figures.
    
    Carga, parsea, segmenta e indexa un documento técnico complejo en Qdrant.
    """
    global ingestion_pipeline, rag_engine
    if not ingestion_pipeline:
        ingestion_pipeline = DocumentIngestionPipeline()

    # Validate file extension
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    filename = file.filename or "uploaded_document"
    _, ext = os.path.splitext(filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato de archivo no soportado '{ext}'. Formatos admitidos: {list(allowed_extensions)}"
        )

    # Read binary contents
    try:
        content = await file.read()
    except Exception as exc:
        logger.error(f"Failed to read incoming file stream: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al leer los datos del archivo subido."
        )

    # Check size limit
    max_bytes = get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo excede el tamaño máximo permitido de {get_settings().MAX_UPLOAD_SIZE_MB}MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo proporcionado está vacío (0 bytes)."
        )

    try:
        # Run ingestion pipeline
        summary = ingestion_pipeline.ingest_document(file_bytes=content, filename=filename)
        
        # Invalidate / refresh BM25 cache in rag_engine so new document is immediately retrievable
        if rag_engine:
            rag_engine.sync_sparse_index(force_refresh=True)

        return summary
    except Exception as exc:
        logger.error(f"Ingestion pipeline failure for {filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo durante el procesamiento del documento: {str(exc)}"
        )


@app.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Engine"]
)
async def query_rag(request: QueryRequest) -> RAGQueryResponse:
    """
    Executes hybrid retrieval (Dense + Sparse), Cross-Encoder re-ranking,
    and hallucination-free generation with source citations.
    
    Ejecuta recuperación híbrida, re-ordenamiento neuronal y generación con citas.
    """
    global rag_engine
    if not rag_engine:
        rag_engine = HybridRAGEngine()

    try:
        response = rag_engine.execute_query(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            alpha=request.alpha
        )
        return response
    except Exception as exc:
        logger.error(f"Query processing failure for '{request.query}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento de la consulta RAG: {str(exc)}"
        )


# Direct execution entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=(settings.APP_ENV == "development")
    )
