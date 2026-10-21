"""
==============================================================================
HyperFetch-RAG: Central Configuration Module / Módulo de Configuración Central
==============================================================================
Manages environment variables, default configurations, and runtime settings
using Pydantic Settings (Pydantic v2).
Gestiona variables de entorno, configuraciones por defecto y parámetros de
ejecución utilizando Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings schema with strict environment parsing and validation.
    Esquema de configuración de la aplicación con validación estricta de tipos.
    """

    # --- Application Metadata ---
    APP_NAME: str = Field(
        default="HyperFetch-RAG",
        description="Application identifier / Nombre identificador de la aplicación."
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application release version / Versión de lanzamiento de la aplicación."
    )
    APP_ENV: str = Field(
        default="production",
        description="Runtime environment: 'development', 'staging', or 'production'."
    )
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Network host interface to bind the API server."
    )
    API_PORT: int = Field(
        default=8000,
        description="Port exposed by the API server."
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )

    # --- OpenAI Configuration ---
    OPENAI_API_KEY: SecretStr = Field(
        default=SecretStr("sk-dummy-key-for-development-change-in-production"),
        description="Secret API key for OpenAI LLM and embeddings."
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o",
        description="OpenAI LLM model for grounded answer synthesis."
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for dense representation."
    )
    EMBEDDING_DIMENSION: int = Field(
        default=1536,
        description="Dimensionality of the dense embedding vectors (1536 for text-embedding-3-small)."
    )

    # --- Qdrant Vector Database ---
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Host address of the Qdrant vector database."
    )
    QDRANT_PORT: int = Field(
        default=6333,
        description="HTTP REST port for Qdrant."
    )
    QDRANT_GRPC_PORT: int = Field(
        default=6334,
        description="gRPC port for high-throughput vector ingestion."
    )
    QDRANT_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Optional API key for cloud-hosted or authenticated Qdrant cluster."
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="hyperfetch_knowledge_base",
        description="Target Qdrant collection name for vector indexing."
    )

    # --- Cross-Encoder Re-Ranking ---
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model identifier for the cross-encoder reranker."
    )
    USE_RERANKER: bool = Field(
        default=True,
        description="Flag to enable or bypass the cross-encoder reranking stage."
    )

    # --- Document Ingestion & Chunking ---
    CHUNK_SIZE: int = Field(
        default=800,
        description="Maximum semantic token chunk size for document splitting."
    )
    CHUNK_OVERLAP: int = Field(
        default=150,
        description="Token overlap between consecutive chunks to preserve contextual continuity."
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        description="Maximum allowed uploaded file size in megabytes."
    )

    # --- Hybrid Retrieval & Scoring ---
    TOP_K_DENSE: int = Field(
        default=15,
        description="Number of candidate passages retrieved via dense vector similarity."
    )
    TOP_K_SPARSE: int = Field(
        default=15,
        description="Number of candidate passages retrieved via BM25 lexical search."
    )
    TOP_K_FINAL: int = Field(
        default=5,
        description="Final number of top passages passed to the LLM context window after reranking."
    )
    HYBRID_ALPHA: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight balance for Hybrid Fusion: 1.0 = Pure Dense, 0.0 = Pure Sparse BM25."
    )
    RECIPROCAL_RANK_K: int = Field(
        default=60,
        description="Smoothing constant 'k' for Reciprocal Rank Fusion (RRF)."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, overlap_val: int, info) -> int:
        """
        Ensures chunk overlap is strictly smaller than chunk size.
        Verifica que el solapamiento de chunks sea menor que el tamaño del chunk.
        """
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if overlap_val >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({overlap_val}) must be strictly smaller than CHUNK_SIZE ({chunk_size})"
            )
        return overlap_val

    def get_masked_openai_key(self) -> str:
        """
        Returns a masked representation of the OpenAI API key for safe telemetry.
        Retorna la clave API ofuscada para registros y telemetría segura.
        """
        raw_key = self.OPENAI_API_KEY.get_secret_value()
        if not raw_key or len(raw_key) < 8:
            return "********"
        return f"{raw_key[:4]}...{raw_key[-4:]}"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached accessor returning the global singleton Settings instance.
    Retorna la instancia única de configuración cacheada.
    """
    return Settings()
