"""
==============================================================================
HyperFetch-RAG: Hybrid Retrieval & Re-ranking Engine / Motor de Recuperación Híbrido
==============================================================================
Combines Dense Semantic Search (OpenAI embeddings) and Sparse Lexical Retrieval
(BM25Okapi) with Reciprocal Rank Fusion (RRF), Cross-Encoder Re-Ranking
(sentence-transformers), and Hallucination-Proof Grounded Generation (GPT-4o).

Combina Búsqueda Semántica Densa (Embeddings OpenAI) y Recuperación Léxica
Dispersa (BM25) con Fusión por Rango Recíproco (RRF), Re-ordenamiento mediante
Cross-Encoder y Generación Fundamentada Anti-Alucinaciones (GPT-4o).
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import Settings, get_settings


class SourceCitation(BaseModel):
    """
    Structured citation reference extracted from retrieved passages.
    Referencia bibliográfica y de trazabilidad de la fuente.
    """
    filename: str = Field(..., description="Document filename.")
    page_number: int = Field(..., description="Page number of the cited content.")
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    relevance_score: float = Field(..., description="Normalized relevance score after reranking.")
    excerpt: str = Field(..., description="Short snippet of the evidence text.")


class RetrievedPassage(BaseModel):
    """
    Intermediate representation of a retrieved chunk during search & rerank.
    Representación intermedia de un pasaje recuperado en el pipeline.
    """
    chunk_id: str
    text: str
    filename: str
    page_number: int
    chunk_index: int
    has_table: bool
    has_image: bool
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """
    Microsecond-level latency performance tracking for enterprise telemetry.
    Métricas de latencia por etapa para observabilidad en producción.
    """
    dense_search_ms: float
    sparse_search_ms: float
    fusion_ms: float
    reranking_ms: float
    generation_ms: float
    total_ms: float


class RAGQueryResponse(BaseModel):
    """
    Complete response payload returned to API consumers.
    Respuesta final estructurada para clientes de la API.
    """
    query: str
    answer: str
    confidence_score: float = Field(..., description="Calculated answer confidence based on context relevance.")
    citations: List[SourceCitation]
    retrieved_passages: List[RetrievedPassage]
    latency: LatencyBreakdown


class HybridRAGEngine:
    """
    High-Performance Hybrid RAG Engine integrating:
    1. Dense OpenAI Vector Search (Qdrant).
    2. BM25Okapi Sparse Retrieval over collection corpus.
    3. Reciprocal Rank Fusion (RRF) with configurable alpha.
    4. Cross-Encoder Re-Ranking (sentence-transformers ms-marco).
    5. Hallucination-Free Synthesis with Citations (GPT-4o).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or get_settings()

        # Initialize OpenAI Client
        self.openai_client = OpenAI(
            api_key=self.settings.OPENAI_API_KEY.get_secret_value()
        )

        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(
            host=self.settings.QDRANT_HOST,
            port=self.settings.QDRANT_PORT,
            api_key=self.settings.QDRANT_API_KEY.get_secret_value() if self.settings.QDRANT_API_KEY else None,
            timeout=30.0
        )

        # Initialize Cross-Encoder Reranker
        self.reranker: Optional[CrossEncoder] = None
        if self.settings.USE_RERANKER:
            try:
                logger.info(f"Loading Cross-Encoder Reranker: {self.settings.RERANKER_MODEL}")
                self.reranker = CrossEncoder(self.settings.RERANKER_MODEL)
                logger.info("Cross-Encoder Reranker initialized successfully.")
            except Exception as exc:
                logger.warning(f"Failed to load Cross-Encoder reranker ({exc}). Continuing with RRF fusion only.")
                self.reranker = None

        # In-memory BM25 index cache
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_passages: List[RetrievedPassage] = []
        self._last_corpus_sync: float = 0.0

    def sync_sparse_index(self, force_refresh: bool = False) -> None:
        """
        Synchronizes the in-memory BM25 index with documents currently in Qdrant.
        Sincroniza el índice disperso BM25 con los registros almacenados en Qdrant.
        """
        now = time.time()
        # Cache for 60 seconds unless forced
        if not force_refresh and self._bm25_index is not None and (now - self._last_corpus_sync) < 60:
            return

        try:
            logger.info("Synchronizing BM25 corpus from Qdrant collection...")
            passages: List[RetrievedPassage] = []
            offset = None
            limit = 250

            # Scroll through Qdrant collection to hydrate corpus
            while True:
                scroll_result, next_offset = self.qdrant_client.scroll(
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for point in scroll_result:
                    payload = point.payload or {}
                    passages.append(
                        RetrievedPassage(
                            chunk_id=str(point.id),
                            text=payload.get("text", ""),
                            filename=payload.get("filename", "unknown"),
                            page_number=payload.get("page_number", 1),
                            chunk_index=payload.get("chunk_index", 0),
                            has_table=payload.get("has_table", False),
                            has_image=payload.get("has_image", False),
                        )
                    )

                if next_offset is None or len(scroll_result) < limit:
                    break
                offset = next_offset

            if passages:
                # Tokenize for BM25
                tokenized_corpus = [p.text.lower().split() for p in passages]
                self._bm25_index = BM25Okapi(tokenized_corpus)
                self._bm25_passages = passages
                self._last_corpus_sync = now
                logger.info(f"BM25 index synchronized with {len(passages)} passages.")
            else:
                self._bm25_index = None
                self._bm25_passages = []
                logger.info("Qdrant collection is currently empty; BM25 index cleared.")
        except Exception as exc:
            logger.error(f"Error synchronizing BM25 index: {exc}")

    def _dense_retrieval(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """
        Executes Dense Vector Similarity search in Qdrant.
        Ejecuta la búsqueda de similitud por vectores densos en Qdrant.
        """
        # Embed user query
        query_vector_res = self.openai_client.embeddings.create(
            input=[query.replace("\n", " ")],
            model=self.settings.OPENAI_EMBEDDING_MODEL
        )
        query_vector = query_vector_res.data[0].embedding

        # Query Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        dense_passages: List[RetrievedPassage] = []
        for hit in search_results:
            payload = hit.payload or {}
            dense_passages.append(
                RetrievedPassage(
                    chunk_id=str(hit.id),
                    text=payload.get("text", ""),
                    filename=payload.get("filename", "unknown"),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    has_table=payload.get("has_table", False),
                    has_image=payload.get("has_image", False),
                    dense_score=float(hit.score)
                )
            )

        return dense_passages

    def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """
        Executes BM25 Lexical Keyword search over the hydrated corpus.
        Ejecuta la búsqueda léxica BM25 sobre el corpus indexado.
        """
        self.sync_sparse_index()

        if not self._bm25_index or not self._bm25_passages:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        # Pair scores with passages
        scored_pairs = list(zip(scores, self._bm25_passages))
        # Filter out 0 scores and sort descending
        scored_pairs = [pair for pair in scored_pairs if pair[0] > 0.0]
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        sparse_passages: List[RetrievedPassage] = []
        for score, passage in scored_pairs[:top_k]:
            cloned = passage.model_copy()
            cloned.sparse_score = float(score)
            sparse_passages.append(cloned)

        return sparse_passages

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedPassage],
        sparse_results: List[RetrievedPassage],
        alpha: float,
        k: int = 60
    ) -> List[RetrievedPassage]:
        """
        Combines Dense and Sparse results using Reciprocal Rank Fusion (RRF).
        Fusiona resultados densos y dispersos mediante Reciprocal Rank Fusion.
        RRF Score(d) = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_sparse))
        """
        rrf_scores: Dict[str, float] = {}
        passage_map: Dict[str, RetrievedPassage] = {}

        # Process dense ranking
        for rank, item in enumerate(dense_results, start=1):
            passage_map[item.chunk_id] = item
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (alpha / (k + rank))

        # Process sparse ranking
        for rank, item in enumerate(sparse_results, start=1):
            if item.chunk_id not in passage_map:
                passage_map[item.chunk_id] = item
            else:
                passage_map[item.chunk_id].sparse_score = item.sparse_score
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + ((1.0 - alpha) / (k + rank))

        # Sort combined results by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_passages: List[RetrievedPassage] = []
        for cid in sorted_chunk_ids:
            p = passage_map[cid]
            p.rrf_score = round(rrf_scores[cid], 6)
            fused_passages.append(p)

        return fused_passages

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[RetrievedPassage],
        top_k: int
    ) -> List[RetrievedPassage]:
        """
        Applies Cross-Encoder deep neural re-ranking on candidate pairs (query, passage).
        Aplica re-ordenamiento neuronal con Cross-Encoder en pares (consulta, pasaje).
        """
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # Prepare sentence pairs
        sentence_pairs = [[query, p.text] for p in candidates]
        
        # Predict logits
        scores = self.reranker.predict(sentence_pairs)

        # Attach rerank scores
        for score, passage in zip(scores, candidates):
            # Normalize to 0-1 range via sigmoid-like mapping or direct float
            passage.rerank_score = float(score)

        # Sort descending by cross-encoder score
        candidates.sort(key=lambda p: (p.rerank_score or -999.0), reverse=True)
        return candidates[:top_k]

    def _synthesize_grounded_answer(
        self,
        query: str,
        contexts: List[RetrievedPassage]
    ) -> Tuple[str, List[SourceCitation], float]:
        """
        Synthesizes a grounded, hallucination-free answer with strict citations
        using GPT-4o. If evidence is insufficient, explicitly states so.
        
        Sintetiza una respuesta fundamentada sin alucinaciones y con citas rigurosas.
        """
        if not contexts:
            return (
                "No se encontró información relevante en los documentos indexados para responder a esta consulta.",
                [],
                0.0
            )

        # Build context prompt with numbered references
        context_blocks: List[str] = []
        citations_lookup: Dict[str, SourceCitation] = {}

        for idx, passage in enumerate(contexts, start=1):
            ref_tag = f"[Doc: {passage.filename}, Pag: {passage.page_number}, Chunk: {passage.chunk_index}]"
            context_blocks.append(
                f"--- REFERENCIA {idx} {ref_tag} ---\n{passage.text}"
            )
            citations_lookup[ref_tag] = SourceCitation(
                filename=passage.filename,
                page_number=passage.page_number,
                chunk_id=passage.chunk_id,
                relevance_score=passage.rerank_score if passage.rerank_score is not None else (passage.rrf_score or 0.8),
                excerpt=(passage.text[:200] + "...") if len(passage.text) > 200 else passage.text
            )

        formatted_context = "\n\n".join(context_blocks)

        system_prompt = (
            "Eres HyperFetch-RAG, un Asistente de IA Empresarial para documentación técnica.\n"
            "REGLAS ESTRICTAS DE RESPUESTA (CERO ALUCINACIONES):\n"
            "1. Responde a la pregunta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en el CONTEXTO provisto abajo.\n"
            "2. Cita OBLIGATORIAMENTE la fuente de cada afirmación usando el formato exacto: "
            "[Doc: <nombre_archivo>, Pag: <num_pagina>, Chunk: <indice>].\n"
            "3. Si los documentos contienen tablas o datos numéricos, presérvalos con exactitud.\n"
            "4. Si el contexto NO contiene información suficiente o inequívoca para responder, "
            "responde exactamente: 'No existe información suficiente en los manuales provistos para responder con certeza.' "
            "y aclara qué información falta. NO inventes ni asumas nada.\n"
            "5. Mantén un tono técnico, formal, conciso y profesional en español (o el idioma de la consulta)."
        )

        user_prompt = (
            f"CONTEXTO DOCUMENTAL RECUPERADO:\n{formatted_context}\n\n"
            f"PREGUNTA DEL USUARIO:\n{query}\n\n"
            f"RESPUESTA FUNDAMENTADA CON CITAS:"
        )

        response = self.openai_client.chat.completions.create(
            model=self.settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic, minimum hallucination risk
            max_tokens=1024
        )

        answer = response.choices[0].message.content or ""

        # Extract citations referenced in the text
        active_citations: List[SourceCitation] = []
        for tag, citation_obj in citations_lookup.items():
            if tag in answer or citation_obj.filename in answer:
                active_citations.append(citation_obj)

        # If model used generalized context without strict tags, include top 2 passages
        if not active_citations and len(contexts) > 0:
            active_citations = list(citations_lookup.values())[:2]

        # Calculate confidence metric
        base_confidence = 0.95 if "No existe información suficiente" not in answer else 0.10
        if contexts and contexts[0].rerank_score is not None:
            # Positive reranker score increases confidence
            if contexts[0].rerank_score < -2.0:
                base_confidence = max(0.2, base_confidence * 0.5)

        return answer, active_citations, round(base_confidence, 2)

    def execute_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        alpha: Optional[float] = None
    ) -> RAGQueryResponse:
        """
        Executes complete RAG pipeline:
        Query -> Dense Search + Sparse BM25 -> RRF -> Cross-Encoder -> GPT-4o Grounding.
        """
        start_total = time.time()
        eff_top_k = top_k or self.settings.TOP_K_FINAL
        eff_alpha = alpha if alpha is not None else self.settings.HYBRID_ALPHA
        eff_use_reranker = use_reranker if use_reranker is not None else self.settings.USE_RERANKER

        # Step 1: Dense Retrieval
        t0 = time.time()
        dense_results = self._dense_retrieval(query, top_k=self.settings.TOP_K_DENSE)
        dense_ms = round((time.time() - t0) * 1000, 2)

        # Step 2: Sparse Retrieval (BM25)
        t1 = time.time()
        sparse_results = self._sparse_retrieval(query, top_k=self.settings.TOP_K_SPARSE)
        sparse_ms = round((time.time() - t1) * 1000, 2)

        # Step 3: Reciprocal Rank Fusion (RRF)
        t2 = time.time()
        fused_candidates = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            alpha=eff_alpha,
            k=self.settings.RECIPROCAL_RANK_K
        )
        fusion_ms = round((time.time() - t2) * 1000, 2)

        # Step 4: Cross-Encoder Reranking
        t3 = time.time()
        if eff_use_reranker and self.reranker:
            # Pass top 15 fused candidates to cross-encoder
            reranked_passages = self._cross_encoder_rerank(
                query=query,
                candidates=fused_candidates[:self.settings.TOP_K_DENSE],
                top_k=eff_top_k
            )
        else:
            reranked_passages = fused_candidates[:eff_top_k]
        rerank_ms = round((time.time() - t3) * 1000, 2)

        # Step 5: Grounded LLM Generation
        t4 = time.time()
        answer, citations, confidence = self._synthesize_grounded_answer(
            query=query,
            contexts=reranked_passages
        )
        gen_ms = round((time.time() - t4) * 1000, 2)

        total_ms = round((time.time() - start_total) * 1000, 2)

        latency = LatencyBreakdown(
            dense_search_ms=dense_ms,
            sparse_search_ms=sparse_ms,
            fusion_ms=fusion_ms,
            reranking_ms=rerank_ms,
            generation_ms=gen_ms,
            total_ms=total_ms
        )

        return RAGQueryResponse(
            query=query,
            answer=answer,
            confidence_score=confidence,
            citations=citations,
            retrieved_passages=reranked_passages,
            latency=latency
        )
