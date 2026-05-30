import os
import re
import json
import asyncio
import logging
from typing import Any, Dict, List
from arxiv_scholar.llm.service import LLMService

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from arxiv_scholar.retrieval.router import Route, route_query, MLQueryRouter

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
        reranker_model_name: str = None,
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        self.reranker_model = None
        if reranker_model_name:
            self.reranker_model = TextCrossEncoder(model_name=reranker_model_name)
        
        # Initialize ML router
        self.router = MLQueryRouter()
        
        # Initialize LLM Service
        self.llm_service = LLMService()

    async def retrieve(self, query: str, limit: int = 20, use_reranker: bool = False) -> List[Dict[str, Any]]:
        logger.info(f"Orchestrator.retrieve called with query: '{query}', limit={limit}, use_reranker={use_reranker}")
        # Compute dense embedding to feed to the ML router
        dense_vec = list(self.dense_model.embed([query]))[0].tolist()
        route = self.router.route(query, query_vector=dense_vec)
        logger.info(f"Router selected route: {route.name}")
        # Note: In a real system, we'd return the route to log the prometheus path metric
        # We will attach the path to the returned payload for benchmarking scripts to read
        
        if route == Route.DIRECT:
            results = await self._execute_direct(query, limit, use_reranker)
        elif route == Route.DECOMPOSE:
            results = await self._execute_decompose(query, limit, use_reranker)
        elif route == Route.HYDE:
            if not self.llm_service.client:
                results = await self._execute_direct(query, limit, use_reranker)
            else:
                results = await self._execute_hyde(query, limit, use_reranker)
        else:
            results = await self._execute_direct(query, limit, use_reranker)
            
        # Attach route metadata so benchmark can track it
        for r in results:
            r["_query_path"] = route.value
            
        return results

    def _hybrid_search_sync(self, query: str, limit: int = 20, dense_query: str = None, use_reranker: bool = False) -> List[Any]:
        # dense_query can be the generated abstract for HyDE
        dq = dense_query if dense_query else query
        
        # FastEmbed operations are synchronous and CPU bound
        dense_vec = list(self.dense_model.embed([dq]))[0].tolist()
        sparse_vec = list(self.sparse_model.embed([query]))[0]
        
        fetch_limit = limit * 5 if (use_reranker and self.reranker_model) else limit
        logger.debug(f"Executing Qdrant hybrid search for query: '{query}', limit={limit}, fetch_limit={fetch_limit}")

        # Qdrant networking
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using="", limit=fetch_limit),
                Prefetch(
                    query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                    using="bm25",
                    limit=fetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
        )
        return response.points

    async def _execute_direct(self, query: str, limit: int = 20, use_reranker: bool = False) -> List[Dict[str, Any]]:
        points = await asyncio.to_thread(self._hybrid_search_sync, query, limit, None, use_reranker)
        return await asyncio.to_thread(self._format_results, points, query, limit, use_reranker)

    async def _execute_hyde(self, query: str, limit: int = 20, use_reranker: bool = False) -> List[Dict[str, Any]]:
        # 1. Generate hypothetical abstract via API
        abstract = await self.llm_service.generate_hyde_abstract(query)
        logger.info(f"Generated HyDE abstract: '{abstract[:100]}...'")
        
        # 2. Hybrid search (Dense uses abstract, Sparse uses original query)
        points = await asyncio.to_thread(self._hybrid_search_sync, query, limit, abstract, use_reranker)
        return await asyncio.to_thread(self._format_results, points, query, limit, use_reranker)

    async def _execute_decompose(self, query: str, limit: int = 20, use_reranker: bool = False) -> List[Dict[str, Any]]:
        if not self.llm_service.client:
            logger.warning("No LLM client configured for DECOMPOSE. Falling back to DIRECT.")
            return await self._execute_direct(query, limit, use_reranker)
            
        # 1. Generate fully contextualized sub-queries via LLM
        sub_queries = await self.llm_service.decompose_query(query)
        
        if not sub_queries:
            sub_queries = [query]
            
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
        # 2. Fire concurrent searches for each sub-query
        tasks = [self._execute_direct(sq, limit, use_reranker) for sq in sub_queries]
        results_lists = await asyncio.gather(*tasks)
        
        # 3. Merge and deduplicate interface
        all_results = []
        seen = set()
        for r_list in results_lists:
            for r in r_list:
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    all_results.append(r)
                    
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        return all_results[:limit]

    def _format_results(self, points: List[Any], query_text: str = None, limit: int = 20, use_reranker: bool = False) -> List[Dict[str, Any]]:
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "chunk_id": str(point.id),  # MUST be exactly str(point.id) to match eval ground-truths (UUIDs)
                "text": payload.get("content", ""),
                "score": point.score,
                "metadata": payload.get("metadata", {}),
            })
            
        if use_reranker and self.reranker_model and results and query_text:
            logger.debug(f"Applying cross-encoder reranking for {len(results)} chunks")
            documents = [res["text"][:500] for res in results]
            cross_scores = list(self.reranker_model.rerank(query_text, documents))
            
            for i, res in enumerate(results):
                res["score"] = float(cross_scores[i])
                
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
        return results[:limit]
