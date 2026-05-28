import os
import re
import json
import asyncio
import logging
from typing import Any, Dict, List
import openai
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding

from arxiv_scholar.retrieval.router import Route, route_query

logger = logging.getLogger(__name__)

class AdvancedRetriever:
    def __init__(
        self,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        ) if api_key else None

    async def retrieve(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        route = route_query(query)
        # Note: In a real system, we'd return the route to log the prometheus path metric
        # We will attach the path to the returned payload for benchmarking scripts to read
        
        if route == Route.DIRECT:
            results = await self._execute_direct(query, limit)
        elif route == Route.DECOMPOSE:
            results = await self._execute_decompose(query, limit)
        elif route == Route.HYDE:
            if not self.llm_client:
                results = await self._execute_direct(query, limit)
            else:
                results = await self._execute_hyde(query, limit)
        else:
            results = await self._execute_direct(query, limit)
            
        # Attach route metadata so benchmark can track it
        for r in results:
            r["_query_path"] = route.value
            
        return results

    def _hybrid_search_sync(self, query: str, limit: int = 20, dense_query: str = None) -> List[Any]:
        # dense_query can be the generated abstract for HyDE
        dq = dense_query if dense_query else query
        
        # FastEmbed operations are synchronous and CPU bound
        dense_vec = list(self.dense_model.embed([dq]))[0].tolist()
        sparse_vec = list(self.sparse_model.embed([query]))[0]

        # Qdrant networking
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using="", limit=limit*2),
                Prefetch(
                    query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                    using="bm25",
                    limit=limit*2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
        return response.points

    async def _execute_direct(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        points = await asyncio.to_thread(self._hybrid_search_sync, query, limit)
        return self._format_results(points)

    def _generate_hyde_sync(self, query: str) -> str:
        prompt = f"""
        Please write a brief academic abstract that directly answers the following query. 
        Do not use conversational filler, just write the abstract.
        Query: {query}
        """
        response = self.llm_client.chat.completions.create(
            model="anthropic/claude-3.5-haiku",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    async def _execute_hyde(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        # 1. Generate hypothetical abstract via API
        abstract = await asyncio.to_thread(self._generate_hyde_sync, query)
        
        # 2. Hybrid search (Dense uses abstract, Sparse uses original query)
        points = await asyncio.to_thread(self._hybrid_search_sync, query, limit, abstract)
        return self._format_results(points)

    def _generate_decomposition_sync(self, query: str) -> List[str]:
        prompt = f"""
        Decompose this query into independent, fully contextualized sub-queries. 
        Each sub-query must be able to stand completely on its own for a search engine.
        For example: "Accuracy of BERT vs GPT-3" -> ["What is the accuracy of BERT?", "What is the accuracy of GPT-3?"]
        
        Query: {query}
        
        Return ONLY valid JSON in this exact format:
        {{
            "sub_queries": ["sub query 1", "sub query 2"]
        }}
        """
        response = self.llm_client.chat.completions.create(
            model="anthropic/claude-3.5-haiku",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Safely strip markdown formatting if the model included it despite the json_object constraint
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return data.get("sub_queries", [query])
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON for decomposition: {e} | Content: {content}")
            return [query]

    async def _execute_decompose(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.llm_client:
            logger.warning("No LLM client configured for DECOMPOSE. Falling back to DIRECT.")
            return await self._execute_direct(query, limit)
            
        # 1. Generate fully contextualized sub-queries via LLM
        sub_queries = await asyncio.to_thread(self._generate_decomposition_sync, query)
        
        if not sub_queries:
            sub_queries = [query]
            
        # 2. Fire concurrent searches for each sub-query
        tasks = [self._execute_direct(sq, limit) for sq in sub_queries]
        results_lists = await asyncio.gather(*tasks)
        
        # 3. Merge and deduplicate interface
        all_results = []
        seen = set()
        for r_list in results_lists:
            for r in r_list:
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    all_results.append(r)
                    
        return all_results[:limit]

    def _format_results(self, points: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "chunk_id": str(point.id),  # MUST be exactly str(point.id) to match eval ground-truths (UUIDs)
                "text": payload.get("content", ""),
                "score": point.score,
                "metadata": payload.get("metadata", {}),
            })
        return results
