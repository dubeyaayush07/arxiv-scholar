import time
import os
import logging
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from arxiv_scholar.llm.service import LLMService

from arxiv_scholar.retrieval.orchestrator import Orchestrator
from arxiv_scholar.api.schema import (
    QueryRequest, 
    SourceNode, 
    StreamMetadataEvent, 
    StreamTokenEvent, 
    StreamDoneEvent
)

# Import config (falling back to localhost if not available)
try:
    from configs.config import QDRANT_HOST, QDRANT_PORT
except ImportError:
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333

logger = logging.getLogger(__name__)

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Orchestrator with ML Router and BGE Re-ranker...")
    retriever = Orchestrator(
        collection_name="arxiv_papers",
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        reranker_model_name="BAAI/bge-reranker-base"
    )
    
    llm_service = LLMService()
    
    app_state["retriever"] = retriever
    app_state["llm_service"] = llm_service
    
    yield
    
    # Shutdown
    app_state.clear()

app = FastAPI(title="Arxiv Scholar RAG API", lifespan=lifespan)

@app.post("/api/v1/query")
async def query_endpoint(request: QueryRequest):
    logger.info(f"Received query request: query='{request.query}', limit={request.limit}, rerank={request.use_reranker}")
    start_time = time.perf_counter()
    
    retriever = app_state.get("retriever")
    llm_service = app_state.get("llm_service")
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
        
    async def _stream_response():
        try:
            # 1. Retrieve & Re-rank
            # Orchestrator is natively async, so we await it directly
            logger.debug(f"Starting retrieval for query: '{request.query}'")
            chunks = await retriever.retrieve(
                request.query,
                limit=request.limit,
                use_reranker=request.use_reranker
            )
            logger.debug(f"Retrieval completed. Fetched {len(chunks)} chunks.")
            
            # 2. Contextualize
            context_blocks = []
            sources = []
            paper_urls_set = set()
            
            for i, chunk in enumerate(chunks):
                arxiv_id = chunk["metadata"].get("arxiv_id")
                url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"Unknown Source {i+1}"
                if arxiv_id:
                    paper_urls_set.add(url)
                    
                context_blocks.append(f"Context {i+1} (Source: {url}):\n{chunk['text']}")
                sources.append(
                    SourceNode(
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=chunk["score"],
                        metadata=chunk["metadata"]
                    )
                )
                
            context_str = "\n\n".join(context_blocks)
            paper_urls = list(paper_urls_set)
            
            # YIELD 1: Metadata Event (Sent instantly)
            meta_event = StreamMetadataEvent(sources=sources, paper_urls=paper_urls)
            yield f"data: {meta_event.model_dump_json()}\n\n"
            
            # 3. LLM Synthesis Streaming
            if llm_service and llm_service.client and context_str:
                logger.debug(f"Starting LLM stream synthesis for query: '{request.query}'")
                stream = llm_service.stream_synthesis(request.query, context_str)
                
                # YIELD 2: Token Events
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        token_event = StreamTokenEvent(content=content)
                        yield f"data: {token_event.model_dump_json()}\n\n"
                        
                logger.debug(f"LLM stream synthesis completed for query: '{request.query}'")
                        
            # YIELD 3: Done Event
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Request completed successfully in {latency:.2f}ms")
            done_event = StreamDoneEvent(latency_ms=latency)
            yield f"data: {done_event.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error(f"Error during retrieval for query '{request.query}': {e}", exc_info=True)
            yield f"data: {{\"type\": \"error\", \"detail\": \"{str(e)}\"}}\n\n"
            
    return StreamingResponse(_stream_response(), media_type="text/event-stream")
