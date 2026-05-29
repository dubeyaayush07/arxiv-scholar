import time
import os
import logging
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from arxiv_scholar.retrieval.retrieval import HybridRetriever
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
    logger.info("Initializing HybridRetriever with BGE Re-ranker...")
    retriever = HybridRetriever(
        collection_name="arxiv_papers",
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        reranker_model_name="BAAI/bge-reranker-base"
    )
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    llm_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    ) if api_key else None
    
    app_state["retriever"] = retriever
    app_state["llm_client"] = llm_client
    
    yield
    
    # Shutdown
    app_state.clear()

app = FastAPI(title="Arxiv Scholar RAG API", lifespan=lifespan)

@app.post("/api/v1/query")
async def query_endpoint(request: QueryRequest):
    start_time = time.perf_counter()
    
    retriever = app_state.get("retriever")
    llm_client = app_state.get("llm_client")
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
        
    async def _stream_response():
        try:
            # 1. Retrieve & Re-rank
            # HybridRetriever is synchronous in main branch, so we wrap it in a thread
            chunks = await asyncio.to_thread(
                retriever.retrieve,
                request.query,
                limit=request.limit,
                use_reranker=request.use_reranker
            )
            
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
            if llm_client and context_str:
                prompt = f"""You are an academic research assistant. 
Answer the user's query comprehensively based ONLY on the provided context chunks. 
If the answer is not in the context, state that clearly. Cite your sources where applicable using the provided Source URL (e.g. [https://arxiv.org/abs/2010.05432]).

CRITICAL: Return ONLY the final answer. Do NOT use conversational filler (e.g. "Based on the provided context..."). Start your answer immediately.

Context:
{context_str}

Query: {request.query}
"""
                stream = await llm_client.chat.completions.create(
                    model="anthropic/claude-3.5-haiku",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    stream=True
                )
                
                # YIELD 2: Token Events
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        token_event = StreamTokenEvent(content=content)
                        yield f"data: {token_event.model_dump_json()}\n\n"
                        
            # YIELD 3: Done Event
            latency = (time.perf_counter() - start_time) * 1000
            done_event = StreamDoneEvent(latency_ms=latency)
            yield f"data: {done_event.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            yield f"data: {{\"type\": \"error\", \"detail\": \"{str(e)}\"}}\n\n"
            
    return StreamingResponse(_stream_response(), media_type="text/event-stream")
