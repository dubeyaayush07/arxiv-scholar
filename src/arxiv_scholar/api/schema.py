from pydantic import BaseModel
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    use_reranker: bool = True

class SourceNode(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class StreamMetadataEvent(BaseModel):
    type: str = "metadata"
    sources: List[SourceNode]
    paper_urls: List[str]

class StreamTokenEvent(BaseModel):
    type: str = "token"
    content: str

class StreamDoneEvent(BaseModel):
    type: str = "done"
    latency_ms: float
