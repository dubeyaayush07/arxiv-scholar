import os
import json
import random
import argparse
import logging
from tqdm import tqdm
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import importlib.util
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.py"))
spec = importlib.util.spec_from_file_location("local_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchValue, FusionQuery, Fusion, SparseVector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def setup_clients():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing! Export it before running.")
        
    llm_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    qdrant_client_obj = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    dense_embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    sparse_embedder = SparseBM25Embedder()
    
    return llm_client, qdrant_client_obj, dense_embedder, sparse_embedder


def sample_chunks(qdrant_client_obj, collection_name, sample_size=500):
    logger.info(f"Sampling {sample_size} chunks from Qdrant...")
    points, _ = qdrant_client_obj.scroll(
        collection_name=collection_name,
        limit=sample_size * 2, 
        with_payload=True,
        with_vectors=False
    )
    
    random.shuffle(points)
    return points[:sample_size]

def parse_llm_json(response, default_return=None):
    if not response or not hasattr(response, "choices") or not response.choices:
        return default_return
    
    content = response.choices[0].message.content
    if not content:
        return default_return
        
    # Strip markdown code block fences if the model aggressively included them
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    # Extract just the JSON object to ignore conversational filler
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        content = content[start_idx:end_idx+1]
        
    try:
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON: {e} | Content: {content}")
        return default_return


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=60),
    retry=retry_if_exception_type((openai.APIError, openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
    before_sleep=lambda retry_state: logger.warning(f"LLM network error/rate-limit. Retrying in {retry_state.next_action.sleep:.1f}s...")
)
def _call_llm(llm_client, model, messages, response_format):
    return llm_client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format
    )

def generate_seed_query(llm_client, chunk_text: str) -> dict:
    prompt = f"""
    You are an AI researcher. Given the following text chunk from an arXiv paper, 
    generate ONE highly specific query that this exact chunk perfectly answers.
    The query must sound like a human typing into a search engine.
    
    Chunk: {chunk_text}
    
    Return ONLY JSON in this format: {{"query": "your generated query"}}
    """
    
    response = _call_llm(
        llm_client=llm_client,
        model="anthropic/claude-3.5-haiku",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = parse_llm_json(response)
    if not result:
        raise ValueError("LLM returned empty or invalid JSON")
    return result

def check_false_negatives_batch(llm_client, query: str, chunks: list[str]) -> list[bool]:
    """Checks multiple chunks in one LLM call to save prompt tokens."""
    if not chunks:
        return []
        
    formatted_chunks = ""
    for i, c in enumerate(chunks):
        # Truncate chunk to 1000 characters to aggressively save input tokens
        formatted_chunks += f"--- Chunk {i} ---\n{c[:1000]}\n\n"
        
    prompt = f"""
    Query: {query}
    
    Here are {len(chunks)} chunks of text. For each chunk, determine if it successfully answers or strongly addresses the query.
    {formatted_chunks}
    
    Return ONLY valid JSON in this exact format, with NO comments or explanations:
    {{
        "results": [true, false]
    }}
    """
    response = _call_llm(
        llm_client=llm_client,
        model="anthropic/claude-3.5-haiku",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = parse_llm_json(response, default_return={})
    results_list = result.get("results", [])
    
    # Pad or truncate list in case LLM hallucinates length
    while len(results_list) < len(chunks):
        results_list.append(False)
    return [bool(x) for x in results_list[:len(chunks)]]


def hybrid_search(qdrant_client_obj, dense_embedder, sparse_embedder, query: str, collection_name: str, top_k=10):
    dense_vec = dense_embedder.embed([query])[0]
    sparse_vec = sparse_embedder.embed([query])[0]

    results = qdrant_client_obj.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_vec, using="", limit=top_k*2),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="bm25",
                limit=top_k*2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    return results.points


def generate_adversarial_query(llm_client, chunk_text: str, original_query: str, trap_type: str) -> str:
    if trap_type == "negation":
        prompt = f"""
        Original Query: {original_query}
        Context: {chunk_text}
        
        Rewrite this query to ask for the EXACT OPPOSITE or a critique of the concept. 
        Example: "How does attention work?" -> "Which papers prove attention mechanisms fail on long sequences?"
        
        Return ONLY JSON in this format: {{"query": "your adversarial query"}}
        """
    else: 
        prompt = f"""
        Original Query: {original_query}
        
        Rewrite this query to require BOTH the original information AND an arbitrary academic constraint (e.g., year, framework, author).
        Example: "Add a constraint that the paper must have been published in 2023 and uses PyTorch."
        
        Return ONLY JSON in this format: {{"query": "your adversarial query"}}
        """
        
    response = _call_llm(
        llm_client=llm_client,
        model="anthropic/claude-3.5-haiku",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = parse_llm_json(response, default_return={"query": original_query})
    return result.get("query", original_query)


def main(args):
    llm_client, qdrant_client_obj, dense_embedder, sparse_embedder = setup_clients()
    collection_name = args.collection
    
    sample_size = 2 if args.trial else 100
    seed_points = sample_chunks(qdrant_client_obj, collection_name, sample_size)
    
    os.makedirs("data", exist_ok=True)
    if args.out_file:
        out_file = args.out_file
    else:
        out_file = "data/eval_dataset_trial.jsonl" if args.trial else "data/eval_dataset.jsonl"
    
    logger.info(f"Generating evaluation dataset for {sample_size} queries...")
    
    with open(out_file, "w") as f:
        for idx, point in enumerate(tqdm(seed_points)):
            chunk_text = point.payload.get("content", "")
            if not chunk_text:
                continue
                
            try:
                seed_data = generate_seed_query(llm_client, chunk_text)
                base_query = seed_data["query"]
            except Exception as e:
                logger.error(f"LLM failed on seed generation after retries: {e}")
                continue
                
            trap = random.choices(["none", "negation", "compositional"], weights=[0.8, 0.1, 0.1])[0]
            if trap != "none":
                final_query = generate_adversarial_query(llm_client, chunk_text, base_query, trap)
                query_type = trap
            else:
                final_query = base_query
                query_type = "standard"
                
            raw_results = hybrid_search(qdrant_client_obj, dense_embedder, sparse_embedder, final_query, collection_name)
            
            candidates = []
            for result in raw_results:
                if str(result.id) == str(point.id):
                    continue
                candidates.append(result)
                if len(candidates) >= 5:
                    break
                    
            candidate_texts = [c.payload.get("content", "") for c in candidates]
            batch_results = check_false_negatives_batch(llm_client, final_query, candidate_texts)
            
            hard_negatives = []
            for candidate, is_false_neg in zip(candidates, batch_results):
                if not is_false_neg:
                    hard_negatives.append({
                        "chunk_id": str(candidate.id),
                        "text": candidate.payload.get("content", ""),
                        "type": "retrieval_mined"
                    })
                    
            eval_obj = {
                "query_id": f"q_{idx:03d}",
                "query": final_query,
                "query_type": query_type,
                "positive_chunk": {
                    "chunk_id": str(point.id),
                    "text": chunk_text
                },
                "hard_negatives": hard_negatives
            }
            f.write(json.dumps(eval_obj) + "\n")
            f.flush()

    logger.info(f"Done! Dataset saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", action="store_true", help="Run a small trial batch (N=2)")
    parser.add_argument("--collection", type=str, default="arxiv_papers", help="Qdrant collection name")
    parser.add_argument("--out_file", type=str, default=None, help="Output JSONL file path")
    args = parser.parse_args()
    main(args)
