import os
import time
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from prometheus_client import start_http_server, Summary, Histogram, Counter
import asyncio

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.py"))
spec = importlib.util.spec_from_file_location("local_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

from arxiv_scholar.retrieval.orchestrator import AdvancedRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prometheus Metrics
RETRIEVAL_LATENCY = Summary('retrieval_latency_seconds', 'Time spent retrieving from Qdrant')
RETRIEVAL_LATENCY_HIST = Histogram('retrieval_latency_histogram_seconds', 'Retrieval latency histogram', buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
QUERY_PATH_COUNTER = Counter('query_path_total', 'Distribution of query paths taken', ['path'])

def calculate_ndcg(retrieved_ids, target_id, hard_negative_ids, k=10):
    dcg = 0.0
    idcg = 1.0 # Max possible gain is 1.0 at rank 1, plus 0.1 for up to K-1 hard negatives
    
    # Calculate IDCG dynamically based on available hard negatives
    for i in range(1, min(k, len(hard_negative_ids) + 1)):
        idcg += 0.1 / np.log2((i+1) + 1)
        
    for i, res_id in enumerate(retrieved_ids[:k]):
        rank = i + 1
        if str(res_id) == str(target_id):
            rel = 1.0
        elif str(res_id) in hard_negative_ids:
            rel = 0.1
        else:
            rel = 0.0
            
        dcg += rel / np.log2(rank + 1)
        
    return dcg / idcg if idcg > 0 else 0.0

async def run_evaluation(data_file: str, collection_name: str):
    logger.info(f"Loading eval dataset: {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}. Did you run generate_eval_dataset.py?")
        
    with open(data_file, "r") as f:
        queries = [json.loads(line) for line in f]
        
    # Filter out adversarial queries due to logical ground-truth mismatch
    queries = [q for q in queries if q.get("query_type") == "standard"]
        
    retriever = AdvancedRetriever(
        collection_name=collection_name, 
        qdrant_host=config.QDRANT_HOST, 
        qdrant_port=config.QDRANT_PORT,
        reranker_model_name="jinaai/jina-reranker-v1-tiny-en"
    )
    
    results_list = []
    
    for use_reranker in [False, True]:
        mode_name = f"{collection_name} (Reranked)" if use_reranker else f"{collection_name} (Baseline)"
        
        recalls_5 = []
        recalls_10 = []
        recalls_20 = []
        ndcgs = []
        latencies = []
        
        logger.info(f"Running benchmarking against {mode_name} for {len(queries)} queries...")
        for q in tqdm(queries):
            query_text = q["query"]
            target_id = q["positive_chunk"]["chunk_id"]
            hard_neg_ids = [hn["chunk_id"] for hn in q["hard_negatives"]]
            
            with RETRIEVAL_LATENCY.time():
                start_t = time.perf_counter()
                results = await retriever.retrieve(query_text, limit=20, use_reranker=use_reranker)
                end_t = time.perf_counter()
                latency = end_t - start_t
                RETRIEVAL_LATENCY_HIST.observe(latency)
                
            if results:
                path = results[0].get("_query_path", "direct")
                QUERY_PATH_COUNTER.labels(path=path).inc()
                
            retrieved_ids = [str(res["chunk_id"]) for res in results]
            
            # Calculate Metrics
            r5 = 1.0 if target_id in retrieved_ids[:5] else 0.0
            r10 = 1.0 if target_id in retrieved_ids[:10] else 0.0
            r20 = 1.0 if target_id in retrieved_ids[:20] else 0.0
            
            ndcg_val = calculate_ndcg(retrieved_ids, target_id, hard_neg_ids, k=10)
            
            recalls_5.append(r5)
            recalls_10.append(r10)
            recalls_20.append(r20)
            ndcgs.append(ndcg_val)
            latencies.append(latency)
            
        metrics = {
            "Collection": mode_name,
            "Queries": len(queries),
            "Recall@5": np.mean(recalls_5),
            "Recall@10": np.mean(recalls_10),
            "Recall@20": np.mean(recalls_20),
            "nDCG@10": np.mean(ndcgs),
            "Latency_p50": np.percentile(latencies, 50),
            "Latency_p95": np.percentile(latencies, 95),
            "Latency_p99": np.percentile(latencies, 99),
            "Avg_Latency_ms": np.mean(latencies) * 1000
        }
        results_list.append(metrics)
        
    return results_list

def calculate_cost(latency_mean_ms):
    # AWS EC2 t3.xlarge (4 vCPUs, 16GB) is ~$0.166/hr -> ~$0.000046/sec
    # We estimate cost based on embedding + retrieval compute time per 1000 queries
    cost_per_sec = 0.166 / 3600
    sec_per_1k = (latency_mean_ms / 1000.0) * 1000
    return sec_per_1k * cost_per_sec

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval_dataset.jsonl")
    parser.add_argument("--metrics-port", type=int, default=8000)
    args = parser.parse_args()
    
    # Start prometheus metrics server
    logger.info(f"Starting Prometheus endpoint on port {args.metrics_port}")
    start_http_server(args.metrics_port)
    
    collections = ["arxiv_papers"]
    results = []
    
    for coll in collections:
        res_list = await run_evaluation(args.data, coll)
        for res in res_list:
            res["Cost_per_1k"] = calculate_cost(res["Avg_Latency_ms"])
            results.append(res)
        
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    format_str = "{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(format_str.format("Collection", "Recall@5", "Recall@10", "Recall@20", "nDCG@10", "p95 (ms)", "p99 (ms)", "Cost/1k"))
    print("-" * 110)
    
    for r in results:
        print(format_str.format(
            r["Collection"],
            f"{r['Recall@5']:.3f}",
            f"{r['Recall@10']:.3f}",
            f"{r['Recall@20']:.3f}",
            f"{r['nDCG@10']:.3f}",
            f"{r['Latency_p95']*1000:.1f}",
            f"{r['Latency_p99']*1000:.1f}",
            f"${r['Cost_per_1k']:.4f}"
        ))
    print("="*110)
    
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Sleep to allow prometheus scraper to hit the endpoint if desired
    logger.info("Benchmark complete. Serving metrics for 10 seconds...")
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main_async())
