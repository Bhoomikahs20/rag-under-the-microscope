"""
Chapter 7 Generator-Scaling Experiment Runner
Evaluates google/flan-t5-base (250M) vs google/flan-t5-large (780M)
Under Finalized Retrieval Pipeline:
250w/50o Chunks, BGE Dense + BM25 Hybrid (alpha=0.5), K_init=15, Cross-Encoder Reranking, K_final=5.
"""

import sys
import os
import json
import time
import re
import string
import numpy as np
import torch
import faiss
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chunker import chunk_corpus
from src.rag.generator import RAGGenerator
from src.utils.logging_config import setup_logger

logger = setup_logger("generator_scaling_runner")

SEED = 42
torch.manual_seed(SEED)

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text for BM25 indexing (lowercased alphanumeric tokens)."""
    return re.findall(r'\w+', text.lower())

def normalize_answer(s: str) -> str:
    """Lowercases text, removes punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_correctness(prediction: str, ground_truth: str) -> dict:
    """Computes exact match, token F1, and correctness assessment."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)

    exact_match = (norm_pred == norm_gold)
    substring_match = (norm_gold in norm_pred) or (norm_pred in norm_gold and len(norm_pred) > 2)

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        f1 = float(exact_match)
    elif num_same == 0:
        f1 = 0.0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

    is_correct = exact_match or (f1 >= 0.8) or (substring_match and len(norm_pred) > 0)

    return {
        "exact_match": exact_match,
        "f1_score": round(f1, 4),
        "substring_match": substring_match,
        "is_correct": bool(is_correct)
    }

def classify_failure(
    query_idx: int,
    query_id: str,
    question: str,
    gold_answer: str,
    rag_answer: str,
    gold_doc_ids: set,
    retrieved_doc_ids: set
) -> str:
    """Classifies failure using established error-analysis methodology."""
    rel_retrieved = gold_doc_ids.intersection(retrieved_doc_ids)
    all_gold_retrieved = (len(rel_retrieved) == len(gold_doc_ids))
    
    if query_id == "5adf48b25542995ec70e8fb5":
        return "Gold-Answer / Data-Quality Issue"
        
    if not all_gold_retrieved:
        return "Retrieval Failure"
        
    if "The relevant information to answer the above question is" in rag_answer or len(rag_answer) > 100:
        return "Generation Failure"
        
    return "Multi-Hop Reasoning Failure"

class DenseVectorRetriever:
    """FAISS Vector Retriever with BAAI/bge-small-en-v1.5."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        logger.info(f"Initializing SentenceTransformer: '{model_name}'...")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> float:
        t0 = time.time()
        self.chunks = chunks
        formatted_texts = [
            f"{c['title']}: {c['text']}" if c.get("title") else c["text"] for c in chunks
        ]

        logger.info(f"Encoding {len(chunks)} chunks with '{self.model_name}'...")
        embeddings = self.embedder.encode(
            formatted_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        elapsed = round(time.time() - t0, 4)
        logger.info(f"Dense FAISS index built with {self.index.ntotal} vectors in {elapsed}s.")
        return elapsed

    def search(self, query: str, top_k: int = 15) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.time()
        q_embed = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, indices = self.index.search(q_embed, top_k)
        latency = (time.time() - t0) * 1000.0  # ms

        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["similarity_score"] = float(score)
            chunk_copy["rank"] = rank + 1
            retrieved.append(chunk_copy)

        return retrieved, latency

class BM25LexicalRetriever:
    """BM25 Lexical Retriever."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        t0 = time.time()
        self.corpus_tokens = [
            tokenize_text(f"{c['title']}: {c['text']}" if c.get("title") else c["text"]) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.indexing_time = round(time.time() - t0, 4)

    def search(self, query: str, top_k: int = 15) -> Tuple[List[Dict[str, Any]], float]:
        t0 = time.time()
        query_tokens = tokenize_text(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        latency = (time.time() - t0) * 1000.0  # ms

        results = []
        for rank, idx in enumerate(top_indices):
            c_copy = dict(self.chunks[idx])
            c_copy["bm25_score"] = float(scores[idx])
            c_copy["rank"] = rank + 1
            results.append(c_copy)
        return results, latency

def perform_hybrid_fusion(
    query: str,
    dense_retriever: DenseVectorRetriever,
    bm25_retriever: BM25LexicalRetriever,
    top_k: int = 15,
    alpha: float = 0.5
) -> Tuple[List[Dict[str, Any]], float]:
    """Performs normalized score fusion: alpha * Dense_Norm + (1-alpha) * BM25_Norm."""
    t0 = time.time()
    
    dense_results, _ = dense_retriever.search(query, top_k=len(dense_retriever.chunks))
    dense_map = {c["chunk_id"]: c["similarity_score"] for c in dense_results}

    query_tokens = tokenize_text(query)
    bm25_scores = bm25_retriever.bm25.get_scores(query_tokens)
    bm25_map = {bm25_retriever.chunks[i]["chunk_id"]: float(bm25_scores[i]) for i in range(len(bm25_scores))}

    d_vals = list(dense_map.values())
    b_vals = list(bm25_map.values())

    d_min, d_max = min(d_vals), max(d_vals)
    b_min, b_max = min(b_vals), max(b_vals)

    d_range = (d_max - d_min) if (d_max - d_min) > 1e-8 else 1.0
    b_range = (b_max - b_min) if (b_max - b_min) > 1e-8 else 1.0

    hybrid_chunks = []
    for c in dense_retriever.chunks:
        cid = c["chunk_id"]
        raw_d = dense_map[cid]
        raw_b = bm25_map[cid]

        norm_d = (raw_d - d_min) / d_range
        norm_b = (raw_b - b_min) / b_range
        h_score = alpha * norm_d + (1.0 - alpha) * norm_b

        c_copy = dict(c)
        c_copy["dense_score"] = round(raw_d, 4)
        c_copy["bm25_score"] = round(raw_b, 4)
        c_copy["norm_dense_score"] = round(norm_d, 4)
        c_copy["norm_bm25_score"] = round(norm_b, 4)
        c_copy["hybrid_score"] = round(h_score, 4)
        hybrid_chunks.append(c_copy)

    hybrid_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)
    latency = (time.time() - t0) * 1000.0  # ms

    for rank, c in enumerate(hybrid_chunks[:top_k], 1):
        c["rank"] = rank

    return hybrid_chunks[:top_k], latency

def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Computes mean, p50, and p95 for latencies in ms."""
    if not latencies:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2)
    }

def main():
    t_total_start = time.time()
    output_dir = "data/processed/rag_baseline/chapter_7_ablation/generator_scaling"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading pilot datasets for Generator Scaling experiment...")
    corpus_file = "data/processed/hotpotqa_pilot_corpus.json"
    queries_file = "data/processed/hotpotqa_pilot_queries.json"
    qrels_file = "data/processed/hotpotqa_pilot_qrels.json"
    baseline_results_file = "data/processed/rag_baseline/rag_results.json"

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    # Qrels mapping
    qrels_map = {}
    for q in qrels:
        qid = str(q["query_id"])
        if qid not in qrels_map:
            qrels_map[qid] = set()
        qrels_map[qid].add(str(q["corpus_id"]))

    # Gold answers mapping
    gold_answers = {}
    with open(baseline_results_file, "r", encoding="utf-8") as f:
        b_data = json.load(f)
        for item in b_data["per_question_results"]:
            gold_answers[item["query_id"]] = item["gold_answer"]

    # 1. Standard Chunks (250w / 50o)
    chunks = chunk_corpus(corpus, chunk_size_words=250, chunk_overlap_words=50)

    # 2. Build Retrievers
    bge_model_id = "BAAI/bge-small-en-v1.5"
    dense_retriever = DenseVectorRetriever(model_name=bge_model_id, seed=SEED)
    dense_idx_time = dense_retriever.build_index(chunks)

    bm25_retriever = BM25LexicalRetriever(chunks)
    bm25_idx_time = bm25_retriever.indexing_time
    total_indexing_time = round(dense_idx_time + bm25_idx_time, 4)

    # 3. Reranker Model
    reranker_model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info(f"Loading CrossEncoder reranker: '{reranker_model_id}'...")
    reranker = CrossEncoder(reranker_model_id)

    # Fixed parameters
    K_INIT = 15
    K_FINAL = 5

    # Models to compare
    generators_to_test = [
        {
            "model_key": "flan_t5_base",
            "model_id": "google/flan-t5-base",
            "name": "google/flan-t5-base (250M Baseline)",
            "params_millions": 250
        },
        {
            "model_key": "flan_t5_large",
            "model_id": "google/flan-t5-large",
            "name": "google/flan-t5-large (780M Larger LLM)",
            "params_millions": 780
        }
    ]

    all_gen_summary = {}
    all_gen_per_question = {}

    for gen_spec in generators_to_test:
        gen_key = gen_spec["model_key"]
        gen_model_id = gen_spec["model_id"]
        logger.info("==================================================")
        logger.info(f"Evaluating Generator Model: {gen_spec['name']}...")

        # Initialize generator model
        generator = RAGGenerator(model_id=gen_model_id, seed=SEED)

        retrieval_latencies = []
        reranking_latencies = []
        generation_latencies = []
        total_online_latencies = []

        hits = 0
        total_recall = 0.0
        total_precision = 0.0
        correct_count = 0
        em_count = 0

        failure_counts = {
            "Retrieval Failure": 0,
            "Multi-Hop Reasoning Failure": 0,
            "Generation Failure": 0,
            "Gold-Answer / Data-Quality Issue": 0,
            "Insufficient Context": 0,
            "Answer-Format Mismatch": 0
        }

        per_q_records = []
        t_eval_start = time.time()

        for idx, q_obj in enumerate(queries):
            t_q_start = time.time()
            qid = str(q_obj["_id"])
            question = q_obj["text"]
            gold_ans = gold_answers.get(qid, "UNKNOWN")
            gold_doc_ids = qrels_map.get(qid, set())

            # 1. Hybrid Search (K_init = 15)
            candidate_chunks, r_lat = perform_hybrid_fusion(
                question, dense_retriever, bm25_retriever, top_k=K_INIT, alpha=0.5
            )
            retrieval_latencies.append(r_lat)

            # 2. Cross-Encoder Reranking over Candidate Pool
            t_rr_start = time.time()
            pair_inputs = [
                (question, f"{c['title']}: {c['text']}" if c.get("title") else c["text"])
                for c in candidate_chunks
            ]
            rr_scores = reranker.predict(pair_inputs)
            reranked_chunks = []
            for orig_c, score in zip(candidate_chunks, rr_scores):
                c_copy = dict(orig_c)
                c_copy["rerank_score"] = float(score)
                reranked_chunks.append(c_copy)
            reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            for r_idx, c in enumerate(reranked_chunks):
                c["reranked_rank"] = r_idx + 1

            rr_lat = (time.time() - t_rr_start) * 1000.0  # ms
            reranking_latencies.append(rr_lat)

            final_top5_chunks = reranked_chunks[:K_FINAL]

            # Compute retrieval metrics at K_final=5
            retrieved_doc_ids = set(c["doc_id"] for c in final_top5_chunks)
            rel_retrieved = retrieved_doc_ids.intersection(gold_doc_ids)
            num_rel = len(rel_retrieved)
            hit = num_rel > 0
            if hit:
                hits += 1
            rec = num_rel / len(gold_doc_ids) if len(gold_doc_ids) > 0 else 0.0
            prec = num_rel / len(final_top5_chunks) if len(final_top5_chunks) > 0 else 0.0
            total_recall += rec
            total_precision += prec

            # 3. Open-Book Generation with generator model
            gen_output = generator.generate_answer(
                question=question,
                retrieved_chunks=final_top5_chunks,
                max_new_tokens=50
            )
            rag_ans = gen_output["generated_answer"]
            gen_lat = gen_output["latency_seconds"] * 1000.0  # ms
            generation_latencies.append(gen_lat)

            total_q_lat = (time.time() - t_q_start) * 1000.0  # ms
            total_online_latencies.append(total_q_lat)

            eval_metrics = compute_correctness(rag_ans, gold_ans)
            if eval_metrics["is_correct"]:
                correct_count += 1
                failure_cat = None
            else:
                failure_cat = classify_failure(
                    query_idx=idx + 1,
                    query_id=qid,
                    question=question,
                    gold_answer=gold_ans,
                    rag_answer=rag_ans,
                    gold_doc_ids=gold_doc_ids,
                    retrieved_doc_ids=retrieved_doc_ids
                )
                failure_counts[failure_cat] += 1

            if eval_metrics["exact_match"]:
                em_count += 1

            q_record = {
                "query_index": idx + 1,
                "query_id": qid,
                "question": question,
                "gold_answer": gold_ans,
                "rag_generated_answer": rag_ans,
                "correctness_assessment": eval_metrics,
                "failure_category": failure_cat,
                "gold_doc_ids": list(gold_doc_ids),
                "retrieved_doc_ids": list(retrieved_doc_ids),
                "all_gold_retrieved": len(rel_retrieved) == len(gold_doc_ids),
                "online_latency_ms": {
                    "retrieval_ms": round(r_lat, 2),
                    "reranking_ms": round(rr_lat, 2),
                    "generation_ms": round(gen_lat, 2),
                    "total_ms": round(total_q_lat, 2)
                },
                "retrieved_chunks": [
                    {
                        "rank": c["reranked_rank"],
                        "doc_id": c["doc_id"],
                        "chunk_id": c["chunk_id"],
                        "title": c["title"],
                        "rerank_score": round(c["rerank_score"], 4),
                        "snippet": c["text"][:150] + "..."
                    }
                    for c in final_top5_chunks
                ]
            }
            per_q_records.append(q_record)

        n_queries = len(queries)
        eval_runtime_seconds = round(time.time() - t_eval_start, 2)
        gen_lat_summary = compute_percentiles(generation_latencies)
        total_lat_summary = compute_percentiles(total_online_latencies)

        gen_summary_entry = {
            "model_key": gen_key,
            "generator_model": gen_model_id,
            "name": gen_spec["name"],
            "parameters_millions": gen_spec["params_millions"],
            "k_final": K_FINAL,
            "k_init": K_INIT,
            "offline_cost": {
                "context_summarization_precomputation_seconds": 0.0,
                "index_creation_seconds": total_indexing_time,
                "dense_indexing_seconds": dense_idx_time,
                "bm25_indexing_seconds": bm25_idx_time
            },
            "retrieval_metrics": {
                "hit_rate_percent": round((hits / n_queries) * 100, 2),
                "recall_at_5": round(total_recall / n_queries, 4),
                "precision_at_5": round(total_precision / n_queries, 4)
            },
            "generation_metrics": {
                "correct_count": correct_count,
                "answer_accuracy_percent": round((correct_count / n_queries) * 100, 2),
                "exact_match_count": em_count,
                "exact_match_percent": round((em_count / n_queries) * 100, 2)
            },
            "failure_category_counts": failure_counts,
            "online_latency_summary_ms": {
                "retrieval_latency": compute_percentiles(retrieval_latencies),
                "reranking_latency": compute_percentiles(reranking_latencies),
                "generation_latency": gen_lat_summary,
                "total_online_latency": total_lat_summary
            },
            "total_evaluation_runtime_seconds": eval_runtime_seconds
        }

        all_gen_summary[gen_key] = gen_summary_entry
        all_gen_per_question[gen_key] = per_q_records

        logger.info(f"Completed {gen_spec['name']}: Accuracy={gen_summary_entry['generation_metrics']['answer_accuracy_percent']}%, EM={gen_summary_entry['generation_metrics']['exact_match_percent']}%, Gen Mean Latency={gen_lat_summary['mean_ms']}ms, RF={failure_counts['Retrieval Failure']}, MH={failure_counts['Multi-Hop Reasoning Failure']}")

    # Save JSON results
    results_json_path = os.path.join(output_dir, "generator_scaling_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_info": {
                "experiment_name": "Chapter 7 Generator Scaling Experiment",
                "finalized_retrieval_architecture": "Standard Chunks 250w/50o + BGE Dense + BM25 Hybrid (alpha=0.5) + Reranker (K_init=15 -> K_final=5)",
                "models_tested": [g["model_id"] for g in generators_to_test],
                "random_seed": SEED,
                "total_queries": len(queries),
                "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            },
            "summary_metrics": all_gen_summary,
            "per_question_results": all_gen_per_question
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON results to {results_json_path}")
    print(f"Total script runtime: {round(time.time() - t_total_start, 2)}s")

if __name__ == "__main__":
    main()
