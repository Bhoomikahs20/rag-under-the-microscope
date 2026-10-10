"""
Controlled Top-K Retrieval Experiment Runner for RAG Baseline
Evaluates Top-K = 1, 3, 5, 10 on the 50-question linked HotpotQA pilot.
"""

import sys
import os
import json
import time
import re
import string
import torch
from typing import Dict, Any, List

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.chunker import chunk_corpus
from src.rag.retriever import FAISSRetriever, evaluate_retrieval_performance
from src.rag.generator import RAGGenerator
from src.utils.logging_config import setup_logger

logger = setup_logger("top_k_experiment_runner")

SEED = 42
torch.manual_seed(SEED)

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
    """
    Classifies failure using the established error-analysis methodology.
    """
    rel_retrieved = gold_doc_ids.intersection(retrieved_doc_ids)
    all_gold_retrieved = (len(rel_retrieved) == len(gold_doc_ids))
    
    # Check known dataset quality issue (Q13)
    if query_id == "5adf48b25542995ec70e8fb5":
        return "Gold-Answer / Data-Quality Issue"
        
    if not all_gold_retrieved:
        return "Retrieval Failure"
        
    # Check generation prefix repeat failure (e.g. "The relevant information to answer..." or long output prefix)
    if "The relevant information to answer the above question is" in rag_answer or len(rag_answer) > 100:
        return "Generation Failure"
        
    # Otherwise, both gold docs retrieved and model generated reasonable output, but reached wrong conclusion
    return "Multi-Hop Reasoning Failure"


def main():
    t_total_start = time.time()
    output_dir = "data/processed/rag_baseline/top_k_experiment"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading pilot datasets...")
    corpus_file = "data/processed/hotpotqa_pilot_corpus.json"
    queries_file = "data/processed/hotpotqa_pilot_queries.json"
    qrels_file = "data/processed/hotpotqa_pilot_qrels.json"

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    # Build qrels mapping query_id -> set of doc_ids
    qrels_map = {}
    for q in qrels:
        qid = str(q["query_id"])
        if qid not in qrels_map:
            qrels_map[qid] = set()
        qrels_map[qid].add(str(q["corpus_id"]))

    # Fetch gold answers
    from datasets import load_dataset
    hp_ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train", streaming=True)
    pilot_qids = set(q["_id"] for q in queries)
    gold_answers = {}
    for item in hp_ds:
        if item["id"] in pilot_qids:
            gold_answers[item["id"]] = item["answer"]
            if len(gold_answers) == len(pilot_qids):
                break

    # 1. Chunking Corpus
    chunk_size_words = 250
    chunk_overlap_words = 50
    chunks = chunk_corpus(corpus, chunk_size_words=chunk_size_words, chunk_overlap_words=chunk_overlap_words)

    # 2. Embedding & Vector Indexing
    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    retriever = FAISSRetriever(model_name=embedding_model_id, seed=SEED)
    indexing_time = retriever.build_index(chunks)

    # 3. Generator Model
    gen_model_id = "google/flan-t5-base"
    generator = RAGGenerator(model_id=gen_model_id, seed=SEED)

    top_k_values = [1, 3, 5, 10]
    experiment_results_by_k = {}
    k_summary_table = []

    for k in top_k_values:
        logger.info(f"--- Running Controlled RAG Experiment for Top-K = {k} ---")
        t_k_start = time.time()
        
        # Retrieval eval for this k
        retrieval_eval = evaluate_retrieval_performance(
            queries=queries,
            qrels=qrels,
            retriever=retriever,
            top_k=k
        )

        rag_results = []
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

        for idx, q_obj in enumerate(queries):
            qid = str(q_obj["_id"])
            question = q_obj["text"]
            gold_ans = gold_answers.get(qid, "UNKNOWN")

            retrieved_chunks = retriever.search(question, top_k=k)
            retrieved_doc_ids = set(c["doc_id"] for c in retrieved_chunks)
            gold_doc_ids = qrels_map.get(qid, set())

            gen_output = generator.generate_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                max_new_tokens=50
            )
            rag_ans = gen_output["generated_answer"]

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

            rec = {
                "query_index": idx + 1,
                "query_id": qid,
                "question": question,
                "gold_answer": gold_ans,
                "rag_generated_answer": rag_ans,
                "correctness_assessment": eval_metrics,
                "failure_category": failure_cat,
                "gold_doc_ids": list(gold_doc_ids),
                "retrieved_doc_ids": list(retrieved_doc_ids),
                "all_gold_retrieved": len(gold_doc_ids.intersection(retrieved_doc_ids)) == len(gold_doc_ids),
                "retrieved_chunks": [
                    {
                        "rank": c["rank"],
                        "doc_id": c["doc_id"],
                        "chunk_id": c["chunk_id"],
                        "title": c["title"],
                        "similarity_score": round(c["similarity_score"], 4),
                        "snippet": c["text"][:150] + "..."
                    }
                    for c in retrieved_chunks
                ],
                "latency_seconds": gen_output["latency_seconds"]
            }
            rag_results.append(rec)

        runtime_k = round(time.time() - t_k_start, 2)
        acc_pct = round((correct_count / len(queries)) * 100, 2)
        em_pct = round((em_count / len(queries)) * 100, 2)

        k_metrics = {
            "top_k": k,
            "correct_count": correct_count,
            "correct_rate_percent": acc_pct,
            "exact_match_rate_percent": em_pct,
            "retrieval_hit_rate_percent": retrieval_eval["hit_rate_percent"],
            "retrieval_mean_recall_at_k": retrieval_eval["mean_recall_at_k"],
            "retrieval_mean_precision_at_k": retrieval_eval["mean_precision_at_k"],
            "runtime_seconds": runtime_k,
            "failure_category_counts": failure_counts
        }

        experiment_results_by_k[f"top_{k}"] = {
            "metrics": k_metrics,
            "per_question_results": rag_results
        }

        k_summary_table.append(k_metrics)
        logger.info(f"Top-K={k}: HitRate={retrieval_eval['hit_rate_percent']}%, Recall={retrieval_eval['mean_recall_at_k']}, Precision={retrieval_eval['mean_precision_at_k']}, Accuracy={acc_pct}%, EM={em_pct}%, Time={runtime_k}s")

    # Save full JSON experiment results
    json_output_path = os.path.join(output_dir, "top_k_results.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_info": {
                "experiment_name": "Controlled Top-K Retrieval Experiment",
                "random_seed": SEED,
                "top_k_values_tested": top_k_values,
                "embedding_model": embedding_model_id,
                "generator_model": gen_model_id,
                "total_queries": len(queries),
                "total_experiment_runtime_seconds": round(time.time() - t_total_start, 2)
            },
            "k_summary": k_summary_table,
            "results_by_k": experiment_results_by_k
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Top-K Experiment completed! Saved results to {json_output_path}")

if __name__ == "__main__":
    main()
