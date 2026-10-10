import os
import json
import random
from typing import Dict, Any, List
from datasets import load_dataset_builder, load_dataset
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from src.utils.logging_config import setup_logger

logger = setup_logger("loader")

def inspect_schema(hf_path: str = "BeIR/hotpotqa", qrels_hf_path: str = "BeIR/hotpotqa-qrels") -> Dict[str, Any]:
    """
    Inspects Hugging Face dataset schemas for corpus, queries, and qrels.
    """
    schema_info = {}
    
    # Corpus schema
    logger.info(f"Inspecting schema for '{hf_path}' (corpus)...")
    corpus_builder = load_dataset_builder(hf_path, "corpus")
    corpus_ds = load_dataset(hf_path, "corpus", streaming=True)
    sample_corpus = next(iter(corpus_ds["corpus"]))
    schema_info["corpus"] = {
        "hf_repo": hf_path,
        "subset_name": "corpus",
        "split": "corpus",
        "features": {k: str(v) for k, v in corpus_builder.info.features.items()},
        "sample": sample_corpus
    }
    
    # Queries schema
    logger.info(f"Inspecting schema for '{hf_path}' (queries)...")
    queries_builder = load_dataset_builder(hf_path, "queries")
    queries_ds = load_dataset(hf_path, "queries", streaming=True)
    sample_query = next(iter(queries_ds["queries"]))
    schema_info["queries"] = {
        "hf_repo": hf_path,
        "subset_name": "queries",
        "split": "queries",
        "features": {k: str(v) for k, v in queries_builder.info.features.items()},
        "sample": sample_query
    }
    
    # Qrels schema
    logger.info(f"Inspecting schema for '{qrels_hf_path}' (qrels)...")
    qrels_builder = load_dataset_builder(qrels_hf_path)
    qrels_ds = load_dataset(qrels_hf_path, streaming=True)
    sample_qrel = next(iter(qrels_ds["train"]))
    schema_info["qrels"] = {
        "hf_repo": qrels_hf_path,
        "subset_name": "qrels",
        "split": "train",
        "features": {k: str(v) for k, v in qrels_builder.info.features.items()},
        "sample": sample_qrel
    }
    
    return schema_info


def load_linked_pilot_subset(
    hf_path: str = "BeIR/hotpotqa",
    qrels_hf_path: str = "BeIR/hotpotqa-qrels",
    queries_count: int = 50,
    seed: int = 42,
    output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Creates a linked pilot dataset where every selected query has 100% of its ground-truth 
    relevant documents included in the pilot corpus.
    
    Args:
        hf_path: Hugging Face repo for hotpotqa corpus and queries
        qrels_hf_path: Hugging Face repo for hotpotqa qrels
        queries_count: Number of queries to sample
        seed: Random seed for reproducibility
        output_dir: Destination directory for JSON output files
        
    Returns:
        Dict with pilot stats and generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Creating linked pilot dataset with seed={seed} (Target queries: {queries_count})...")
    
    # 1. Sample Queries Reproducibly
    logger.info(f"Streaming queries from '{hf_path}' (queries split)...")
    queries_stream = load_dataset(hf_path, "queries", split="queries", streaming=True)
    
    # Stream candidates (first 1000 queries) and sample reproducibly
    candidate_queries = list(iter(queries_stream.take(1000)))
    random.seed(seed)
    sampled_queries = random.sample(candidate_queries, queries_count)
    selected_qids = [q["_id"] for q in sampled_queries]
    selected_qid_set = set(selected_qids)
    logger.info(f"Selected {len(sampled_queries)} queries reproducibly.")

    # 2. Extract Relevance Relationships (qrels)
    logger.info(f"Loading relevance labels from '{qrels_hf_path}' (train split)...")
    qrels_ds = load_dataset(qrels_hf_path, split="train")
    
    relevant_qrels = []
    target_corpus_ids = set()
    
    for row in qrels_ds:
        qid = row["query-id"]
        if qid in selected_qid_set:
            cid = str(row["corpus-id"])
            score = int(row["score"])
            relevant_qrels.append({
                "query_id": qid,
                "corpus_id": cid,
                "score": score
            })
            target_corpus_ids.add(cid)
            
    logger.info(f"Extracted {len(relevant_qrels)} relevance links across {len(target_corpus_ids)} unique corpus documents.")

    # 3. Retrieve Exact Linked Corpus Documents
    logger.info(f"Retrieving target {len(target_corpus_ids)} corpus documents via PyArrow...")
    parquet_path = hf_hub_download(
        repo_id=hf_path,
        filename="corpus/corpus-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    pf = pq.ParquetFile(parquet_path)
    found_docs = {}
    
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        df["_id"] = df["_id"].astype(str)
        matched = df[df["_id"].isin(target_corpus_ids)]
        for _, doc_row in matched.iterrows():
            found_docs[doc_row["_id"]] = {
                "_id": doc_row["_id"],
                "title": doc_row.get("title", ""),
                "text": doc_row.get("text", "")
            }
        if len(found_docs) == len(target_corpus_ids):
            break
            
    logger.info(f"Successfully retrieved all {len(found_docs)} linked corpus documents!")

    # 4. Save Processed Files
    queries_file = os.path.join(output_dir, "hotpotqa_pilot_queries.json")
    corpus_file = os.path.join(output_dir, "hotpotqa_pilot_corpus.json")
    qrels_file = os.path.join(output_dir, "hotpotqa_pilot_qrels.json")
    
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(sampled_queries, f, indent=2, ensure_ascii=False)
        
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(list(found_docs.values()), f, indent=2, ensure_ascii=False)
        
    with open(qrels_file, "w", encoding="utf-8") as f:
        json.dump(relevant_qrels, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved pilot queries to '{queries_file}'")
    logger.info(f"Saved pilot corpus to '{corpus_file}'")
    logger.info(f"Saved pilot qrels to '{qrels_file}'")
    
    return {
        "queries_count": len(sampled_queries),
        "corpus_count": len(found_docs),
        "qrels_count": len(relevant_qrels),
        "queries_file": queries_file,
        "corpus_file": corpus_file,
        "qrels_file": qrels_file,
        "selected_qids": selected_qids,
        "selected_cids": list(found_docs.keys()),
        "seed": seed
    }
