import os
import json
from datetime import datetime
from typing import Dict, Any
from src.utils.logging_config import setup_logger

logger = setup_logger("manifest")

def create_dataset_manifest(
    hf_path: str = "BeIR/hotpotqa",
    qrels_hf_path: str = "BeIR/hotpotqa-qrels",
    license_name: str = "CC BY-SA 4.0",
    schema_info: Dict[str, Any] = None,
    pilot_stats: Dict[str, Any] = None,
    output_path: str = "data/manifest.json"
) -> Dict[str, Any]:
    """
    Creates a dataset manifest recording sources, licenses, subsets, splits, fields,
    pilot size, reproducible seed, and query-document relevance relationships.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load qrels mapping to store inside manifest as well
    qrels_file = pilot_stats.get("qrels_file") if pilot_stats else ""
    relevance_pairs = []
    if qrels_file and os.path.exists(qrels_file):
        with open(qrels_file, "r", encoding="utf-8") as f:
            relevance_pairs = json.load(f)
            
    manifest_data = {
        "project": "RAG Book Research - Linked HotpotQA Pilot",
        "dataset_metadata": {
            "source_corpus_and_queries": f"https://huggingface.co/datasets/{hf_path}",
            "source_qrels": f"https://huggingface.co/datasets/{qrels_hf_path}",
            "hf_dataset_ids": {
                "corpus_and_queries": hf_path,
                "qrels": qrels_hf_path
            },
            "original_benchmark": "BeIR (Benchmarking Information Retrieval)",
            "underlying_dataset": "HotpotQA (Multi-hop Question Answering)",
            "license": license_name,
            "reproducible_seed": pilot_stats.get("seed", 42) if pilot_stats else 42,
            "sampling_method": "Linked sampling via qrels (100% query-document ground truth coverage)",
            "full_corpus_note": "Full corpus contains ~5.23M Wikipedia paragraphs; pilot subset extracted reproducibly."
        },
        "subsets": {
            "queries": {
                "subset_name": "queries",
                "split": "queries",
                "fields": {
                    "_id": "string (unique query ID)",
                    "title": "string (optional title or empty string)",
                    "text": "string (multi-hop question text)"
                },
                "pilot_sample_count": pilot_stats.get("queries_count", 0) if pilot_stats else 0,
                "local_filepath": pilot_stats.get("queries_file", "") if pilot_stats else ""
            },
            "corpus": {
                "subset_name": "corpus",
                "split": "corpus",
                "fields": {
                    "_id": "string (unique document ID)",
                    "title": "string (Wikipedia article title)",
                    "text": "string (paragraph content)"
                },
                "pilot_sample_count": pilot_stats.get("corpus_count", 0) if pilot_stats else 0,
                "local_filepath": pilot_stats.get("corpus_file", "") if pilot_stats else ""
            },
            "qrels": {
                "subset_name": "qrels",
                "split": "train",
                "fields": {
                    "query_id": "string (matching query _id)",
                    "corpus_id": "string (matching document _id)",
                    "score": "integer (relevance label, 1 = relevant)"
                },
                "pilot_sample_count": pilot_stats.get("qrels_count", 0) if pilot_stats else 0,
                "local_filepath": pilot_stats.get("qrels_file", "") if pilot_stats else ""
            }
        },
        "linked_pilot_summary": {
            "queries_count": pilot_stats.get("queries_count", 0) if pilot_stats else 0,
            "corpus_documents_count": pilot_stats.get("corpus_count", 0) if pilot_stats else 0,
            "relevance_links_count": len(relevance_pairs),
            "coverage": "100% of selected queries have all relevant documents present in the pilot corpus",
            "query_ids": pilot_stats.get("selected_qids", []) if pilot_stats else [],
            "corpus_ids": pilot_stats.get("selected_cids", []) if pilot_stats else [],
            "relevance_relationships": relevance_pairs
        },
        "created_at": datetime.now().isoformat(),
        "manifest_version": "2.0"
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Updated dataset manifest saved to '{output_path}'")
    return manifest_data


def verify_pilot_dataset(manifest_path: str = "data/manifest.json") -> bool:
    """
    Verifies local pilot dataset files and validates query-document relevance links.
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found at '{manifest_path}'")
        return False
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    queries_file = manifest["subsets"]["queries"]["local_filepath"]
    corpus_file = manifest["subsets"]["corpus"]["local_filepath"]
    qrels_file = manifest["subsets"]["qrels"]["local_filepath"]
    
    for path_key, filepath in [("queries", queries_file), ("corpus", corpus_file), ("qrels", qrels_file)]:
        if not os.path.exists(filepath):
            logger.error(f"File missing for {path_key}: '{filepath}'")
            return False
            
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = json.load(f)
        
    query_id_set = set(q["_id"] for q in queries)
    corpus_id_set = set(doc["_id"] for doc in corpus)
    
    # Check that every qrel refers to a query and document present in the pilot!
    missing_queries = []
    missing_docs = []
    
    for link in qrels:
        qid = link["query_id"]
        cid = link["corpus_id"]
        if qid not in query_id_set:
            missing_queries.append(qid)
        if cid not in corpus_id_set:
            missing_docs.append(cid)
            
    assert len(missing_queries) == 0, f"Qrels refer to queries missing from pilot: {missing_queries}"
    assert len(missing_docs) == 0, f"Qrels refer to corpus docs missing from pilot: {missing_docs}"
    
    logger.info("=== Linked Pilot Dataset Verification Succeeded ===")
    logger.info(f"Verified {len(queries)} queries, {len(corpus)} corpus documents, and {len(qrels)} relevance links.")
    logger.info(f"Coverage: 100% of relevance links point to documents present in pilot corpus.")
    return True
