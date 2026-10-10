"""
FAISS Vector Retriever Module
"""

import time
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.utils.logging_config import setup_logger

logger = setup_logger("retriever")

class FAISSRetriever:
    """FAISS-based Dense Vector Retriever."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        logger.info(f"Initializing SentenceTransformer model: '{model_name}'...")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
    def build_index(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Encodes chunk texts and builds a FAISS Inner Product (Cosine Similarity) index.
        
        Args:
            chunks: List of chunk metadata dictionaries
            
        Returns:
            Indexing time in seconds.
        """
        t0 = time.time()
        self.chunks = chunks
        
        # Format text with title prefix for richer context embeddings
        formatted_texts = [
            f"{c['title']}: {c['text']}" if c.get("title") else c["text"]
            for c in chunks
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
        
        elapsed = round(time.time() - t0, 3)
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors in {elapsed}s.")
        return elapsed

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches FAISS index for top_k relevant chunks.
        
        Args:
            query: Question text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunk dicts with added 'similarity_score' field.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built yet. Call build_index first.")
            
        q_embed = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        
        scores, indices = self.index.search(q_embed, top_k)
        
        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["similarity_score"] = float(score)
            chunk_copy["rank"] = rank + 1
            retrieved.append(chunk_copy)
            
        return retrieved


def evaluate_retrieval_performance(
    queries: List[Dict[str, Any]],
    qrels: List[Dict[str, Any]],
    retriever: FAISSRetriever,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates retrieval Recall@k, Precision@k, and Hit Rate against ground-truth qrels.
    """
    query_to_gold_docs = {}
    for link in qrels:
        qid = str(link["query_id"])
        cid = str(link["corpus_id"])
        if qid not in query_to_gold_docs:
            query_to_gold_docs[qid] = set()
        query_to_gold_docs[qid].add(cid)
        
    hits = 0
    total_recall_sum = 0.0
    total_precision_sum = 0.0
    
    per_query_retrieval_stats = []
    
    for q in queries:
        qid = str(q["_id"])
        qtext = q["text"]
        gold_doc_ids = query_to_gold_docs.get(qid, set())
        
        retrieved_chunks = retriever.search(qtext, top_k=top_k)
        retrieved_doc_ids = set(c["doc_id"] for c in retrieved_chunks)
        
        relevant_retrieved = retrieved_doc_ids.intersection(gold_doc_ids)
        num_rel_retrieved = len(relevant_retrieved)
        
        hit = num_rel_retrieved > 0
        if hit:
            hits += 1
            
        recall = num_rel_retrieved / len(gold_doc_ids) if len(gold_doc_ids) > 0 else 0.0
        precision = num_rel_retrieved / len(retrieved_doc_ids) if len(retrieved_doc_ids) > 0 else 0.0
        
        total_recall_sum += recall
        total_precision_sum += precision
        
        per_query_retrieval_stats.append({
            "query_id": qid,
            "gold_doc_ids": list(gold_doc_ids),
            "retrieved_doc_ids": list(retrieved_doc_ids),
            "relevant_retrieved_ids": list(relevant_retrieved),
            "hit": hit,
            "recall": round(recall, 4),
            "precision": round(precision, 4)
        })
        
    n_queries = len(queries)
    return {
        "top_k": top_k,
        "hit_rate_percent": round((hits / n_queries) * 100, 2),
        "mean_recall_at_k": round(total_recall_sum / n_queries, 4),
        "mean_precision_at_k": round(total_precision_sum / n_queries, 4),
        "per_query_retrieval_stats": per_query_retrieval_stats
    }
