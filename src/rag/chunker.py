"""
RAG Text Chunker Module
"""

from typing import List, Dict, Any

def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    chunk_size_words: int = 250,
    chunk_overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits a document into word-based chunks with configurable overlap.
    
    Args:
        doc_id: Document ID
        title: Document title
        text: Document text content
        chunk_size_words: Target word count per chunk
        chunk_overlap_words: Overlap word count between consecutive chunks
        
    Returns:
        List of chunk dictionaries containing chunk metadata and text.
    """
    words = text.split()
    if len(words) == 0:
        return [{
            "chunk_id": f"{doc_id}_c0",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": 0,
            "text": text,
            "word_count": 0
        }]
        
    chunks = []
    start = 0
    chunk_idx = 0
    step = max(1, chunk_size_words - chunk_overlap_words)
    
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{doc_id}_c{chunk_idx}",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": chunk_idx,
            "text": chunk_text,
            "word_count": len(words[start:end])
        })
        chunk_idx += 1
        start += step
        if end >= len(words):
            break
            
    return chunks

def chunk_corpus(
    corpus: List[Dict[str, Any]],
    chunk_size_words: int = 250,
    chunk_overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """Chunks an entire list of corpus document dictionaries."""
    all_chunks = []
    for doc in corpus:
        doc_id = str(doc["_id"])
        title = doc.get("title", "")
        text = doc.get("text", "")
        chunks = chunk_document(
            doc_id=doc_id,
            title=title,
            text=text,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words
        )
        all_chunks.extend(chunks)
    return all_chunks
