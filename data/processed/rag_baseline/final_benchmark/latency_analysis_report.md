# Final Benchmark Latency Decomposition Analysis

**Protocol:** `PROTO_FINAL_BENCHMARK`  
**Dataset:** Test Set (N=40 Queries)  

---

## 1. Online Component Latency Breakdown

| Component | Mean Latency (ms) | P50 Median (ms) | P95 Latency (ms) | Percentage of Total Online Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Dense Vector Search (`bge-small-v1.5`)** | 45.59 ms | 44.38 ms | 58.43 ms | 0.6% |
| **Lexical Search (`BM25Okapi`)** | 1.07 ms | 0.92 ms | 1.62 ms | 0.0% |
| **Cross-Encoder Reranking (`MiniLM-L6`)**| 788.71 ms | 734.05 ms | 1083.9 ms | 10.8% |
| **LLM Generation (`flan-t5-large`)** | 6436.84 ms | 5965.8 ms | 8483.31 ms | 88.5% |
| **Total Online End-to-End** | **7272.35 ms** | **6787.3 ms** | **9487.72 ms** | **100.0%** |

---

## 2. Offline Precomputation & Indexing Costs

- **Dense Vector Embedding & Indexing:** 13.3178 s (100 corpus chunks embedded into FAISS Flat IP index).
- **BM25 Lexical Indexing:** 0.0088 s.
- **Contextual Summary Precomputation:** `0.0000 s` (Contextual document summaries omitted per audited ablation).
- **Total Offline Indexing Runtime:** 13.3266 s.
