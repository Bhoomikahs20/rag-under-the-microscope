# Final RAG Benchmark Report (`PROTO_FINAL_BENCHMARK`)

**Protocol ID:** `PROTO_FINAL_BENCHMARK`  
**Evaluation Date:** 2026-10-07  
**Dataset Split:** Leakage-Safe Split (10 Dev Queries / 40 Held-out Test Queries)  
**Primary Result File:** [`final_benchmark_results.json`](file:///D:/Rag book work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json)  

---

## 1. Executive Benchmark Summary

This report documents the final benchmark execution of the frozen RAG architecture on the held-out **40-query Test Set** under protocol `PROTO_FINAL_BENCHMARK`. All hyperparameters, decoding strategies, prompt templates, and random seeds ($42$) were frozen prior to evaluation.

### Authoritative Benchmark Results (Test Set N=40):
- **Answer Accuracy:** **90.0%** (95% CI: [80.0%, 97.5%])
- **Exact Match:** **85.0%** (95% CI: [72.5%, 95.0%])
- **Hit Rate @ 5:** **100.0%** (95% CI: [100.0%, 100.0%])
- **Recall @ 5:** **1.0** (95% CI: [1.0, 1.0])
- **Precision @ 5:** **0.4** (95% CI: [0.4, 0.4])
- **Total Online Latency (Mean / P50 / P95):** **7272.35 ms / 6787.3 ms / 9487.72 ms**

---

## 2. Frozen Architecture & Protocol Specifications

| Pipeline Stage | Selection / Parameter | Specification |
| :--- | :--- | :--- |
| **Document Chunking** | 250w / 50o Standard Chunks | Title + Text format (No contextual document summaries) |
| **Dense Retriever** | `BAAI/bge-small-en-v1.5` | 384-d, Cosine similarity, FAISS Flat IP Index |
| **Lexical Retriever** | `BM25Okapi` | rank_bm25 over lowercased tokens |
| **Score Fusion** | Min-Max Normalized Hybrid Fusion ($lpha=0.5$) | Candidate pool $K_{\text{init}} = 15$ |
| **Reranker Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranked context window $K_{\text{final}} = 5$ |
| **Generator LLM** | `google/flan-t5-large` (780M) | Greedy decoding ($T=0.0$, `max_new_tokens=50`, seed $42$) |

---

## 3. Comprehensive Performance Metrics & 95% Confidence Intervals

| Metric Category | Metric Name | Test Set Result (N=40) | Empirical 95% Bootstrap CI | Dev Set Result (N=10) |
| :--- | :--- | :---: | :---: | :---: |
| **Accuracy** | Answer Accuracy (%) | **90.0%** | [80.0%, 97.5%] | 100.0% |
| | Exact Match (%) | **85.0%** | [72.5%, 95.0%] | 80.0% |
| **Retrieval** | Hit Rate @ 5 (%) | **100.0%** | [100.0%, 100.0%] | 100.0% |
| | Recall @ 5 | **1.0** | [1.0, 1.0] | 1.0 |
| | Precision @ 5 | **0.4** | [0.4, 0.4] | 0.4000 |
| **Latency** | Generation Mean Latency | **6436.84 ms** | [5959.18 ms, 7021.75 ms] | - |
| | Total Online Mean Latency | **7272.35 ms** | [6768.42 ms, 7874.62 ms] | - |

---

## 4. Failure Category Breakdown (Test Set N=40)

| Failure Category | Count | Percentage | Primary Root Cause & Description |
| :--- | :---: | :---: | :--- |
| **Retrieval Failure (RF)** | 0 | 0.0% | Target gold chunk missing from top-5 context |
| **Multi-Hop Reasoning Failure (MH)** | 3 | 7.5% | Gold chunks retrieved, but LLM failed complex multi-step reasoning |
| **Generation Failure (GF)** | 0 | 0.0% | Prompt boilerplate echo or formatting failure |
| **Insufficient Context (IC)** | 0 | 0.0% | Required facts missing from corpus |
| **Gold-Answer / Data-Quality Issue (DQ)**| 1 | 2.5% | Dataset label annotation defect |
| **Total Failures** | **4** | **10.0%** | Overall error rate on Test set |

---

## 5. Latency & Compute Breakdown

- **Dense Retrieval (`bge-small-en-v1.5`):** Mean 45.59 ms (P50: 44.38 ms, P95: 58.43 ms)
- **BM25 Retrieval:** Mean 1.07 ms (P50: 0.92 ms, P95: 1.62 ms)
- **Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`):** Mean 788.71 ms (P50: 734.05 ms, P95: 1083.9 ms)
- **LLM Generation (`flan-t5-large`):** Mean 6436.84 ms (P50: 5965.8 ms, P95: 8483.31 ms)
- **Total Online End-to-End Latency:** Mean 7272.35 ms (P50: 6787.3 ms, P95: 9487.72 ms)
- **Offline Indexing Time:** Total 13.3266 s (Dense: 13.3178 s, BM25: 0.0088 s)

---

## 6. Output Artifact Integrity

All raw evaluation records and reports have been generated under `data/processed/rag_baseline/final_benchmark/`:
- `final_benchmark_results.json`
- `final_benchmark_report.md`
- `per_question_results.jsonl`
- `latency_analysis_report.md`
- `failure_analysis_report.md`
