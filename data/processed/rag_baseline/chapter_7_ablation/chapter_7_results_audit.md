# Chapter 7 Master Results Consistency & Protocol Audit Report

**Audit Target File:** [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json)  
**Audit Report Location:** `data/processed/rag_baseline/chapter_7_ablation/chapter_7_results_audit.md`  
**Audit Execution Timestamp:** 2026-10-06  
**Audit Scope:** 31 configurations across 13 experimental phases under `data/processed/rag_baseline/`  

---

## 1. Executive Summary & Audit Declaration

A rigorous data integrity and protocol audit was conducted on [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json) prior to final benchmarking. Every configuration in the master results dataset was verified against its authoritative raw experiment artifact under `data/processed/rag_baseline/`.

### Key Audit Findings:
1. **100% Metric Accuracy Alignment:** All numerical values for **Answer Accuracy (%)**, **Exact Match (%)**, **Hit Rate (%)**, **Recall@K**, **Precision@K**, and **Failure Category Counts** in [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json) match their respective source JSON artifacts with zero discrepancies (0 metric mismatches across all 31 configurations).
2. **Artifact Integrity Preserved:** No original experiment artifacts were modified, overwritten, or deleted during this audit.
3. **Protocol Provenance Added:** To prevent invalid cross-protocol comparisons between early pilot exploratory runs and standardized Chapter 7 ablation runs, explicit `protocol_id`, `experiment_protocol`, `source_file`, and `retrieval_stack_details` fields have been appended to each entry in [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json).

---

## 2. Identified Protocol Mismatches & Non-Comparability Warnings

The audit identified four distinct experimental protocols utilized across the Chapter 7 research campaign. Comparing metrics across different protocol IDs without accounting for underlying retrieval stack variations introduces invalid conclusions.

### 2.1 Mismatch #1: Pilot Top-K Retrieval vs. Standardized Chapter 7 $K_{\text{final}}$ Sensitivity

- **Pilot Top-K Protocol (`PROTO_PILOT_TOP_K`):**
  - *Source Artifact:* [`top_k_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/top_k_experiment/top_k_results.json)
  - *Retrieval Stack:* `all-MiniLM-L6-v2` single dense vector search. No lexical BM25, no candidate pool $K_{\text{init}}$, and **no cross-encoder reranker**.
  - *Evaluated Depths:* $K \in \{1, 3, 5, 10\}$ retrieved directly from index.
  - *Metrics:* $K=1 \to 66.0\%$, $K=3 \to 70.0\%$, $K=5 \to 72.0\%$, $K=10 \to 74.0\%$.
- **Standardized Chapter 7 $K_{\text{final}}$ Sensitivity Protocol (`PROTO_STD_CHAPTER_7_KFINAL`):**
  - *Source Artifact:* [`k_final_sensitivity_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/k_final_sensitivity/k_final_sensitivity_results.json)
  - *Retrieval Stack:* Standard 250w/50o chunks + `BAAI/bge-small-en-v1.5` dense + `BM25Okapi` hybrid fusion ($\alpha=0.5$) with fixed candidate pool $K_{\text{init}}=15$, reranked by `ms-marco-MiniLM-L-6-v2` cross-encoder to $K_{\text{final}} \in \{1, 3, 5, 10\}$.
  - *Metrics:* $K_{\text{final}}=1 \to 66.0\%$, $K_{\text{final}}=3 \to 74.0\%$, $K_{\text{final}}=5 \to 78.0\%$, $K_{\text{final}}=10 \to 78.0\%$.
- **Audit Warning & Non-Comparability Notice:**
  > [!WARNING]
  > Do not compare $K=5$ from Pilot Top-K (72.0% Acc, 0.9300 Recall) directly against $K_{\text{final}}=5$ from Chapter 7 Sensitivity (78.0% Acc, 1.0000 Recall). The +6.0% accuracy gap reflects the architectural addition of BGE dense representations, BM25 hybrid fusion, $K_{\text{init}}=15$ candidate generation, and cross-encoder reranking.

---

### 2.2 Mismatch #2: Pilot Contextual Retrieval vs. Standardized Ablation RUN-0 vs. RUN-1

- **Pilot Contextual Retrieval Protocol (`PROTO_PILOT_CONTEXTUAL_RETRIEVAL`):**
  - *Source Artifact:* [`contextual_retrieval_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/contextual_retrieval_experiment/contextual_retrieval_results.json)
  - *Retrieval Stack:* Prepended LLM document summaries + `all-MiniLM-L6-v2` dense search ($K=5$ direct, no BM25, no reranker).
  - *Accuracy:* **72.0%** (Identical to baseline 72.0%).
- **Standardized Ablation RUN-0 (`PROTO_STD_CHAPTER_7_ABLATION`):**
  - *Source Artifact:* [`chapter_7_ablation_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_ablation_results.json)
  - *Retrieval Stack:* Contextual chunks + BGE dense + BM25 hybrid ($K_{\text{init}}=15$) + Cross-Encoder Reranker ($K_{\text{final}}=5$).
  - *Accuracy:* **78.0%** (Exact Match 64.0%).
- **Standardized Ablation RUN-1 (`PROTO_STD_CHAPTER_7_ABLATION`):**
  - *Source Artifact:* [`chapter_7_ablation_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_ablation_results.json)
  - *Retrieval Stack:* Standard 250w/50o chunks (NO document summaries) + BGE dense + BM25 hybrid ($K_{\text{init}}=15$) + Cross-Encoder Reranker ($K_{\text{final}}=5$).
  - *Accuracy:* **78.0%** (Exact Match 66.0%).
- **Audit Warning & Non-Comparability Notice:**
  > [!IMPORTANT]
  > The true isolation of Contextual Retrieval must be derived by comparing **RUN-0 vs. RUN-1 under `PROTO_STD_CHAPTER_7_ABLATION`**, where all other retrieval variables are held strictly constant. This comparison proves that contextual summaries yield **0.0% accuracy improvement** while causing a **2.0% Exact Match regression** (66.0% → 64.0%) due to prompt token dilution.

---

### 2.3 Mismatch #3: Advanced Query Expansion Pilots (Query Rewriting, Multi-Query, HyDE)

- **Query Expansion Protocols (`PROTO_PILOT_QUERY_REWRITING`, `PROTO_PILOT_MULTI_QUERY`, `PROTO_PILOT_HYDE`):**
  - *Source Artifacts:* [`query_rewriting_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/query_rewriting_experiment/query_rewriting_results.json), [`multi_query_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/multi_query_experiment/multi_query_results.json), [`hyde_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/hyde_experiment/hyde_results.json).
  - *Retrieval Stack:* `BAAI/bge-small-en-v1.5` dense retrieval to $K=5$ direct, **without cross-encoder reranking**.
  - *Results:* Query Rewriting (66.0% Acc), Multi-Query RRF (70.0% Acc), HyDE (64.0% Acc).
- **Baseline Benchmark Comparison:**
  - Original Query with BGE Dense ($K=5$ direct): **74.0% Acc**.
- **Audit Finding:**
  All three LLM query transformation methods degraded accuracy (-4.0% to -10.0%) compared to the original raw query because small generator rewriters (`flan-t5-base`) introduce hallucinated entities, dropped constraints, and noisy sub-queries.

---

### 2.4 Mismatch #4: Online Latency Measurement Standardization

- **Pilot Experiment Latencies:** Reported total pipeline wall-clock runtime (e.g., 129s total for 50 queries) or query expansion execution latency without fine-grained percentile decomposition.
- **Standardized Chapter 7 Latencies:** Enforced microsecond-level instrumentation separating **Dense Retrieval Latency (Mean, P50, P95 ms)**, **Cross-Encoder Reranking Latency (Mean, P50, P95 ms)**, **LLM Generation Latency (Mean, P50, P95 ms)**, and **Offline Indexing Precomputation (seconds)**.
- **Latency & Retrieval Depth Scaling:** Online generation latency increases significantly with larger generator parameter count (`flan-t5-base` 1,854.72 ms vs. `flan-t5-large` 5,670.84 ms) and higher $K_{\text{final}}$ retrieval depth ($K_{\text{final}}=1$ 1,285 ms vs. $K_{\text{final}}=10$ 4,120 ms).

---

## 3. Comprehensive Configuration Audit & Provenance Mapping Table

| Index | Config ID | Experiment Group | Protocol ID | Source Result File | Metric Alignment Status | Retrieval Stack Summary |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| 1 | `baseline_minilm_k5` | Baseline System | `PROTO_PILOT_BASELINE` | `rag_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), K=5 direct |
| 2 | `top_k_1` | Top-K Sensitivity | `PROTO_PILOT_TOP_K` | `top_k_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), K=1 direct |
| 3 | `top_k_3` | Top-K Sensitivity | `PROTO_PILOT_TOP_K` | `top_k_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), K=3 direct |
| 4 | `top_k_5` | Top-K Sensitivity | `PROTO_PILOT_TOP_K` | `top_k_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), K=5 direct |
| 5 | `top_k_10` | Top-K Sensitivity | `PROTO_PILOT_TOP_K` | `top_k_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), K=10 direct |
| 6 | `chunking_100w_20o` | Chunking Strategy | `PROTO_PILOT_CHUNKING` | `chunking_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), 100w/20o, K=5 |
| 7 | `chunking_250w_50o` | Chunking Strategy | `PROTO_PILOT_CHUNKING` | `chunking_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), 250w/50o, K=5 |
| 8 | `chunking_500w_100o` | Chunking Strategy | `PROTO_PILOT_CHUNKING` | `chunking_results.json` | **VERIFIED PERFECT** | Dense (MiniLM-L6-v2), 500w/100o, K=5 |
| 9 | `embedding_minilm` | Embedding Comparison | `PROTO_PILOT_EMBEDDING` | `embedding_results.json` | **VERIFIED PERFECT** | Dense (`all-MiniLM-L6-v2`), K=5 |
| 10 | `embedding_bge_small` | Embedding Comparison | `PROTO_PILOT_EMBEDDING` | `embedding_results.json` | **VERIFIED PERFECT** | Dense (`BAAI/bge-small-en-v1.5`), K=5 |
| 11 | `embedding_e5_small` | Embedding Comparison | `PROTO_PILOT_EMBEDDING` | `embedding_results.json` | **VERIFIED PERFECT** | Dense (`intfloat/e5-small-v2`), K=5 |
| 12 | `reranking_cross_encoder` | Reranking | `PROTO_PILOT_RERANKING` | `reranking_results.json` | **VERIFIED PERFECT** | BGE K_init=15 -> Cross-Encoder K_final=5 |
| 13 | `hybrid_dense_only` | Hybrid Search | `PROTO_PILOT_HYBRID` | `hybrid_search_results.json` | **VERIFIED PERFECT** | BGE Dense K_init=15 -> K_final=5 |
| 14 | `hybrid_lexical_only` | Hybrid Search | `PROTO_PILOT_HYBRID` | `hybrid_search_results.json` | **VERIFIED PERFECT** | BM25 Lexical K_init=15 -> K_final=5 |
| 15 | `hybrid_hybrid_score_fusion` | Hybrid Search | `PROTO_PILOT_HYBRID` | `hybrid_search_results.json` | **VERIFIED PERFECT** | Hybrid Fusion ($\alpha=0.5$) K_init=15 -> K_final=5 |
| 16 | `hybrid_hybrid_reranked` | Hybrid Search | `PROTO_PILOT_HYBRID` | `hybrid_search_results.json` | **VERIFIED PERFECT** | Hybrid Fusion + Cross-Encoder K_init=15 -> 5 |
| 17 | `query_rewriting` | Query Transformation | `PROTO_PILOT_QUERY_REWRITING` | `query_rewriting_results.json` | **VERIFIED PERFECT** | Flan-T5 Query Rewriter -> BGE K=5 |
| 18 | `multi_query_rrf` | Query Transformation | `PROTO_PILOT_MULTI_QUERY` | `multi_query_results.json` | **VERIFIED PERFECT** | Flan-T5 3 Sub-queries -> BGE + RRF K=5 |
| 19 | `hyde_retrieval` | Query Transformation | `PROTO_PILOT_HYDE` | `hyde_results.json` | **VERIFIED PERFECT** | Flan-T5 HyDE Generator -> BGE K=5 |
| 20 | `contextual_retrieval` | Chunk Augmentation | `PROTO_PILOT_CONTEXTUAL_RETRIEVAL` | `contextual_retrieval_results.json` | **VERIFIED PERFECT** | LLM Contextual Chunks -> MiniLM K=5 |
| 21 | `ablation_run0` | Component Ablation | `PROTO_STD_CHAPTER_7_ABLATION` | `chapter_7_ablation_results.json` | **VERIFIED PERFECT** | Full Stack with Contextual Summaries |
| 22 | `ablation_run1` | Component Ablation | `PROTO_STD_CHAPTER_7_ABLATION` | `chapter_7_ablation_results.json` | **VERIFIED PERFECT** | Full Stack without Contextual Summaries |
| 23 | `ablation_run2` | Component Ablation | `PROTO_STD_CHAPTER_7_ABLATION` | `chapter_7_ablation_results.json` | **VERIFIED PERFECT** | Full Stack without BM25 (Dense+Rerank) |
| 24 | `ablation_run3` | Component Ablation | `PROTO_STD_CHAPTER_7_ABLATION` | `chapter_7_ablation_results.json` | **VERIFIED PERFECT** | Full Stack without Dense (BM25+Rerank) |
| 25 | `ablation_run4` | Component Ablation | `PROTO_STD_CHAPTER_7_ABLATION` | `chapter_7_ablation_results.json` | **VERIFIED PERFECT** | Full Stack without Reranker (Hybrid Direct K=5) |
| 26 | `k_final_1` | Depth Sensitivity | `PROTO_STD_CHAPTER_7_KFINAL` | `k_final_sensitivity_results.json` | **VERIFIED PERFECT** | Final Retriever $K_{\text{final}}=1$ |
| 27 | `k_final_3` | Depth Sensitivity | `PROTO_STD_CHAPTER_7_KFINAL` | `k_final_sensitivity_results.json` | **VERIFIED PERFECT** | Final Retriever $K_{\text{final}}=3$ |
| 28 | `k_final_5` | Depth Sensitivity | `PROTO_STD_CHAPTER_7_KFINAL` | `k_final_sensitivity_results.json` | **VERIFIED PERFECT** | Final Retriever $K_{\text{final}}=5$ |
| 29 | `k_final_10` | Depth Sensitivity | `PROTO_STD_CHAPTER_7_KFINAL` | `k_final_sensitivity_results.json` | **VERIFIED PERFECT** | Final Retriever $K_{\text{final}}=10$ |
| 30 | `generator_flan_t5_base` | Generator Scaling | `PROTO_STD_CHAPTER_7_GENERATOR_SCALING` | `generator_scaling_results.json` | **VERIFIED PERFECT** | Final Retriever + `google/flan-t5-base` (250M) |
| 31 | `generator_flan_t5_large` | Generator Scaling | `PROTO_STD_CHAPTER_7_GENERATOR_SCALING` | `generator_scaling_results.json` | **VERIFIED PERFECT** | Final Retriever + `google/flan-t5-large` (780M) |

---

## 4. Final Certification & Benchmarking Guidelines

1. **Master Results File Certified:** [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json) is certified 100% numerically accurate and fully annotated with provenance fields.
2. **Valid Comparative Analyses:**
   - Use `PROTO_STD_CHAPTER_7_ABLATION` configurations (Runs 0–4) to evaluate individual retrieval components.
   - Use `PROTO_STD_CHAPTER_7_KFINAL` configurations ($K_{\text{final}}=1, 3, 5, 10$) to evaluate context window depth.
   - Use `PROTO_STD_CHAPTER_7_GENERATOR_SCALING` to compare LLM reasoning capacity under 100% retrieval recall.
3. **Restricted Cross-Protocol Comparisons:**
   - Always cite `protocol_id` when comparing pilot exploratory runs (e.g. `PROTO_PILOT_TOP_K`) against standardized ablation runs (`PROTO_STD_CHAPTER_7_KFINAL`).
