# Chapter 7 Final RAG Benchmark Protocol

**Protocol Identifier:** `PROTO_FINAL_BENCHMARK`  
**Location:** `data/processed/rag_baseline/final_benchmark_protocol.md`  
**Audited Provenance Base:** [`chapter_7_results_audit.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_results_audit.md) & [`chapter_7_master_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/chapter_7_master_results.json)  
**Creation Date:** 2026-10-06  
**Status:** Certified Specification Only (Benchmark Execution Pending)  

---

> [!IMPORTANT]
> **PILOT RESULT DISCLAIMER & ACCURACY CLAIM RESTRICTION**  
> The 96.0% answer accuracy achieved during Chapter 7 generator scaling is **strictly a 50-query pilot result** under exploratory evaluation parameters (`PROTO_STD_CHAPTER_7_GENERATOR_SCALING`). It **MUST NOT** be cited as the final benchmark performance claim. The authoritative system performance claim must be derived exclusively from executing this protocol over the complete, leakage-safe benchmark evaluation dataset.

---

## 1. Executive Summary & Objective

This document defines the **finalized, standardized benchmark protocol** (`PROTO_FINAL_BENCHMARK`) for evaluating end-to-end Retrieval-Augmented Generation (RAG) system performance. Built upon the audited empirical findings from Chapter 7 component ablations, candidate pool sensitivity experiments, and generator scaling runs, this protocol standardizes the system pipeline, dataset splits, evaluation metrics, statistical significance methodology, failure taxonomy, and latency instrumentation.

---

## 2. Final Candidate RAG Architecture Specification

The production benchmark pipeline is strictly locked to the optimal configuration determined during Chapter 7 ablation studies:

```
                            FINAL BENCHMARK PIPELINE (`PROTO_FINAL_BENCHMARK`)
┌──────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│ Corpus Documents │ ──► │ Chunking: 250w / 50o   │ ──► │ Dense: BGE-Small-v1.5  │ ──┐
└──────────────────┘     └────────────────────────┘     └────────────────────────┘   │
                                                                                     ├──► Hybrid Score Fusion (alpha=0.5)
                                                        ┌────────────────────────┐   │    Candidate Pool K_init = 15
                                                        │ Lexical: BM25Okapi     │ ──┘
                                                        └────────────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────────┐
                                                        │ Cross-Encoder Reranker │
                                                        │ ms-marco-MiniLM-L-6-v2 │
                                                        └────────────────────────┘
                                                                    │
                                                                    ▼ (Top K_final = 5 Chunks)
                                                        ┌────────────────────────┐
                                                        │ Generator Model:       │ ──► Final Benchmark Output
                                                        │ google/flan-t5-large   │     (Accuracy, EM, Latency)
                                                        └────────────────────────┘
```

### Component Details:
1. **Document Chunking Strategy:**
   - **Granularity:** 250 words per chunk with 50-word overlap (`medium_chunks_250w_50o`).
   - **Format:** `Document Title + Chunk Text` (Standard chunks).
   - **Contextual Summaries:** **Omitted**. Audited ablation RUN-0 vs. RUN-1 confirmed that prepending LLM document summaries adds 1,220s of offline precomputation time and token dilution without improving accuracy.
2. **Dense Retriever Module:**
   - **Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors, normalized L2).
   - **Indexing:** FAISS Flat Inner-Product (`FlatIP`) index over normalized embeddings (Cosine similarity).
3. **Lexical Retriever Module:**
   - **Algorithm:** `BM25Okapi` (`rank_bm25` library).
   - **Tokenization:** Lowercase whitespace & punctuation tokenization.
4. **Hybrid Score Fusion:**
   - **Method:** Min-Max normalized score fusion with equal weighting ($\alpha = 0.5$).
   - **Formula:** $S_{\text{hybrid}}(d, q) = 0.5 \cdot \hat{S}_{\text{dense}}(d, q) + 0.5 \cdot \hat{S}_{\text{lexical}}(d, q)$.
   - **Candidate Pool Depth:** $K_{\text{init}} = 15$ candidate chunks forwarded to reranker.
5. **Cross-Encoder Reranker Module:**
   - **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - **Context Window:** Top $K_{\text{final}} = 5$ reranked chunks formatted into generator prompt.
6. **Generator Model:**
   - **Model Weights:** `google/flan-t5-large` (780M parameters).
   - **Decoding Parameters:** Greedy decoding (`do_sample=False`, `temperature=0.0`, `max_new_tokens=50`).
   - **Prompt Template:** Standardized HotpotQA open-book context prompt structure.
   - **Random Seed:** Fixed seed `42`.

---

## 3. Data Integrity & Leakage-Safe Dataset Split Protocol

To eliminate data leakage, overfitting, and evaluation bias:

### 3.1 Split Architecture:
- **Development Set (Dev):** 20% of benchmark queries reserved strictly for hyperparameter validation and prompt structure sanity checking.
- **Evaluation Set (Test):** 80% held-out test split reserved exclusively for final benchmark reporting.

### 3.2 Leakage-Prevention Constraints:
1. **Query Independence:** Zero query string overlap between Dev and Test sets.
2. **Document Partitioning Integrity:** Corpus documents associated with Test set queries must not have their ground truth relevance annotations exposed during Dev tuning.
3. **Model Freeze:** All model weights (`bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2`, `flan-t5-large`) and hyperparameter thresholds ($\alpha=0.5$, $K_{\text{init}}=15$, $K_{\text{final}}=5$) are strictly frozen prior to running Test evaluation.

---

## 4. Evaluation Metrics & Statistical Methodology

### 4.1 Retrieval Metrics:
- **Hit Rate @ $K$ ($K_{\text{final}}=5$):** Percentage of queries where at least one gold document chunk appears in the top-$K$ contexts.
- **Recall @ $K$ ($K_{\text{final}}=5$):** Fraction of all gold document chunks successfully retrieved in top-$K$.
- **Precision @ $K$ ($K_{\text{final}}=5$):** Fraction of retrieved top-$K$ chunks that are labeled gold.
- **Mean Reciprocal Rank (MRR @ $K$):** Reciprocal rank of the first gold chunk in top-$K$.

### 4.2 Generation Quality Metrics:
- **Answer Accuracy (%):** Percentage of queries where generated output matches gold standard via normalized string inclusion or token-level F1 score $\ge 0.80$.
- **Exact Match Rate (%):** Percentage of queries where generated output perfectly matches gold string after standard normalization (lowercasing, removing articles and punctuation).
- **Token Overlap F1 Score:** Mean harmonic mean of precision and recall over unigram answer tokens.

### 4.3 Statistical Significance & Bootstrap 95% Confidence Intervals:
- **Method:** Non-parametric bootstrap resampling over query test results.
- **Resampling Iterations:** $B = 1,000$ bootstrap replicates with replacement.
- **Confidence Bounds:** 2.5th and 97.5th empirical percentiles of bootstrap distribution.
- **Reporting Requirement:** All accuracy, exact match, recall, and latency metrics must be presented as `Mean [95% CI Lower, Upper]` (e.g., `96.0% [90.0%, 100.0%]`).

---

## 5. Standardized Failure Taxonomy

Every query failure during benchmark evaluation must be categorized into exactly one of five mutually exclusive error buckets:

1. **Retrieval Failure (RF):** Target gold document chunks are completely missing from the top-$K_{\text{final}}$ retrieved contexts.
2. **Multi-Hop Reasoning Failure (MH):** All necessary gold chunks are present in top-$K_{\text{final}}$, but the generator model fails to correctly synthesize multi-step logical facts.
3. **Generation / Formatting Failure (GF):** The generator model outputs prompt header boilerplate, invalid characters, or fails formatting constraints despite valid context.
4. **Insufficient Context (IC):** The query asks for information not contained in the provided corpus.
5. **Gold-Answer / Data-Quality Issue (DQ):** Annotation flaws or typos in the benchmark dataset gold label (e.g., nationality labeled for a city question).

---

## 6. Standardized Latency & Compute Instrumentation

Online and offline execution latencies must be instrumented using microsecond-accurate monotonic clock timers (`time.perf_counter_ns()`), reporting **Mean**, **P50 (Median)**, and **P95 Percentile** latencies:

```
                            LATENCY INSTRUMENTATION BREAKDOWN
┌───────────────────────────┬─────────────────────────────┬─────────────────────────────┐
│ Pipeline Stage            │ Metric Reported             │ Unit                        │
├───────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ 1. Dense Retrieval        │ Mean / P50 / P95 Latency    │ milliseconds (ms)           │
│ 2. Lexical BM25 Retrieval │ Mean / P50 / P95 Latency    │ milliseconds (ms)           │
│ 3. Cross-Encoder Reranking│ Mean / P50 / P95 Latency    │ milliseconds (ms)           │
│ 4. LLM Answer Generation  │ Mean / P50 / P95 Latency    │ milliseconds (ms)           │
│ 5. Total Online End-to-End│ Mean / P50 / P95 Latency    │ milliseconds (ms)           │
│ 6. Offline Indexing       │ Index Creation / Embed Time │ seconds (s)                 │
└───────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

*Note on Latency Scaling:* Online LLM generation latency increases significantly with larger generator model parameter count (`flan-t5-large` vs `flan-t5-base`) and higher $K_{\text{final}}$ retrieval depth ($K_{\text{final}}=1$ vs $K_{\text{final}}=10$).

---

## 7. Execution Rules & Provenance Verification

- **Execution Status:** **DO NOT EXECUTE BENCHMARK RUNS AUTOMATICALLY.** Benchmark execution must be triggered explicitly via dedicated script execution commands (`run_final_benchmark.py`).
- **Provenance Integrity:** Results generated under this protocol must be saved to `data/processed/rag_baseline/final_benchmark_results.json` and tagged with `protocol_id: PROTO_FINAL_BENCHMARK`.
