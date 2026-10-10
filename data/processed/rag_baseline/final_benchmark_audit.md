# Final RAG Benchmark Results & Protocol Audit Report

**Audit Status:** **`AUDIT PASSED — READY FOR CHAPTER 7`**  
**Protocol ID:** `PROTO_FINAL_BENCHMARK`  
**Audit Report Path:** `data/processed/rag_baseline/final_benchmark_audit.md`  
**Target Results File:** [`final_benchmark_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json)  
**Target Per-Question File:** [`per_question_results.jsonl`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/per_question_results.jsonl)  
**Target Summary Report:** [`final_benchmark_report.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_report.md)  
**Audit Timestamp:** 2026-10-07  

---

## 1. Audit Declaration & Summary Status

A comprehensive, end-to-end data integrity, statistical, and protocol compliance audit was conducted on the completed `PROTO_FINAL_BENCHMARK` results prior to initiating Chapter 7 writing. 

### Audit Declaration:
- **Status:** **`AUDIT PASSED — READY FOR CHAPTER 7`**
- **Artifact Integrity:** All raw per-question logs, summary metric JSONs, and Markdown reports under `data/processed/rag_baseline/final_benchmark/` are 100% verified, consistent, and unaltered.
- **Protocol Adherence:** The evaluation strictly adhered to `PROTO_FINAL_BENCHMARK` specified in `data/processed/rag_baseline/final_benchmark_protocol.md`.

---

## 2. Metric Verification Table (Raw JSONL vs Summary Reports)

Every metric reported in [`final_benchmark_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json) and [`final_benchmark_report.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_report.md) was independently recalculated from [`per_question_results.jsonl`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/per_question_results.jsonl):

| Metric | Raw JSONL Calculation (Test Set N=40) | Reported in `final_benchmark_results.json` | Reported in `final_benchmark_report.md` | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Answer Accuracy (%)** | **90.00%** (36/40) | 90.0% | 90.0% | **MATCH (VERIFIED)** |
| **Exact Match Rate (%)** | **85.00%** (34/40) | 85.0% | 85.0% | **MATCH (VERIFIED)** |
| **Hit Rate @ 5 (%)** | **100.00%** (40/40) | 100.0% | 100.0% | **MATCH (VERIFIED)** |
| **Recall @ 5** | **1.0000** | 1.0 | 1.0 | **MATCH (VERIFIED)** |
| **Precision @ 5** | **0.4000** | 0.4 | 0.4 | **MATCH (VERIFIED)** |
| **Dev Answer Accuracy (%)**| **100.00%** (10/10) | 100.0% | 100.0% | **MATCH (VERIFIED)** |
| **Dev Exact Match (%)** | **80.00%** (8/10) | 80.0% | 80.0% | **MATCH (VERIFIED)** |

---

## 3. Bootstrap 95% Confidence Interval Verification ($B=1,000$, Seed 42)

Empirical percentile 95% Confidence Intervals were independently recomputed using non-parametric bootstrap resampling ($B=1,000$ iterations, random seed `42`) over the 40 test query records:

| Performance Metric | Point Estimate | Independent 95% Bootstrap CI | JSON & Report Stated 95% CI | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Answer Accuracy** | 90.00% | **[80.00%, 97.50%]** | [80.0%, 97.5%] | **MATCH (VERIFIED)** |
| **Exact Match Rate** | 85.00% | **[72.50%, 95.00%]** | [72.5%, 95.0%] | **MATCH (VERIFIED)** |
| **Hit Rate @ 5** | 100.00% | **[100.00%, 100.00%]** | [100.0%, 100.0%] | **MATCH (VERIFIED)** |
| **Recall @ 5** | 1.0000 | **[1.0000, 1.0000]** | [1.0, 1.0] | **MATCH (VERIFIED)** |
| **Precision @ 5** | 0.4000 | **[0.4000, 0.4000]** | [0.4, 0.4] | **MATCH (VERIFIED)** |
| **Generation Latency (Mean)** | 6,436.84 ms | **[5,959.18 ms, 7,021.75 ms]** | [5959.18 ms, 7021.75 ms] | **MATCH (VERIFIED)** |
| **Total Online Latency (Mean)** | 7,272.35 ms | **[6,768.42 ms, 7,874.62 ms]** | [6768.42 ms, 7874.62 ms] | **MATCH (VERIFIED)** |

---

## 4. Leakage & Split Verification

- **Total Query Count:** 50 Benchmark Queries (`data/processed/hotpotqa_pilot_queries.json`).
- **Dev Set Allocation:** 10 Queries (20.0%, Indices 0–9).
- **Test Set Allocation:** 40 Queries (80.0%, Indices 10–49).
- **Intersection Audit:** `dev_qids.intersection(test_qids) = 0` (Zero Query ID overlap).
- **Annotation Leakage Audit:** No test set relevance annotations (`qrels`) or gold answers were accessible to hyperparameter tuning scripts.
- **Split Verification Status:** **PASSED (LEAKAGE-SAFE)**.

---

## 5. Failure Taxonomy & Test Error Audit

The 4 errors occurring on the 40-query Test Set were individually audited against the fixed 5-category failure taxonomy (RF, MH, GF, IC, DQ):

| Failure Category | Test Count | Dev Count | Re-evaluated Error Queries | Root Cause & Justification |
| :--- | :---: | :---: | :--- | :--- |
| **Retrieval Failure (RF)** | 0 | 0 | None | Top-5 hit rate was 100% across all test queries. |
| **Multi-Hop Reasoning Failure (MH)** | 3 | 0 | QID `5a8c5d795542995e66a475d4`<br>QID `5a71532e5542994082a3e7d6`<br>QID `5ab4826f5542990594ba9c3c` | All gold chunks present, but LLM missed Pinewood Studios detail (Q1), returned empty string for 1999 movie year (Q18), or picked 2003 season year vs 2004 game year (Q30). |
| **Generation Failure (GF)** | 0 | 0 | None | Zero prompt header echoes or formatting failures. |
| **Insufficient Context (IC)** | 0 | 0 | None | Corpus contained all required gold passages. |
| **Gold-Answer / Data-Quality Issue (DQ)**| 1 | 0 | QID `5adf48b25542995ec70e8fb5` | Dataset annotation defect: Gold answer is `British` for a city-level query. |
| **Total Test Errors** | **4** | **0** | **3 MH + 1 DQ = 4 Errors** | **Accounts for 100% of test errors (36/40 = 90.0% Acc).** |

---

## 6. Online & Offline Latency Decomposition Audit

Component latencies reported in [`latency_analysis_report.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/latency_analysis_report.md) were verified against individual query timing logs:

| Component Stage | Mean Latency (ms) | P50 Median (ms) | P95 Latency (ms) | Latency Share (%) | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense Search (`bge-small-en-v1.5`)** | 45.59 ms | 44.38 ms | 58.43 ms | 0.6% | **VERIFIED PERFECT** |
| **Lexical Search (`BM25Okapi`)** | 1.07 ms | 0.92 ms | 1.62 ms | 0.0% | **VERIFIED PERFECT** |
| **Cross-Encoder Reranking (`MiniLM-L6`)**| 788.71 ms | 734.05 ms | 1,083.90 ms | 10.8% | **VERIFIED PERFECT** |
| **LLM Generation (`flan-t5-large`)** | 6,436.84 ms | 5,965.80 ms | 8,483.31 ms | 88.5% | **VERIFIED PERFECT** |
| **Total Online End-to-End** | **7,272.35 ms** | **6,787.30 ms** | **9,487.72 ms** | **100.0%** | **VERIFIED PERFECT** |
| **Offline Indexing Time (s)** | **13.3266 s** | - | - | - | **VERIFIED PERFECT** |

---

## 7. Protocol Compliance Checklist

- [x] **Document Chunking:** Standard 250w/50o chunks, Title + Text format (No LLM contextual summaries).
- [x] **Dense Embedding Model:** `BAAI/bge-small-en-v1.5` (384-d, FAISS Flat IP).
- [x] **Lexical Model:** `BM25Okapi`.
- [x] **Score Fusion:** Min-Max normalized hybrid fusion ($\alpha=0.5$), candidate pool $K_{\text{init}}=15$.
- [x] **Cross-Encoder Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` down to $K_{\text{final}}=5$.
- [x] **Generator LLM:** `google/flan-t5-large` (780M, greedy $T=0.0$, `max_new_tokens=50`, fixed seed `42`).
- [x] **Dataset Split:** Leakage-safe 10 Dev / 40 Test split.
- [x] **Statistical Significance:** $B=1,000$ non-parametric bootstrap resampling 95% CIs.

---

## 8. Separation of Pilot Exploratory Result vs. Benchmark Claim

> [!IMPORTANT]
> **CLEAR SEPARATION OF EXPERIMENTAL CLAIMS:**
> - **Chapter 7 Exploratory Scaling Pilot Result:** **96.0% Answer Accuracy** (48/50 correct) was obtained during initial generator scaling (`PROTO_STD_CHAPTER_7_GENERATOR_SCALING`) on an unpartitioned 50-query exploratory set.
> - **Authoritative Benchmark Claim (`PROTO_FINAL_BENCHMARK`):** **90.00% Answer Accuracy [80.00%, 97.50%]** (36/40 correct) and **85.00% Exact Match [72.50%, 95.00%]** (34/40 exact) is the sole official benchmark result on the held-out **40-query Test Set**.

---

## 9. Discrepancies & Flagged Items

- **Discrepancies Found:** **ZERO**. All metrics, percentiles, bootstrap CIs, split sizes, and failure counts match perfectly across all result artifacts.
- **Final Audit Conclusion:** **`AUDIT PASSED — READY FOR CHAPTER 7`**.
