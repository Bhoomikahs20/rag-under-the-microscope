# Controlled Multi-Query Expansion Experiment Report

## Executive Summary

This report presents a controlled evaluation of **Multi-Query Expansion** applied to the standardized **HotpotQA 50-Question RAG Benchmark**. The experiment compares the single-query **MiniLM-L6-v2 $K=5$ Baseline** against a **Multi-Query Expansion Pipeline** that retains the original query as Query 0 while generating 2 additional query variations per question using `google/flan-t5-base`. 

Candidates retrieved from each query variation are merged and deduplicated using vector similarity score max-fusion to yield a final Top-5 context window for generation with `FLAN-T5-Base`.

### Key Experimental Findings
1. **Retrieval Recall & Precision:** Multi-Query Expansion achieved a **0.9400 Recall@5** (down 1.0% from 0.9500 in the baseline) and **0.3760 Precision@5** (down 1.0% from 0.3800).
2. **End-to-End Accuracy:** Answer Accuracy reached **72.0%** (36/50 correct) compared to **78.0%** (39/50 correct) in the Original-Query $K=5$ baseline. Exact Match was **60.0%** (vs 64.0% baseline).
3. **Retrieval Rescues vs. Displacement Trade-off:**
   - **Rescued Hop (Q5 - `5ae5fb97`):** Multi-query expansion successfully retrieved the missing 2nd-hop document (*State of Mexico*) by generating the concise variant `"Mexico"`, bringing Q5 from partial retrieval failure to full multi-doc retrieval (`all_gold_retrieved = True`).
   - **Candidate Displacement Penalty (Q34 - `5ae7ac1f` & Q35 - `5a746917`):** Overly broad entity query variations (e.g. `"André Aciman"` and `"Robert Evans (film)"`) produced high cosine similarity scores (0.82–0.84) on generic biography chunks. During candidate merging, these generic chunks displaced the specific multi-hop ground-truth documents from the top-5 window, leading to 2 new retrieval failures.
4. **Latency Overhead:** Query expansion added **2.34 seconds of latency per question** (total pipeline runtime: 224.68 seconds vs 125.22 seconds baseline).

---

## Controlled Experimental Design

To ensure a strict head-to-head comparison, all underlying components were kept identical to the baseline:

| Component / Parameter | Standard Baseline | Multi-Query Expansion Pipeline |
| :--- | :--- | :--- |
| **Corpus & Questions** | 100 HotpotQA Pilot Chunks, 50 Multi-Hop Queries | 100 HotpotQA Pilot Chunks, 50 Multi-Hop Queries |
| **Chunking Strategy** | 250 words / 50 words overlap | 250 words / 50 words overlap |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Index** | FAISS `IndexFlatIP` (Cosine Similarity) | FAISS `IndexFlatIP` (Cosine Similarity) |
| **Expander Model** | N/A (Original Query Only) | `google/flan-t5-base` (Query 0 + 2 Variations) |
| **Candidate Fusion** | N/A | Max-Score Merging & Top-5 Deduplication |
| **Final Context Window** | $K=5$ Chunks | $K=5$ Chunks |
| **Generator Model** | `google/flan-t5-base` | `google/flan-t5-base` |
| **Generation Settings** | Greedy ($T=0.0$), `max_new_tokens=50` | Greedy ($T=0.0$), `max_new_tokens=50` |
| **Random Seed** | `42` | `42` |

---

## Quantitative Results & Metric Comparison

### Overall Performance Overview

| Metric | Original Query Baseline ($K=5$) | Multi-Query Expansion ($K=5$) | Absolute Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Hit Rate@5 (%)** | **100.0%** | **100.0%** | 0.0% | 0.0% |
| **Recall@5** | **0.9500** | **0.9400** | -0.0100 | -1.05% |
| **Precision@5** | **0.3800** | **0.3760** | -0.0040 | -1.05% |
| **Answer Accuracy (%)** | **78.0%** (39/50) | **72.0%** (36/50) | -6.0% | -7.69% |
| **Exact Match (%)** | **64.0%** (32/50) | **60.0%** (30/50) | -4.0% | -6.25% |
| **Avg Latency / Query** | **2.50s** | **4.49s** | +1.99s | +79.6% |
| **Expansion Latency / Query** | 0.00s | **2.34s** | +2.34s | N/A |
| **Total Runtime** | **125.22s** | **224.68s** | +99.46s | +79.4% |

---

## Comprehensive Failure Category Breakdown

Failures across both pipelines were systematically classified using the standardized 6-category taxonomy:

| Failure Category | Baseline ($K=5$) Count | Multi-Query ($K=5$) Count | Category Net Shift | Primary Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Failure** | 4 | **5** | +1 | High-similarity generic entity chunks displaced multi-hop target docs on Q34 & Q35 (partially offset by Q5 rescue). |
| **Multi-Hop Reasoning Failure** | 5 | **6** | +1 | Context pollution from expanded query chunks caused generator confusion on Q14 & Q31. |
| **Generation Failure** | 1 | **2** | +1 | Verbose / repetitive prompt template fallback triggered on Q17 & Q23. |
| **Gold-Answer / Data-Quality Issue** | 1 | **1** | 0 | Ground-truth string mismatch on Q13 (`New York City` vs `Washington, D.C.`). |
| **Insufficient Context** | 0 | **0** | 0 | Retained at zero. |
| **Answer-Format Mismatch** | 0 | **0** | 0 | Retained at zero. |
| **Total Failures** | **11** | **14** | **+3** | Net loss of 3 questions. |

---

## Detailed Case Studies: Rescues & Regressions

### 1. The Multi-Query Rescue (Q5 - `5ae5fb97`)
- **Question:** *"Edmundo 'Eddie' Rodriguez was born in what Mexican state?"*
- **Gold Answer:** `State of Mexico` | **Gold Docs:** `1499129` (*Eddie Rodriguez*), `3634471` (*State of Mexico*)
- **Baseline Behavior:** Retrieved only `1499129` (*Eddie Rodriguez*). Failed to retrieve `3634471` because the original query lacked explicit mention of "State of Mexico" (`all_gold_retrieved = False`).
- **Multi-Query Expansion Behavior:**
  - *Generated Variation 2:* `"Mexico"`
  - *Retrieval Output:* Searching for `"Mexico"` returned chunk `3634471_c0` (*State of Mexico*) with high similarity (0.642). Max-score candidate merging successfully included both gold documents in the top-5 window (`all_gold_retrieved = True`).

### 2. Retrieval Failure Regression via Candidate Displacement (Q34 - `5ae7ac1f`)
- **Question:** *"What is the name of the Los Angeles-based production company that produced a film based on the novel by André Aciman?"*
- **Gold Answer:** `Frenesy Film Company` / `Water's End Productions`
- **Baseline Behavior:** Retrieved all gold documents (`all_gold_retrieved = True`).
- **Multi-Query Expansion Behavior:**
  - *Generated Variation 1:* `"André Aciman"`
  - *Displacement Effect:* Searching specifically for `"André Aciman"` retrieved generic author biography chunks with very high similarity (0.82), which pushed the target production company chunk out of the final Top-5 candidate window (`all_gold_retrieved = False`).

### 3. Reasoner Distraction & Context Noise (Q31 - `5ab8f1fa`)
- **Question:** *"Are Gary Lightbody and Dustin Kensrue from the same country?"*
- **Gold Answer:** `no` (Gary Lightbody is Northern Irish/UK, Dustin Kensrue is American).
- **Multi-Query Expansion Behavior:**
  - *Generated Variation 1:* `"Gary Lightbody and Dustin Kensrue are American football players."` (Hallucinated query rewrite).
  - *Impact:* The hallucinated search query fetched irrelevant sports chunks, causing `FLAN-T5-Base` to generate an incorrect answer.

---

## Practical Engineering Recommendations

1. **Retain Original Query (Query 0):** Always keep the original user prompt as the primary query anchor to prevent complete semantic drift.
2. **Implement Score Normalization or RRF:** Max-score merging can allow broad entity queries to overpower specific multi-hop queries. Using **Reciprocal Rank Fusion (RRF)** or **Rank-Based Merging** reduces the risk of candidate displacement.
3. **Thresholding Query Variations:** Apply quality filtering on LLM-generated query variations to filter out hallucinated statements before issuing search requests.
4. **Hybrid Search Over Multi-Query:** Controlled experiments on this benchmark demonstrate that **Hybrid Search (Dense + BM25)** achieves superior Recall@5 (**0.9800**) and Accuracy (**78.0%**) without the ~2.34s/query latency penalty of LLM query expansion.

---

## Saved Artifacts Verification

All outputs, generated queries, retrieved candidate logs, and quantitative metric results have been saved under:
`data/processed/rag_baseline/multi_query_experiment/`

- **Per-Question & Summary Metrics JSON:** [`data/processed/rag_baseline/multi_query_experiment/multi_query_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/multi_query_experiment/multi_query_results.json)
- **Comparative Markdown Report:** [`data/processed/rag_baseline/multi_query_experiment/multi_query_comparison_report.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/multi_query_experiment/multi_query_comparison_report.md)
