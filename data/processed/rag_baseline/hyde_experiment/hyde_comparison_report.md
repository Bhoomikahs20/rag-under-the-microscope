# Controlled HyDE (Hypothetical Document Embeddings) Experiment Report

## Executive Summary

This report presents a controlled evaluation of **HyDE (Hypothetical Document Embeddings)** applied to the standardized **HotpotQA 50-Question RAG Benchmark**. The experiment compares the single-query **MiniLM-L6-v2 $K=5$ Baseline** against a **HyDE RAG Pipeline** that uses `google/flan-t5-base` to generate a hypothetical answer document for each question, embeds that hypothetical document using `sentence-transformers/all-MiniLM-L6-v2`, retrieves top-5 candidate chunks from the FAISS vector index, and passes them to `FLAN-T5-Base` for final answer generation.

### Key Experimental Findings
1. **Severe Retrieval Degradation:** HyDE caused a sharp drop in retrieval performance. **Recall@5 fell from 0.9500 to 0.6600** (-29.0% absolute), **Hit Rate@5 dropped from 100.0% to 78.0%** (-22.0% absolute), and **Precision@5 dropped from 0.3800 to 0.2640**.
2. **End-to-End Accuracy Collapse:** Answer Accuracy dropped from **78.0% (39/50 correct)** in the baseline to **54.0% (27/50 correct)** under HyDE (-24.0% absolute, -30.8% relative). Exact Match fell from **64.0% (32/50)** to **42.0% (21/50)**.
3. **Retrieval Failure Explosion:** Retrieval failures quadrupled from **4 to 18 questions (+14 failures)**.
4. **Root Cause Analysis (Hallucination Bias):** Small zero-shot LLMs (`FLAN-T5-Base`) attempting to generate hypothetical passages without corpus context hallucinate incorrect entities, dates, and director names (e.g. asserting *Peter Cook* wrote *Absolutely Fabulous* on Q7 or *John F. Kennedy* was UC chancellor on Q11). Dense embeddings of these hallucinated passages steer FAISS vector search into incorrect semantic clusters, fetching chunks that match the *hallucination* rather than the ground-truth facts required for multi-hop reasoning.

---

## Controlled Experimental Design

To ensure a strict head-to-head comparison, all underlying components were kept identical to the baseline:

| Component / Parameter | Standard Baseline | HyDE Pipeline |
| :--- | :--- | :--- |
| **Corpus & Questions** | 100 HotpotQA Pilot Chunks, 50 Multi-Hop Queries | 100 HotpotQA Pilot Chunks, 50 Multi-Hop Queries |
| **Chunking Strategy** | 250 words / 50 words overlap | 250 words / 50 words overlap |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Index** | FAISS `IndexFlatIP` (Cosine Similarity) | FAISS `IndexFlatIP` (Cosine Similarity) |
| **Hypothetical Doc Generator** | N/A (Original Query Used) | `google/flan-t5-base` (`max_new_tokens=80`, $T=0.0$) |
| **Retrieval Query Source** | Original Question Text | Generated Hypothetical Passage Embedding |
| **Final Context Window** | $K=5$ Chunks | $K=5$ Chunks |
| **Answer Generator Model** | `google/flan-t5-base` | `google/flan-t5-base` |
| **Generation Settings** | Greedy ($T=0.0$), `max_new_tokens=50` | Greedy ($T=0.0$), `max_new_tokens=50` |
| **Random Seed** | `42` | `42` |

---

## Quantitative Results & Metric Comparison

### Overall Performance Overview

| Metric | Original Query Baseline ($K=5$) | HyDE Pipeline ($K=5$) | Absolute Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Hit Rate@5 (%)** | **100.0%** | **78.0%** | -22.0% | -22.00% |
| **Recall@5** | **0.9500** | **0.6600** | -0.2900 | -30.53% |
| **Precision@5** | **0.3800** | **0.2640** | -0.1160 | -30.53% |
| **Answer Accuracy (%)** | **78.0%** (39/50) | **54.0%** (27/50) | -24.0% | -30.77% |
| **Exact Match (%)** | **64.0%** (32/50) | **42.0%** (21/50) | -22.0% | -34.38% |
| **Avg HyDE Doc Gen Latency** | 0.00s | **0.46s** | +0.46s | N/A |
| **Avg Total Latency / Query** | **2.50s** | **1.72s** | -0.78s | -31.2% |
| **Total Experiment Runtime** | **125.22s** | **97.78s** | -27.44s | -21.9% |

---

## Comprehensive Failure Category Breakdown

Failures across both pipelines were systematically classified using the standardized 6-category taxonomy:

| Failure Category | Baseline ($K=5$) Count | HyDE Pipeline ($K=5$) Count | Category Net Shift | Primary Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Failure** | 4 | **18** | **+14** | Hallucinated hypothetical passages shifted vector embeddings into wrong semantic regions, failing to retrieve gold multi-hop docs. |
| **Multi-Hop Reasoning Failure** | 5 | **3** | -2 | Fewer questions reached the generator with intact context due to prior retrieval failure. |
| **Generation Failure** | 1 | **1** | 0 | Same prompt template fallback on Q17. |
| **Gold-Answer / Data-Quality Issue** | 1 | **1** | 0 | Ground-truth string mismatch on Q13 (`New York City` vs `Washington, D.C.`). |
| **Insufficient Context** | 0 | **0** | 0 | Retained at zero. |
| **Answer-Format Mismatch** | 0 | **0** | 0 | Retained at zero. |
| **Total Failures** | **11** | **23** | **+12** | Net loss of 12 questions overall. |

---

## Detailed Case Studies: Qualitative Analysis of HyDE Failure Mechanisms

### 1. Entity Hallucination Steers Dense Vector Retrieval Off-Target (Q7 - `5a7257ce`)
- **Question:** *"This English comedian wrote the British sitcom 'Absolutely Fabulous'"*
- **Gold Answer:** `Jennifer Saunders` | **Gold Docs:** `704381` (*Jennifer Saunders*), `103833` (*Absolutely Fabulous*)
- **Baseline Behavior:** Retrieved all gold documents (`Recall@5 = 1.0`). Generated correct answer `Jennifer Saunders`.
- **HyDE Behavior:**
  - *Generated Hypothetical Doc:* `"The British sitcom 'Absolutely Fabulous' was written by Peter Cook."` (Factual hallucination: *Peter Cook*).
  - *Retrieval Impact:* Embedding this hallucinated document led FAISS to retrieve chunks about Peter Cook and 1960s British comedy satire rather than Jennifer Saunders. Gold document `704381` was completely missed (`all_gold_retrieved = False`), resulting in a **Retrieval Failure**.

### 2. Multi-Hop Disruption on Complex Comparisons (Q14 - `5a89c271`)
- **Question:** *"Which was released first, God's Angry Man or The World's Best Prom?"*
- **Gold Answer:** `God's Angry Man` (1981 vs 2006)
- **Baseline Behavior:** Retrieved both film documents (`Recall@5 = 1.0`). Generated `God's Angry Man`.
- **HyDE Behavior:**
  - *Generated Hypothetical Doc:* `"The World's Best Prom was released in the late 1950s."` (Hallucinated date: *late 1950s*).
  - *Retrieval Impact:* The dense embedding searched for 1950s film history chunks, missing the actual 1981 *God's Angry Man* chunk. Resulted in a **Retrieval Failure**.

### 3. Factual Distortion on Person/Title Lookups (Q6 - `5abd0e83`)
- **Question:** *"Sylvia Sidney starred in the film Beetlejuice directed by who?"*
- **Gold Answer:** `Tim Burton`
- **Baseline Behavior:** Retrieved *Beetlejuice* and *Tim Burton* chunks (`Recall@5 = 1.0`). Generated `Tim Burton`.
- **HyDE Behavior:**
  - *Generated Hypothetical Doc:* `"John Carpenter"`
  - *Retrieval Impact:* Vector search retrieved horror director chunks matching *John Carpenter*, missing the *Tim Burton* chunk (`Recall@5 = 0.0`).

---

## Practical Engineering Recommendations

1. **Do Not Use Zero-Shot HyDE with Small Local Models on Multi-Hop Facts:** Small models (<1B parameters) without corpus context hallucinate entity names and dates, making zero-shot HyDE highly detrimental to retrieval precision.
2. **Combine Question Text with Hypothetical Passages:** If HyDE is deployed, concatenating the original question $q$ with the hypothetical passage $d_{hypo}$ (or performing hybrid score fusion $v_{search} = \alpha v_q + (1-\alpha) v_{hypo}$) prevents semantic drift caused by hallucinated terms.
3. **Prefer Reranking or Hybrid Search:** Controlled benchmark results show that **Hybrid Search (Dense + BM25)** achieves **0.9800 Recall@5** and **78.0% Accuracy**, whereas **Cross-Encoder Reranking** achieves **0.9600 Recall@5** and **78.0% Accuracy**, both dramatically outperforming HyDE (0.6600 Recall@5 / 54.0% Accuracy).

---

## Saved Artifacts Verification

All outputs, generated hypothetical passages, retrieved candidate logs, and quantitative metric results have been saved under:
`data/processed/rag_baseline/hyde_experiment/`

- **Per-Question & Summary Metrics JSON:** [`data/processed/rag_baseline/hyde_experiment/hyde_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/hyde_experiment/hyde_results.json)
- **Comparative Markdown Report:** [`data/processed/rag_baseline/hyde_experiment/hyde_comparison_report.md`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/hyde_experiment/hyde_comparison_report.md)
