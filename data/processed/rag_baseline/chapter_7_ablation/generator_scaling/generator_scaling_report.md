# Chapter 7 Generator Scaling Experiment Report: Flan-T5-Base (250M) vs Flan-T5-Large (780M)

**Experiment Location:** `data/processed/rag_baseline/chapter_7_ablation/generator_scaling/`  
**Raw Results Data:** [`generator_scaling_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/chapter_7_ablation/generator_scaling/generator_scaling_results.json)  
**Evaluation Timestamp:** 2026-10-06  

---

## 1. Executive Summary

This experiment evaluates the impact of **scaling the generator LLM capacity** while holding the **finalized Chapter 7 retrieval pipeline** strictly fixed. The finalized retrieval architecture consists of standard 250w/50o chunks (without contextual document summaries), BAAI/bge-small-en-v1.5 dense retrieval + BM25Okapi lexical retrieval with normalized weighted score fusion ($\alpha=0.5$), candidate pool $K_{\text{init}}=15$, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), and final reranked context window $K_{\text{final}}=5$.

We compare the baseline 250M-parameter generator (**`google/flan-t5-base`**) against a 3.12× larger instruction-tuned model (**`google/flan-t5-large`**, 780M parameters) available in the environment. Both generators were evaluated on the identical set of 50 HotpotQA queries, gold labels, prompt template, greedy decoding strategy ($T=0.0$, `max_new_tokens=50`), and random seed ($42$).

### Key Empirical Findings:
1. **Significant Accuracy Uplift (+18.0% Absolute):** Scaling from `flan-t5-base` to `flan-t5-large` increases **Answer Accuracy from 78.0% (39/50) to 96.0% (48/50)** and **Exact Match from 66.0% (33/50) to 84.0% (42/50)** under identical 100% retrieval recall ($K_{\text{final}}=5$).
2. **Drastic Reduction in Multi-Hop Reasoning Failures:** `flan-t5-large` resolved **7 out of 8 multi-hop reasoning failures** present in `flan-t5-base`, reducing multi-hop errors by **87.5%**. The larger parameter count allows the model to correctly synthesize multi-step facts across multiple retrieved chunks (e.g., mapping birthplace city to state, or resolving voice actor identities).
3. **Elimination of Generation Formatting Failures:** `flan-t5-base` exhibited 2 prompt boilerplate repetition failures on long queries, whereas `flan-t5-large` had **0 generation failures (100% formatting compliance)**.
4. **Latency and Compute Cost Trade-off:** Mean generation latency increased from **1,854.72 ms** (`base`) to **5,670.84 ms** (`large`), a **3.06× slowdown** in model inference. Total online end-to-end latency increased from **2,598.07 ms** to **6,485.10 ms** (**2.50× slowdown**). Each +1.0% accuracy gain requires ~212 ms of additional generation latency on CPU.

---

## 2. Experimental Setup & Fixed Variables

To ensure strict experimental control, only the generator model weights were modified between runs:

- **Corpus:** 100 HotpotQA Wikipedia documents (`data/processed/hotpotqa_pilot_corpus.json`).
- **Queries & Qrels:** 50 multi-hop benchmark queries (`data/processed/hotpotqa_pilot_queries.json` and `data/processed/hotpotqa_pilot_qrels.json`).
- **Chunking Configuration:** 250 words per chunk with 50-word overlap (`medium_chunks_250w_50o`). Title + Text format (No contextual summaries).
- **Retrieval Pipeline:**
  - **Dense Retriever:** `BAAI/bge-small-en-v1.5` (384-d, cosine similarity, FAISS flat IP index).
  - **Lexical Retriever:** `BM25Okapi` (rank_bm25, tokenized text).
  - **Score Fusion:** Min-Max normalized hybrid fusion with equal weighting ($\alpha=0.5$).
  - **Candidate Depth:** $K_{\text{init}} = 15$ top hybrid chunks.
  - **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - **Context Window:** $K_{\text{final}} = 5$ top reranked chunks formatted into open-book prompt context.
- **Generator Evaluation Conditions:**
  - **Prompt Structure:** Standardized HotpotQA open-book context prompt.
  - **Decoding Parameters:** Greedy search (`do_sample=False`, `temperature=0.0`, `max_new_tokens=50`).
  - **Seed:** 42.

---

## 3. Quantitative Comparison & Performance Metrics

| Metric Category | Metric | Baseline: `google/flan-t5-base` (250M) | Larger LLM: `google/flan-t5-large` (780M) | Absolute Change | Relative Change / Scaling Factor |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model Spec** | Parameters (M) | 250M | 780M | +530M | 3.12× |
| **Retrieval** | Hit Rate @ 5 | 100.0% | 100.0% | 0.0% | 1.00× |
| | Recall @ 5 | 1.0000 | 1.0000 | 0.0000 | 1.00× |
| | Precision @ 5 | 0.4000 | 0.4000 | 0.0000 | 1.00× |
| **Generation Accuracy** | **Answer Accuracy (%)** | **78.0%** (39/50) | **96.0%** (48/50) | **+18.0%** | **1.23× (+23.1% relative)** |
| | **Exact Match (%)** | **66.0%** (33/50) | **84.0%** (42/50) | **+18.0%** | **1.27× (+27.3% relative)** |
| **Failure Categories** | Retrieval Failure | 0 | 0 | 0 | - |
| | Multi-Hop Reasoning Failure | 8 | 1 | -7 | **-87.5%** |
| | Generation Failure | 2 | 0 | -2 | **-100.0%** |
| | Gold-Answer / Data-Quality Issue | 1 | 1 | 0 | 0.0% |
| **Online Latency (Mean)**| Dense + BM25 Retrieval | 39.70 ms | 46.57 ms | +6.87 ms | 1.17× |
| | Cross-Encoder Reranking | 703.47 ms | 767.56 ms | +64.09 ms | 1.09× |
| | **LLM Generation** | **1,854.72 ms** | **5,670.84 ms** | **+3,816.12 ms** | **3.06×** |
| | **Total Online Latency** | **2,598.07 ms** | **6,485.10 ms** | **+3,887.03 ms** | **2.50×** |
| **Online Latency (P50)**| Retrieval P50 | 42.45 ms | 44.06 ms | +1.61 ms | 1.04× |
| | Reranking P50 | 701.96 ms | 788.10 ms | +86.14 ms | 1.12× |
| | Generation P50 | 1,653.00 ms | 5,704.00 ms | +4,051.00 ms | 3.45× |
| | **Total P50 Online** | **2,454.52 ms** | **6,352.45 ms** | **+3,897.93 ms** | **2.59×** |
| **Online Latency (P95)**| Retrieval P95 | 54.19 ms | 63.79 ms | +9.60 ms | 1.18× |
| | Reranking P95 | 910.50 ms | 913.43 ms | +2.93 ms | 1.00× |
| | Generation P95 | 3,696.60 ms | 8,085.55 ms | +4,388.95 ms | 2.19× |
| | **Total P95 Online** | **4,434.29 ms** | **8,966.45 ms** | **+4,532.16 ms** | **2.02×** |
| **Run Throughput** | Total 50-Query Runtime | 129.91 s | 324.26 s | +194.35 s | 2.50× |

---

## 4. Offline Indexing & Preprocessing Costs

The offline index precomputation costs were identical across both runs because the candidate corpus chunks, embedding model, and BM25 index were reused without modification:

- **Context Summarization Precomputation:** `0.0000 s` (Contextual document summaries omitted in accordance with finalized architecture).
- **Dense Vector Embedding & Indexing (`BAAI/bge-small-en-v1.5`):** `4.4556 s` (100 chunks encoded into 384-dimensional FAISS index).
- **Lexical Indexing (`BM25Okapi`):** `0.0085 s`.
- **Total Offline Indexing Time:** `4.4641 s`.

---

## 5. Detailed Breakdown of Query Resolutions and Failures

### 5.1 Errors Resolved by `flan-t5-large` (9 Queries Fixed)

Scaling from 250M to 780M parameters allowed the generator to overcome complex multi-hop logic barriers:

1. **Query 5a8780385542996e4f30880e (Voice Actor Character Mapping):**
   - *Question:* What character in the television special "The Pound Puppies" did the voice actor of Fred Jones from "Scooby-Doo" voice as?
   - *Gold:* `Howler`
   - *Flan-T5-Base:* `Cooler` (Failed to map Frank Welker to Howler; picked wrong character from Pound Puppies context).
   - *Flan-T5-Large:* `Howler` (Correct multi-hop synthesis across Frank Welker bio & Pound Puppies cast).
2. **Query 5ae5fb975542996de7b71aa8 (Geographic Hop City -> State):**
   - *Question:* Edmundo "Eddie" Rodriguez was born in what Mexican state?
   - *Gold:* `Jalisco`
   - *Flan-T5-Base:* `Guadalajara` (Extracted city of birth but failed second hop to Mexican state).
   - *Flan-T5-Large:* `Jalisco` (Correctly traversed Guadalajara -> Jalisco).
3. **Query 5a7257ce5542990c210a410d (Writer Identification):**
   - *Question:* This English comedian wrote the British sitcom "Absolutely Fabulous"
   - *Gold:* `Jennifer Saunders`
   - *Flan-T5-Base:* `John Sullivan` (Distracted by unrelated sitcom chunk in top-5).
   - *Flan-T5-Large:* `Jennifer Saunders` (Accurately isolated relevant author).
4. **Query 5a89c2715542993b751ca990 (Temporal Chronology):**
   - *Question:* Which was released first, God's Angry Man or The World's Best Prom?
   - *Gold:* `God's Angry Man`
   - *Flan-T5-Base:* `The World's Best Prom` (Inverted release date comparison).
   - *Flan-T5-Large:* `God's Angry Man` (Correctly compared 1975 vs 2006 release years).
5. **Query 5a7e264055429965cec5eaa4 (Prompt Boilerplate Failure):**
   - *Question:* Fulda Cathedral is the burial place of the leading figure in what?
   - *Gold:* `Anglo-Saxon mission`
   - *Flan-T5-Base:* Output prompt header boilerplate string (`"The relevant information to answer..."`).
   - *Flan-T5-Large:* `the Anglo-Saxon mission to the Germanic parts of the Frankish Empire` (Clean answer extraction).
6. **Query 5ab985b5554299131ca4235e (Long-Context Prompt Failure):**
   - *Question:* Player of the year award for Crystal Palace winger playing for Ivory Coast...
   - *Gold:* `Wilfried Zaha`
   - *Flan-T5-Base:* Output prompt header boilerplate string (`"The relevant information to answer..."`).
   - *Flan-T5-Large:* `Wilfried Zaha` (Succinct entity extraction).
7. **Query 5ab8f1fa5542991b5579f077 (Cross-Nationality Comparison):**
   - *Question:* Are Gary Lightbody and Dustin Kensrue from the same country?
   - *Gold:* `no`
   - *Flan-T5-Base:* `yes` (Failed binary comparison).
   - *Flan-T5-Large:* `no` (Correct nationality comparison).
8. **Query 5a74691755429929fddd8412 (Entity Relationship):**
   - *Question:* What friends star's wedding was photographed by Robert Evans?
   - *Gold:* `Jennifer Aniston`
   - *Flan-T5-Base:* `Brad Pitt` (Extracted groom only).
   - *Flan-T5-Large:* `Brad Pitt and Jennifer Aniston` (Exact match substring passed).
9. **Query 5a7541ab5542993748c89803 (Granular Location Pinpointing):**
   - *Question:* What school is located in the southernmost atoll of the archipelago of the Maldives?
   - *Gold:* `Maradhoo Feydhoo School`
   - *Flan-T5-Base:* `Addu City` (Stopped at city/atoll level).
   - *Flan-T5-Large:* `Maradhoo Feydhoo School` (Pinpointed exact school entity).

### 5.2 Remaining Failure Analysis in `flan-t5-large` (2 Out of 50)

Only 2 queries failed under `flan-t5-large` (achieving 96.0% accuracy):

1. **Query 5adf48b25542995ec70e8fb5 (Gold-Answer / Data-Quality Issue):**
   - *Question:* In what city did Amir Khan and Lamont Peterson fight for the welterweight titles in 2011?
   - *Gold Label:* `British` (Dataset annotation flaw—gold answer is nationality rather than city).
   - *Flan-T5-Large Answer:* `Guadalajara` (Hallucinated string due to confusing passage text).
   - *Classification:* **Gold-Answer / Data-Quality Issue**.
2. **Query 5a71532e5542994082a3e7d6 (Multi-Hop Reasoning Breakdown):**
   - *Question:* In what year did the actor who played Edward Snowden act in a movie with Heath Ledger and Julia Styles?
   - *Gold Label:* `1999` (Joseph Gordon-Levitt in "10 Things I Hate About You").
   - *Flan-T5-Large Answer:* `''` (Empty string generation; failed to resolve double hop between Snowden actor and 1999 movie year).
   - *Classification:* **Multi-Hop Reasoning Failure**.

---

## 6. Synthesis & Trade-off Recommendations

```
                                  ACCURACY vs LATENCY TRADE-OFF
┌───────────────────────────┬───────────────────┬───────────────────┬───────────────────────────┐
│ Generator Model           │ Answer Accuracy   │ Generation Latency│ Total Online Latency (P50)│
├───────────────────────────┼───────────────────┼───────────────────┼───────────────────────────┤
│ google/flan-t5-base (250M)│      78.0%        │     1.85s / q     │          2.45s / q        │
│ google/flan-t5-large (780M│      96.0%        │     5.67s / q     │          6.35s / q        │
└───────────────────────────┴───────────────────┴───────────────────┴───────────────────────────┘
               ▲                                                                 │
               │                                                                 ▼
      +18.0% Accuracy Uplift                                            +2.50x Latency Growth
```

### Architectural Takeaways for RAG System Design:
1. **Retrieval is Necessary, but Parameter Count Dictates Reasoning:** Perfect retrieval (Recall@5 = 100%) is insufficient if the generator model lacks sufficient parametric capacity to perform complex multi-hop reasoning over the top-$K$ chunks.
2. **Diminishing Returns vs Compute Budget:** Scaling from 250M to 780M parameters adds ~3.8 seconds per query in generation overhead on CPU. In latency-critical production environments, 2.45s response times (`base`) may be preferred over 6.35s response times (`large`), unless response quality for complex queries is the primary SLA.
3. **Generator Failure Elimination:** Larger instruction tuning fundamentally fixes prompt adherence issues (eliminating prompt boilerplate echoes and formatting collapse), making 700M+ models far more robust for automated RAG pipelines.

---

## 7. Artifact Directory Structure

All artifacts from this experiment have been persisted under `data/processed/rag_baseline/chapter_7_ablation/generator_scaling/`:

```
data/processed/rag_baseline/chapter_7_ablation/generator_scaling/
├── generator_scaling_results.json    # Complete raw JSON output containing per-question metrics & latencies
└── generator_scaling_report.md       # Final comparative Markdown analysis report
```
