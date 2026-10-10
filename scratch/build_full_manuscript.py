import os
import sys

ROOT_DIR = r"d:\Rag book work"
PROCESSED_DIR = r"d:\Rag book work\data\processed\rag_baseline"

MANUSCRIPT_PATH_PROCESSED = os.path.join(PROCESSED_DIR, "rag_book_final_manuscript.md")
MANUSCRIPT_PATH_ROOT = os.path.join(ROOT_DIR, "rag_book_final_manuscript.md")
QA_CHECKLIST_PATH = os.path.join(PROCESSED_DIR, "rag_book_publication_qa.md")

# 1. Load Front Matter
sys.path.append(os.path.join(ROOT_DIR, "scratch"))
from compile_front_matter import FRONT_MATTER
from compile_part1_part2 import PART_1_AND_2_TEXT

# 2. Read Chapter 7
with open(os.path.join(PROCESSED_DIR, "chapter_7_what_actually_matters.md"), "r", encoding="utf-8") as f:
    ch7_text = f.read()

# 3. Read Chapter 8 and apply audit qualifications
with open(os.path.join(PROCESSED_DIR, "chapter_8_vector_databases_frameworks.md"), "r", encoding="utf-8") as f:
    ch8_text = f.read()

# Apply Chapter 8 Qualification 1 (HNSW Memory Sizing & GB vs GiB)
ch8_qual1_old = """*Example:* Scaling to $N = 10,000,000$ chunks using a $d = 768$ dimension embedding model:
- **Raw Vector Memory:** $10^7 \\times 768 \\times 4\\text{ bytes} = 30.72\\text{ GB}$.
- **HNSW Index Overhead ($M=32$):** Index structures add roughly $1.5\\times$ memory, requiring $30.72 \\times 1.5 = 46.08\\text{ GB}$.
- **Payload Cache & Operating System Buffer:** Recommends an additional 20--30% buffer, demanding a node with **at least 64 GB RAM**."""

ch8_qual1_new = """*Example:* Scaling to $N = 10,000,000$ chunks using a $d = 768$ dimension embedding model:
- **Raw Vector Memory:** $10^7 \\times 768 \\times 4\\text{ bytes} = 30.72 \\times 10^9\\text{ bytes} = 30.72\\text{ GB}$ in decimal base-10 notation (which equals **$28.61\\text{ GiB}$** in binary gibibytes).
- **HNSW Index Link Overhead ($M=32$):** HNSW graph link storage requires approximately $(M \\times 4\\text{ to } 8\\text{ bytes}) \\approx 128\\text{ to } 256\\text{ bytes}$ per vector node, independent of vector dimension $d$. For $d=768$ ($3,072\\text{ bytes}$ per raw vector), graph links add $\\approx +8.3\\%$ memory overhead. Accounting for internal node metadata, allocator alignment padding, and index overheads (typically $+20\\%$ to $+50\\%$ total memory), indexing requires $30.72\\text{ GB} \\times 1.5 = 46.08\\text{ GB}$ (or $42.91\\text{ GiB}$).
- **Payload Cache & Operating System Buffer:** Recommends an additional 20--30% buffer for operating system page cache, demanding a host node with **at least 64 GB RAM** (or 64 GiB)."""

if ch8_qual1_old in ch8_text:
    ch8_text = ch8_text.replace(ch8_qual1_old, ch8_qual1_new)

# Apply Chapter 8 Qualification 2 (pgvector RDBMS distinction)
ch8_qual2_old = """| **Primary Deployment Model** | In-Process C++/Python Library | Standalone Server Cluster / Managed Cloud Service |"""
ch8_qual2_new = """| **Primary Deployment Model** | In-Process C++/Python Library | Standalone Server Cluster / Managed Cloud Service / RDBMS Extension |"""
if ch8_qual2_old in ch8_text:
    ch8_text = ch8_text.replace(ch8_qual2_old, ch8_qual2_new)

# Add explicit pgvector note to Chapter 8 Section 8.6
ch8_qual2_note_old = """This embedded design offered significant advantages for benchmarking: zero network latency, microsecond retrieval execution, and full control over scoring logic. However, enterprise deployment requires evaluating the trade-offs between embedded libraries and dedicated vector database servers."""
ch8_qual2_note_new = """This embedded design offered significant advantages for benchmarking: zero network latency, microsecond retrieval execution, and full control over scoring logic. However, enterprise deployment requires evaluating the trade-offs between embedded libraries and dedicated vector database servers.

> [!NOTE]
> **Architectural Qualification on `pgvector`:**  
> Table 8.1 categorizes `pgvector` alongside standalone vector databases. Architecturally, `pgvector` is an extension executing within PostgreSQL (a relational RDBMS), leveraging Postgres WAL, MVCC, `shared_buffers`, and SQL query planning. In contrast, dedicated vector databases (such as `Qdrant` or `Milvus`) are purpose-built vector engines in Rust/Go/C++ with custom segment management and native gRPC interfaces."""
if ch8_qual2_note_old in ch8_text:
    ch8_text = ch8_text.replace(ch8_qual2_note_old, ch8_qual2_note_new)

# Apply Chapter 8 Qualification 3 (Framework telemetry qualification)
ch8_qual3_old = """3. **Loss of Fine-Grained Latency Instrumentation:**  
   As established in our benchmark latency breakdown ([`final_benchmark_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json)), online latency is distributed across distinct pipeline stages:
   $$\\text{Latency}_{\\text{Total}} = \\text{Latency}_{\\text{Dense}} (45.6\\text{ms}) + \\text{Latency}_{\\text{BM25}} (1.1\\text{ms}) + \\text{Latency}_{\\text{Rerank}} (788.7\\text{ms}) + \\text{Latency}_{\\text{LLM}} (6436.8\\text{ms})$$
   Monolithic framework chains (e.g., `chain.invoke()`) blur these stage boundaries, returning only total wall-clock time. This obscures whether a performance bottleneck stems from dense search, cross-encoder reranking, or LLM generation."""

ch8_qual3_new = """3. **Loss of Fine-Grained Latency Instrumentation:**  
   As established in our benchmark latency breakdown ([`final_benchmark_results.json`](file:///d:/Rag%20book%20work/data/processed/rag_baseline/final_benchmark/final_benchmark_results.json)), online latency is distributed across distinct pipeline stages:
   $$\\text{Latency}_{\\text{Total}} = \\text{Latency}_{\\text{Dense}} (45.6\\text{ms}) + \\text{Latency}_{\\text{BM25}} (1.1\\text{ms}) + \\text{Latency}_{\\text{Rerank}} (788.7\\text{ms}) + \\text{Latency}_{\\text{LLM}} (6436.8\\text{ms})$$
   *Framework Telemetry Qualification:* Default out-of-the-box convenience chain wrappers (such as `chain.invoke()`) blur these stage boundaries, returning only total wall-clock time. While modern framework ecosystems (LangChain/LangSmith, LlamaIndex, Haystack 2.x) support callback handlers and OpenTelemetry integration, developers using standard convenience chains must explicitly instrument custom event hooks to isolate whether a latency spike originates from dense search, cross-encoder reranking, or LLM generation."""
if ch8_qual3_old in ch8_text:
    ch8_text = ch8_text.replace(ch8_qual3_old, ch8_qual3_new)

# Apply Chapter 8 Qualification 4 (Pre-filtering HNSW graph qualification)
ch8_qual4_old = """1. **Pre-Filtering:**  
   The metadata engine evaluates relational predicates first, extracting matching Document IDs. Vector search is then executed strictly over this subset.  
   *Failure Mode:* If the matching subset is small, vector indices like HNSW cannot be traversed effectively because graph links are broken, degenerating retrieval into an expensive brute-force scan over isolated nodes."""

ch8_qual4_new = """1. **Pre-Filtering:**  
   The metadata engine evaluates relational predicates first, extracting matching Document IDs. Vector search is then executed strictly over this subset.  
   *Failure Mode & Qualification:* On a single un-partitioned HNSW graph without payload-aware entry points, if the matching metadata subset is small, graph traversal fails because links between valid nodes are broken, degenerating retrieval into a brute-force scan over isolated nodes. Modern vector databases mitigate this by maintaining tenant-partitioned sub-graphs or payload-aware entry point indices."""
if ch8_qual4_old in ch8_text:
    ch8_text = ch8_text.replace(ch8_qual4_old, ch8_qual4_new)


# 4. Read Chapter 9
with open(os.path.join(PROCESSED_DIR, "chapter_9_graph_rag.md"), "r", encoding="utf-8") as f:
    ch9_text = f.read()

# 5. Read Chapter 10
with open(os.path.join(PROCESSED_DIR, "chapter_10_agentic_rag.md"), "r", encoding="utf-8") as f:
    ch10_text = f.read()

# 6. Read Chapter 11
with open(os.path.join(PROCESSED_DIR, "chapter_11_production_rag.md"), "r", encoding="utf-8") as f:
    ch11_text = f.read()

# 7. Read Chapter 12
with open(os.path.join(PROCESSED_DIR, "chapter_12_future_of_rag.md"), "r", encoding="utf-8") as f:
    ch12_text = f.read()

# 8. Load Back Matter
BACK_MATTER = """# Back Matter

---

## References & Primary Source Links

1. **BeIR Benchmark Dataset:**  
   Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). "BeIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021 Datasets and Benchmarks Track*.  
   Dataset URI: [`https://huggingface.co/datasets/BeIR/hotpotqa`](https://huggingface.co/datasets/BeIR/hotpotqa)  
   Qrels URI: [`https://huggingface.co/datasets/BeIR/hotpotqa-qrels`](https://huggingface.co/datasets/BeIR/hotpotqa-qrels)

2. **HotpotQA Benchmark:**  
   Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP 2018*.

3. **FAISS (Facebook AI Similarity Search):**  
   Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 7(3), 535-547.  
   Documentation: [`https://faiss.ai/`](https://faiss.ai/)

4. **Dense Sentence Embeddings (`BAAI/bge-small-en-v1.5`):**  
   Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). "C-Pack: Packaged Resources for General Chinese Embeddings." *arXiv preprint arXiv:2309.07597*.  
   Hugging Face Model: [`https://huggingface.co/BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)

5. **Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`):**  
   Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*.  
   Hugging Face Model: [`https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

6. **FLAN-T5 Instruction-Tuned Generators:**  
   Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Wei, J. (2022). "Scaling Instruction-Finetuned Language Models." *arXiv preprint arXiv:2210.11416*.  
   Hugging Face Model: [`https://huggingface.co/google/flan-t5-large`](https://huggingface.co/google/flan-t5-large)

7. **BM25 Lexical Retrieval:**  
   Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Information Retrieval*, 3(4), 227-296.

8. **Graph RAG & Knowledge Extraction:**  
   Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., ... & Larson, J. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv preprint arXiv:2404.16130*.

9. **Agentic RAG & Dynamic Control Loops:**  
   Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.

---

## Datasets & Software Stack Manifest

### 1. Dataset Manifest Summary (`data/manifest.json`)
- **Corpus Source:** `BeIR/hotpotqa` (5.23M document corpus).
- **Linked Pilot Sampling Seed:** `42`.
- **Pilot Sample Size:** 50 Benchmark Queries, 100 Corpus Passages, 100 Qrels Mappings.
- **Ground Truth Coverage:** **100.0%** (All gold passages present in pilot corpus).
- **Final Benchmark Partition:** Leakage-safe split of 10 Dev Queries (20%) / 40 Test Queries (80%).

### 2. Software Virtual Environment Dependencies
- **Python Runtime:** Python 3.11+
- **Core ML Libraries:** `torch>=2.0.0`, `transformers>=4.30.0`, `sentence-transformers>=2.2.2`, `faiss-cpu>=1.7.4`, `rank-bm25>=0.2.2`.
- **Data Engineering:** `pyarrow>=12.0.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `pyyaml>=6.0`.

---

## Reproducibility & Protocol Execution Notes

All experimental results reported in this book can be reproduced using the execution scripts located under `scripts/` with seed `42`:

### Step-by-Step Reproduction Pipeline:

```powershell
# 1. Activate Environment & Load Linked Dataset
.\.venv\Scripts\python.exe scripts/load_pilot_dataset.py

# 2. Execute Chapter 7 Component Ablations (Runs 0--4)
.\.venv\Scripts\python.exe scripts/run_hybrid_experiment.py

# 3. Execute Authoritative Final Benchmark (PROTO_FINAL_BENCHMARK)
# Evaluates frozen stack: BGE + BM25 + Cross-Encoder Reranker -> flan-t5-large (greedy T=0.0)
# Split: 10 Dev / 40 Test (N=40), B=1,000 bootstrap resamples
```

#### Protocol Parameter Reference:
- **Seed:** `42`
- **Dense Retriever:** `BAAI/bge-small-en-v1.5` ($L_2$-normalized `IndexFlatIP`)
- **Lexical Retriever:** `BM25Okapi` ($\alpha=0.5$ score fusion)
- **Candidate Pool:** $K_{\text{init}}=15$
- **Reranker:** `ms-marco-MiniLM-L-6-v2` truncated to $K_{\text{final}}=5$
- **Generator:** `google/flan-t5-large` ($T=0.0$, `max_new_tokens=50`)

---

## Repository & Code Structure Guide

```text
d:\Rag book work\
├── README.md                   # Project documentation & execution guide
├── requirements.txt            # Python dependencies
├── config/
│   └── config.yaml             # Dataset paths, seed 42, and baseline hyperparams
├── data/
│   ├── processed/
│   │   ├── hotpotqa_pilot_queries.json
│   │   ├── hotpotqa_pilot_corpus.json
│   │   ├── hotpotqa_pilot_qrels.json
│   │   ├── chapter_7_ablation/         # Master results JSON & ablation outputs
│   │   ├── final_benchmark/           # Authoritative benchmark JSON, logs, & CIs
│   │   ├── full_book_final_audit.md   # Complete manuscript audit report
│   │   └── rag_book_final_manuscript.md # Publication-ready assembled manuscript
│   └── manifest.json           # Linked sampling schema manifest
├── src/
│   ├── data/
│   │   ├── loader.py           # PyArrow streaming loader & schema inspection
│   │   └── manifest.py         # Manifest validator
│   ├── rag/
│   │   ├── chunker.py          # Sliding window text chunking (250w/50o)
│   │   ├── retriever.py        # FAISS dense retriever & evaluation metrics
│   │   └── generator.py        # FLAN-T5 generator wrapper (greedy decoding)
│   └── utils/
│       └── logging_config.py   # Microsecond timing and logger setup
└── scripts/                    # Experiment execution scripts
```

---

## Technical Glossary

- **Approximate Nearest Neighbors (ANN):** Vector index algorithms (such as HNSW or IVF-PQ) that trade exact similarity recall for sub-linear $O(\log N)$ query scaling.
- **Contextual Retrieval:** Augmenting text chunks by prepending LLM-generated document summaries prior to embedding generation.
- **Cross-Encoder Reranking:** A full-attention transformer model (`ms-marco-MiniLM-L-6-v2`) that scores query-chunk pairs jointly, providing high precision re-ordering.
- **Dense Vector Retrieval:** Retrieval using low-dimensional continuous vector representations generated by neural bi-encoders (`bge-small-en-v1.5`).
- **Exact Match (EM):** Metric measuring the percentage of model outputs that match gold reference answers character-for-character.
- **Hit Rate @ K:** The percentage of benchmark queries for which at least one ground-truth relevant document is retrieved in the top-$K$ candidates.
- **HNSW (Hierarchical Navigable Small World):** A multi-layer proximity graph index structure enabling fast ANN vector search.
- **Hybrid Score Fusion:** Combining dense vector similarity scores and sparse BM25 lexical scores using Min-Max normalization and convex weighting ($\alpha=0.5$).
- **Inner Product ($S_{\text{IP}}$):** Dot product similarity metric $\mathbf{u} \cdot \mathbf{v}$. Identical to Cosine Similarity when vectors are $L_2$-normalized.
- **IVF-PQ (Inverted File Product Quantization):** Lossy vector compression technique partitioning space into Voronoi cells and quantizing sub-vectors into codebooks.
- **Personalized PageRank (PPR):** A random walk graph centrality algorithm used in Graph RAG to score node relevance relative to query seed entities.
- **Precision @ K:** The fraction of retrieved top-$K$ candidates that are ground-truth relevant.
- **RAG Triad:** Evaluation framework measuring Context Relevance, Groundedness (Faithfulness), and Answer Relevance.
- **Recall @ K:** The fraction of total ground-truth relevant documents retrieved in the top-$K$ candidates.
- **Reciprocal Rank Fusion (RRF):** An rank-based ensembling method $S_{\text{RRF}}(d) = \sum \frac{1}{k + r_i(d)}$ used to fuse rank lists from multiple retrieval tools.
- **Sparse Lexical Search:** Keyword-based retrieval using term frequency-inverse document frequency algorithms (`BM25Okapi`).
- **Time-to-First-Token (TTFT):** Online latency measurement recording the millisecond duration until the generator streams its first completion token.
- **Write-Ahead Logging (WAL):** A database persistence mechanism that appends write operations to an on-disk log prior to memory index insertion.
"""

# Assemble manuscript parts
manuscript_parts = [
    FRONT_MATTER,
    PART_1_AND_2_TEXT,
    "\n# Part III: Controlled Ablations & Authoritative Benchmarks\n\n---\n\n",
    ch7_text,
    "\n# Part IV: Architecture, Infrastructure, & Future Paradigms\n\n---\n\n",
    ch8_text,
    "\n---\n\n",
    ch9_text,
    "\n---\n\n",
    ch10_text,
    "\n---\n\n",
    ch11_text,
    "\n---\n\n",
    ch12_text,
    "\n---\n\n",
    BACK_MATTER
]

full_manuscript = "\n".join(manuscript_parts)

# Write to processed directory
with open(MANUSCRIPT_PATH_PROCESSED, "w", encoding="utf-8") as f:
    f.write(full_manuscript)

# Write to root directory
with open(MANUSCRIPT_PATH_ROOT, "w", encoding="utf-8") as f:
    f.write(full_manuscript)

print(f"Assembled manuscript saved to: {MANUSCRIPT_PATH_PROCESSED}")
print(f"Assembled manuscript saved to: {MANUSCRIPT_PATH_ROOT}")
print(f"Total Manuscript Character Count: {len(full_manuscript):,}")
print(f"Total Manuscript Line Count: {len(full_manuscript.splitlines()):,}")
