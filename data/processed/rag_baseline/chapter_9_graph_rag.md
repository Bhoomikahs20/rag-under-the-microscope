# Chapter 9: Graph RAG: When Relationships Matter More Than Similarity

**File Path:** `chapter_9_graph_rag.md` / `data/processed/rag_baseline/chapter_9_graph_rag.md`  
**Complementary References:**  
- [`chapter_7_what_actually_matters.md`](file:///d:/Rag%20book%20work/chapter_7_what_actually_matters.md) (System-Level Ablations & Benchmark Multi-Hop Failures)  
- [`chapter_8_vector_databases_frameworks.md`](file:///d:/Rag%20book%20work/chapter_8_vector_databases_frameworks.md) (Vector Database Architecture & Indexing Infrastructure)

---

## 9.1 Introduction: Beyond Similarity-Based Retrieval

In Chapter 7 ([`chapter_7_what_actually_matters.md`](file:///d:/Rag%20book%20work/chapter_7_what_actually_matters.md)) and Chapter 8 ([`chapter_8_vector_databases_frameworks.md`](file:///d:/Rag%20book%20work/chapter_8_vector_databases_frameworks.md)), we evaluated dense vector search (`bge-small-en-v1.5`), lexical keyword retrieval (`BM25Okapi`), and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`). Our empirical benchmark demonstrated that combining dense semantic retrieval with lexical search and reranking achieves **100% top-5 retrieval recall** on the HotpotQA test set, driving end-to-end answer accuracy to **90.0%** when coupled with an instruction-tuned generator (`flan-t5-large`).

However, analyzing the remaining errors on our held-out test set reveals a fundamental structural pattern: **all remaining errors were multi-hop reasoning failures (`MH = 3` out of 4 total errors)**. For example:
- **Query QID `5a8c5d795542995e66a475d4`:** Required aggregating location facts across multiple distinct film studio entities ("London's Piccadilly" and "Pinewood Studios").
- **Query QID `5a71532e5542994082a3e7d6`:** Required tracking multi-step co-star relationships across movie titles (Joseph Gordon-Levitt acting alongside Heath Ledger and Julia Stiles in *10 Things I Hate About You*).

### The Fundamental Limit of Vector Similarity
Dense vector search operates by mapping query and document texts into a shared high-dimensional vector space $\mathbb{R}^d$, measuring proximity via inner product or cosine similarity. While highly effective for matching topically similar passages, vector search treats document chunks as **isolated points in space**. It evaluates whether a passage is *similar* to a query, not how concepts across different passages are *connected*.

When answering complex relational queries ("Which parent company owns the supplier that manufactures the component used in Product X?") or global thematic questions ("What are the recurring failure modes across all enterprise software incident reports?"), vector search often retrieves chunks that share keyword or semantic overlap but lack the explicit relational bridges necessary to connect multi-step facts.

```
                  VECTOR SIMILARITY VS. GRAPH CONNECTIVITY
┌────────────────────────────────────────────────────────────────────────┐
│ Vector Space: Chunks mapped as isolated spatial points                 │
│  Chunk A ("Pinewood Studios")  <--- (Cosine Dist) --->  Chunk B ("London")│
├────────────────────────────────────────────────────────────────────────┤
│ Graph Topology: Explicit entities and typed relational edges          │
│  (Casino Royale) ──[PRODUCED_BY]──► (Pinewood Studios) ──[LOCATED_IN]──► (Iver Heath)│
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 9.1: Vector Similarity Proximity vs Explicit Knowledge Graph Topology*

> [!IMPORTANT]
> **Methodological Disclaimer:**  
> While the multi-hop reasoning errors identified in Chapter 7 serve as theoretical motivation for exploring graph-based paradigms, **Graph RAG was NOT evaluated as part of our audited benchmark campaign**. We do NOT claim that Graph RAG would automatically eliminate these multi-hop failures without empirical validation. Graph RAG represents a specialized architectural paradigm with distinct engineering costs, index building overheads, and operational trade-offs.

---

## 9.2 Fundamentals of Knowledge Graphs & Graph Retrieval

### 9.2.1 Knowledge Graphs, Entities, Relations, and Triples

A **Knowledge Graph (KG)** is a directed, labeled graph structure defined as:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$$

where:
- $\mathcal{V}$ is the set of **Entity Nodes** (e.g., people, locations, organizations, concepts).
- $\mathcal{R}$ is the set of **Relation Types** (e.g., `PRODUCED_BY`, `LOCATED_IN`, `ACTED_IN`).
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ is the set of directed **Edges** representing **Fact Triples**:

$$\text{Triple } t = (h, r, t_{\text{tail}}) \in \text{Head Entity} \xrightarrow{\text{Relation}} \text{Tail Entity}$$

*Example Triples:*
- $(\text{Casino Royale}, \text{PRODUCED\_BY}, \text{Eon Productions})$
- $(\text{Eon Productions}, \text{BASED\_IN}, \text{Pinewood Studios})$
- $(\text{Pinewood Studios}, \text{LOCATED\_IN}, \text{Iver Heath, Buckinghamshire})$

#### Property Graphs vs. RDF Triplestores
1. **Property Graphs (e.g., Neo4j, Memgraph):** Nodes and edges store arbitrary key-value properties (e.g., node entity type, relation confidence score, source document chunk ID). This structure is widely preferred in modern Graph RAG implementations due to its flexibility.
2. **RDF Triplestores (e.g., Amazon Neptune, GraphDB):** Formal W3C standards using subject-predicate-object URIs, queried via SPARQL. Highly structured, but less flexible for unstructured document payloads.

---

### 9.2.2 Graph Construction & Ingestion Pipeline

Constructing a Knowledge Graph from an unstructured document corpus requires a four-stage pipeline:

```
                      GRAPH CONSTRUCTION PIPELINE
┌────────────────────────────────────────────────────────────────────────┐
│                        Unstructured Document Corpus                     │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Text Segmentation & Named Entity Recognition (NER / OpenIE / LLM)    │
│    Extract Entities: [Joseph Gordon-Levitt (Person), Heath Ledger...] │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Relation Extraction & Triplet Formation                             │
│    Extract Relation: (Joseph Gordon-Levitt, CO_STARRED_WITH, Heath Ledger)│
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Entity Resolution & Alias Disambiguation                            │
│    Merge Aliases: {"J. Gordon-Levitt", "Joe Levitt"} ──► Canonical Node│
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Graph Indexing & Dual Storage Integration                           │
│    Persist to Property Graph DB + Create Node Vector Index              │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 9.2: The Pipeline Architecture for Knowledge Graph Construction from Unstructured Text*

1. **Information Extraction (IE):**
   - **LLM-Based Extraction:** Prompting an LLM to identify entity nodes and relational triples directly from text chunks. LLM extraction captures nuanced, open-domain semantic relations, but incurs heavy token costs and potential edge hallucination.
   - **Rule / NLP-Based Extraction:** Utilizing Named Entity Recognition (NER) models (spaCy, Transformers) and dependency parsing (OpenIE). Fast and deterministic, but struggles with implicit or cross-sentence relations.
2. **Entity Resolution & Alias Disambiguation (ER):**
   - Identifies when different text strings refer to the same real-world entity (e.g., mapping `"J. Gordon-Levitt"`, `"Joe Levitt"`, and `"Joseph Gordon-Levitt"` to a single canonical node $v_i \in \mathcal{V}$).
   - *Failure Mode:* Failing to merge aliases creates disconnected "island" subgraphs; overly aggressive merging creates false edge collapses that corrupt graph topology.
3. **Graph Storage & Embedding Integration:**
   - Property graph nodes are linked to their originating text chunk IDs in vector storage, creating a **Dual Vector-Graph Storage Architecture**.

---

### 9.2.3 Graph Retrieval & Neighborhood Expansion

Given a user query $\mathbf{q}$, Graph RAG retrieves structural context through a multi-step traversal workflow:

```
                        GRAPH RETRIEVAL WORKFLOW
┌────────────────────────────────────────────────────────────────────────┐
│                              User Query                                │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Seed Entity Linking: Map Query Terms ──► Seed Graph Nodes (v_seed)  │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. k-Hop Neighborhood Expansion: Traverse Edges up to Distance k=2      │
│    Subgraph G_q = { v in V | dist(v_seed, v) <= k }                    │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Centrality Scoring & Subgraph Pruning (Personalized PageRank)       │
│    Filter Top-N Structural Paths based on Query Similarity & Edge Weights│
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Triple Linearization & Prompt Construction                          │
│    Convert Subgraph Triples ──► Natural Language Context for Generator │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 9.3: The Step-by-Step Subgraph Traversal and Context Linearization Workflow*

1. **Seed Entity Extraction & Linking:**
   The query $\mathbf{q}$ is parsed to identify entity mentions. Seed nodes $V_{\text{seed}} \subset \mathcal{V}$ are located using exact string matching or dense similarity search against entity node vector embeddings.
2. **$k$-Hop Neighborhood Expansion:**
   Traverses outgoing and incoming edges up to distance $k$ (typically $k=1$ or $k=2$), constructing a localized query subgraph $\mathcal{G}_{\mathbf{q}}$:
   $$\mathcal{G}_{\mathbf{q}} = \{ v \in \mathcal{V} \mid \min_{v_s \in V_{\text{seed}}} \text{dist}(v_s, v) \le k \}$$
3. **Personalized PageRank (PPR) & Path Scoring:**
   To prevent node explosion, nodes within $\mathcal{G}_{\mathbf{q}}$ are ranked using Personalized PageRank biased toward the seed set $V_{\text{seed}}$:
   $$\mathbf{p}^{(t+1)} = (1 - \alpha) \mathbf{M} \mathbf{p}^{(t)} + \alpha \mathbf{s}_{\mathbf{q}}$$
   where $\mathbf{s}_{\mathbf{q}}$ is the initial query-similarity score vector over seed nodes, $\mathbf{M}$ is the adjacency transition matrix, and $\alpha \in (0, 1)$ is the random walk restart probability.
4. **Graph-Aware Context Construction (Triple Linearization):**
   Top-ranked subgraph paths and triples are converted into natural language strings for LLM generator input:
   $$\text{"Pinewood Studios is located in Iver Heath. Casino Royale was produced by Eon Productions at Pinewood Studios."}$$

---

## 9.3 Architectural Taxonomy: Vector RAG, Hybrid RAG, and Graph RAG

To understand where Graph RAG fits into the system architecture landscape, we contrast it against standard vector RAG and the finalized lexical-dense hybrid stack established in Chapter 7.

### Table 9.1: Comprehensive Architectural Comparison Across Retrieval Paradigms

| Architectural Dimension | Standard Vector RAG | Lexical + Dense Hybrid RAG (Chapter 7 Final Baseline) | Graph RAG (Pure Knowledge Graph) | Dual-Index Graph-Vector Hybrid RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Index Structure** | High-Dimensional Vector Index (`faiss.IndexFlatIP`) | Dense Vector Index + Inverted `BM25Okapi` Index | Property Graph / RDF Triplestore (Neo4j / Memgraph) | Dual Index: Vector Index (Chunks) + Property Graph (KG) |
| **Search Mechanics** | Nearest Neighbor Cosine Distance | Reciprocal Rank Fusion (RRF) / Convex Score Sum | $k$-Hop Graph Traversal / Subgraph Extraction | Seed Vector Search $\to$ $k$-Hop Expansion $\to$ Cross-Encoder Rerank |
| **Multi-Hop Traversal** | Poor (Relies on serendipitous chunk co-location) | Moderate (Cross-Encoder reranking over $K_{\text{init}}=15$) | **High** (Explicit graph edge traversal across entities) | **High** (Combines explicit graph paths with text chunk context) |
| **Global Summary Ability**| Poor (Retrieves local top-$K$ chunks only) | Poor (Bounded by top-$K$ chunk context window) | **High** (Graph community summaries over entity clusters) | **High** (Hierarchical community summaries + text passages) |
| **Offline Indexing Cost** | Low ($O(N \cdot d)$ embedding encode time) | Low (Embedding encode + BM25 inverted table build) | **Extremely High** (LLM extraction tokens + ER logic) | **Extremely High** (Dual vector encoding + LLM KG extraction) |
| **Online Search Latency** | Ultra-Fast (45.6 ms mean vector search) | Fast (45.6 ms dense + 1.1 ms BM25 + 788.7 ms rerank) | Variable (10 ms to 500 ms depending on graph depth) | Moderate to High (Dense search + graph expansion + rerank) |
| **Maintenance & Updates** | Simple (Vector upserts / index rebuilds) | Simple (Vector upserts + BM25 table updates) | Complex (Graph schema updates + entity re-resolution) | Complex (Synchronizing vector indices with graph edges) |

---

## 9.4 Graph RAG Architectures: Dual-Index & Graph-Augmented Pipelines

In enterprise production, pure Graph RAG is rarely deployed in isolation. Instead, systems combine vector indices with knowledge graphs using one of three architectural patterns:

```
                    DUAL-INDEX GRAPH-VECTOR HYBRID RAG
┌────────────────────────────────────────────────────────────────────────┐
│                             User Query                                 │
└────────────────────────────────────────────────────────────────────────┘
                   │                                  │
                   ▼                                  ▼
┌─────────────────────────────────────┐    ┌────────────────────────────┐
│ 1. Vector Retriever (Chunks)        │    │ 2. Graph Retriever (KG)    │
│    • BGE-Small Vector Index         │    │  • Seed Entity Lookup      │
│    • Returns Top-K Text Chunks      │    │  • 2-Hop Subgraph Expansion│
└─────────────────────────────────────┘    └────────────────────────────┘
                   │                                  │
                   └──────────────────┬───────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Context Fusion & Cross-Encoder Reranking                            │
│    Combine Text Chunks + Linearized Triples into Generator Prompt     │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Generator LLM (google/flan-t5-large, T=0.0)                         │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 9.4: Dual-Index Architectural Pattern Integrating Vector Retrieval with Knowledge Graph Expansion*

1. **Graph-Augmented Vector RAG:**  
   Dense vector search retrieves top-$K$ text chunks first. Entities detected in retrieved chunks trigger local $1$-hop graph expansions, appending missing structural facts (e.g., parent company, location) to the generator prompt.
2. **Vector-Augmented Graph RAG:**  
   Graph traversal extracts a candidate subgraph first based on query entity linking. Vector similarity is then used to rank and prune candidate node text blocks.
3. **Dual-Index Parallel Hybrid RAG:**  
   Queries execute in parallel across a chunk vector index and a knowledge graph. Retrieved text passages and linearized triples are merged and reranked via a cross-encoder before context construction.

---

## 9.5 Engineering Trade-offs, Failure Modes, & Operational Challenges

While Graph RAG offers strong capabilities for explicit multi-step relationship reasoning, it introduces severe infrastructure challenges that engineering teams must navigate.

```
                       GRAPH RAG FAILURE TAXONOMY
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Graph Traversal Explosion                                           │
│    High-degree hub nodes cause exponential 2-hop node expansion        │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Entity Disambiguation Collapse                                      │
│    Failing to resolve aliases creates disconnected island subgraphs    │
├────────────────────────────────────────────────────────────────────────┤
│ 3. False-Positive Relation Noise                                       │
│    LLM extraction generates spurious or hallucinated relations         │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Massive Indexing Cost & Latency Penalty                             │
│    LLM extraction costs $10x--100x$ more than dense vector embedding  │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 9.5: Taxonomy of Infrastructure and Traversal Failures in Graph RAG Systems*

### 9.5.1 Detailed Failure Modes:

1. **Graph Traversal Explosion ($k$-Hop Noise):**  
   If a seed entity is a high-degree "hub" node (e.g., `"United States"`, `"London"`, or `"Technology"`), expanding to $k=2$ hops can pull tens of thousands of neighbor nodes into the candidate pool. This floods the context window with irrelevant edges, spiking query latency and causing LLM attention dilution.
2. **Entity Disambiguation Collapse:**  
   - *False Merges:* Incorrectly merging distinct entities (e.g., merging `"John Smith"` the lawyer with `"John Smith"` the physician) creates corrupt shortcut edges across unrelated graph regions.
   - *False Splits:* Failing to resolve aliases (e.g., treating `"IBM"` and `"International Business Machines"` as separate nodes) breaks graph connectivity, preventing multi-hop traversals from completing.
3. **Graph Sparsity & Missing Edge Traps:**  
   Unstructured text often contains implicit relationships that automated extractors miss. If a key connecting edge is omitted during graph construction, graph search fails completely along that path.
4. **Computational & Financial Indexing Cost:**  
   Extracting structured triples from 10,000 document chunks using an LLM requires millions of prompt tokens. In production, Knowledge Graph extraction can cost **$10\times$ to $100\times$ more** in compute and time than generating dense vector embeddings (`bge-small-en-v1.5`).

---

## 9.6 Proposed Illustrative Graph RAG Experiment Protocol

To evaluate whether Graph RAG provides empirical accuracy gains over our finalized Chapter 7 baseline (BGE + BM25 + Cross-Encoder Reranking), we specify a proposed, reproducible experimental protocol.

> [!NOTE]
> **Protocol Status: PROPOSED / ILLUSTRATIVE EXPERIMENT ONLY.**  
> The protocol specification below represents a future experimental design. It was **NOT executed as part of the audited benchmark campaign in Chapter 7**, and no empirical benchmark claims are made regarding its performance.

```
                  PROPOSED ILLUSTRATIVE GRAPH EXPERIMENT PROTOCOL
┌────────────────────────────────────────────────────────────────────────┐
│ Protocol Identifier: PROTO_PROPOSED_GRAPH_HYBRID                       │
├────────────────────────────────────────────────────────────────────────┤
│ Corpus & Split: Leakage-Safe 10 Dev / 40 Test HotpotQA Split           │
│ Knowledge Graph: Neo4j Property Graph built via LLM Triple Extraction  │
│ Vector Retriever: BAAI/bge-small-en-v1.5 (250w/50o Chunks)            │
│ Candidate Pool: Top-10 Vector Chunks + Top-10 Linearized Graph Paths   │
│ Reranker: Cross-Encoder (ms-marco-MiniLM-L-6-v2) to K_final=5          │
│ Generator: google/flan-t5-large (Greedy decoding T=0.0)                │
└────────────────────────────────────────────────────────────────────────┘
```

### Proposed Evaluation Metrics:
1. **Subgraph Path Recall@K:** Percentage of ground-truth multi-hop entity chains captured within retrieved subgraphs.
2. **Multi-Hop Error Delta:** Target evaluation specifically measuring accuracy gains on the 3 multi-hop reasoning failures (`MH = 3`) identified in Chapter 7 auditing.
3. **Indexing Token Overhead:** Measurement of total LLM token consumption and wall-clock time required for KG construction versus standard vector index builds.

---

## 9.7 Decision Framework: When to Use Graph RAG vs. Standard RAG

Adopting Graph RAG adds substantial operational overhead. Engineering teams should evaluate their retrieval requirements against explicit system criteria before choosing a graph architecture.

```
                      GRAPH RAG DECISION FRAMEWORK
                               Query & Data Type
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
       Standard Factoid / Passage                      Explicit Relational /
       Similarity Queries                              Multi-Entity Networks
                 │                                           │
                 ▼                                           ▼
     Is Corpus Structure High?                       Are Entities Highly Connected?
         │             │                                 │             │
        YES            NO                               YES            NO
         │             │                                 │             │
         ▼             ▼                                 ▼             ▼
   Hybrid RAG    Dense Vector RAG                   Graph-Vector    Standard Hybrid
   (Chapter 7)   (Standard)                         Hybrid RAG      RAG (Chapter 7)
```
*Figure 9.6: Pragmatic Decision Framework for Evaluating Graph RAG Adoption*

### Table 9.2: Pragmatic Selection Framework: Standard RAG vs Hybrid RAG vs Graph RAG

| System Criterion | Standard Vector RAG | Lexical + Dense Hybrid (Chapter 7 Final Baseline) | Dual-Index Graph-Vector Hybrid RAG |
| :--- | :--- | :--- | :--- |
| **Target Query Type** | Simple passage similarity & factoid lookup | Keyword, semantic, and multi-document search | Multi-step entity traversal, supply chain lineage |
| **Corpus Characteristics** | Unstructured articles, documentation, blogs | Unstructured documents with structural titles | Semi-structured data, connected entity networks |
| **Multi-Hop Traversal** | Low | Moderate (Cross-encoder reranking over $K_{\text{init}}=15$) | **High** (Explicit $k$-hop path expansion over edges) |
| **Development & Index Cost**| Low ($1\times$) | Low ($1.1\times$) | **Extremely High ($10\times$--$100\times$)** |
| **Query Latency** | Low (< 50 ms) | Low to Moderate (2.5s total w/ generation) | High (+200 ms to +1000 ms graph traversal overhead) |
| **Maintenance Complexity** | Low | Low | High (Graph schema maintenance, entity resolution) |

---

## 9.8 Concise Engineering Checklist for Graph RAG

Use this checklist before implementing Graph RAG in a production environment:

- [ ] **Empirical Failure Audit Completed:** Verify via log inspection that existing vector/hybrid retrieval failures stem from missing relational paths, rather than generator capacity limits (Chapter 7 showed generator scaling eliminated 87.5% of reasoning errors without graph infrastructure).
- [ ] **Entity Resolution Pipeline Validated:** Ensure coreference resolution and alias disambiguation correctly merge entity variants without over-collapsing distinct nodes.
- [ ] **$k$-Hop Traversal Budgeted:** Enforce strict caps on traversal depth ($k \le 2$) and node degree limits to prevent graph traversal explosion on hub entities.
- [ ] **Dual-Index Decoupled:** Maintain decoupled indices for vector chunk text and property graph entity nodes.
- [ ] **Triple Linearization Audited:** Inspect linearized graph context strings to ensure they do not overflow generator context windows or inject prompt dilution.
- [ ] **Indexing Token Budget Approved:** Calculate total LLM token cost for entity/relation extraction across the full corpus prior to pipeline execution.
- [ ] **Fallback Retrieval Active:** Ensure queries with zero detected entities cleanly fall back to dense vector + lexical hybrid search.

---

## 9.9 Chapter Summary

- **Beyond Vector Proximity:** Dense vector similarity maps text passages as isolated spatial points. While highly effective for semantic matching, it cannot represent explicit, multi-step structural connections across disparate documents.
- **Knowledge Graph Mechanics:** Graph RAG structures information into entity nodes and relation triples $(h, r, t)$. Retrieval extracts seed entities, expands localized $k$-hop subgraphs, ranks paths via Personalized PageRank, and linearizes graph triples into LLM prompts.
- **Engineering Challenges:** Graph RAG introduces heavy infrastructure penalties: massive LLM indexing costs ($10\times$--$100\times$ vector embedding), vulnerability to entity resolution errors, graph sparsity, and potential traversal explosion on hub entities.
- **Empirical Perspective:** In Chapter 7, scaling generator capacity (`flan-t5-large`) eliminated 87.5% of multi-hop errors over a hybrid retriever without adding graph infrastructure. Graph RAG represents a powerful but complex specialized architecture that should be adopted only when explicit relational structures cannot be captured by lexical-dense hybrid search.
