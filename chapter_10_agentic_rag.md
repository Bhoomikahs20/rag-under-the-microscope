# Chapter 10: Agentic RAG: Adaptive Retrieval, Planning, and Tool Use

**File Path:** `chapter_10_agentic_rag.md` / `data/processed/rag_baseline/chapter_10_agentic_rag.md`  
**Complementary References:**  
- [`chapter_7_what_actually_matters.md`](file:///d:/Rag%20book%20work/chapter_7_what_actually_matters.md) (System-Level Ablations & Benchmark Multi-Hop Failures)  
- [`chapter_8_vector_databases_frameworks.md`](file:///d:/Rag%20book%20work/chapter_8_vector_databases_frameworks.md) (Vector Database Architecture & Indexing Infrastructure)  
- [`chapter_9_graph_rag.md`](file:///d:/Rag%20book%20work/chapter_9_graph_rag.md) (Knowledge Graph Topology & Structural Multi-Hop Retrieval)

---

## 10.1 Introduction: From Fixed Pipelines to Dynamic Agents

In Chapter 7 ([`chapter_7_what_actually_matters.md`](file:///d:/Rag%20book%20work/chapter_7_what_actually_matters.md)), Chapter 8 ([`chapter_8_vector_databases_frameworks.md`](file:///d:/Rag%20book%20work/chapter_8_vector_databases_frameworks.md)), and Chapter 9 ([`chapter_9_graph_rag.md`](file:///d:/Rag%20book%20work/chapter_9_graph_rag.md)), we evaluated fixed-pipeline RAG architectures. In a conventional fixed pipeline, control flow is static and deterministic:

$$\text{Query } \mathbf{q} \longrightarrow \text{Dense/Lexical Retrieve} \longrightarrow \text{Rerank } (K_{\text{final}}=5) \longrightarrow \text{Prompt Context} \longrightarrow \text{LLM Generate}$$

Our audited final benchmark in Chapter 7 demonstrated that this frozen hybrid pipeline achieves **100% top-5 retrieval recall** and **90.0% answer accuracy** on the HotpotQA test set. However, analyzing the four errors on the held-out test set revealed that **all remaining non-annotation errors were multi-hop reasoning failures (`MH = 3`)**:
- **Query QID `5a8c5d795542995e66a475d4`:** Required aggregating location facts across distinct film studio entities ("London's Piccadilly" and "Pinewood Studios").
- **Query QID `5a71532e5542994082a3e7d6`:** Required sequential multi-step entity resolution (identifying Joseph Gordon-Levitt in *Snowden*, then finding his co-stars in *10 Things I Hate About You*).

### The Structural Limit of Fixed RAG Pipelines
Fixed pipelines evaluate context in a single pass. If the initial user query is ambiguous, underspecified, or requires information that can only be discovered after retrieving an intermediate fact, a fixed pipeline cannot dynamically adapt. It cannot execute a second follow-up retrieval step, select a different retrieval tool, or verify whether the gathered context is sufficient before generating a final response.

```
                      FIXED RAG VS. AGENTIC RAG CONTROL FLOW
┌────────────────────────────────────────────────────────────────────────┐
│ Fixed Pipeline RAG: Single-pass static execution graph                 │
│  User Query ──► Dense/BM25 Search ──► Rerank K=5 ──► LLM Generate      │
├────────────────────────────────────────────────────────────────────────┤
│ Agentic RAG: Dynamic adaptive loop with self-reflection & tool routing │
│  User Query ──► Plan Sub-Queries ──► Select Tool ──► Verify Context    │
│                       ▲                                    │           │
│                       └────── Refine Query (if needed) ────┘           │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 10.1: Linear Fixed-Pipeline Control Flow vs. Dynamic Agentic Control Loops*

### Genuine Adaptive Retrieval vs. Superficial Agent Wrappers
It is critical to distinguish **genuine adaptive retrieval** from superficial framework wrappers. Simply wrapping a fixed RAG chain inside an agent execution wrapper (e.g., calling `Agent(tools=[FixedRAGPipeline()])`) does **not** make a system adaptive if the underlying retrieval strategy remains single-pass and unmanaged.

Genuine Agentic RAG empowers the language model as an **autonomous decision engine** that dynamically:
1. Decomposes complex user requests into sequential sub-goals.
2. Selects between distinct tools (e.g., dense vector search, lexical BM25, SQL queries, graph APIs, code execution).
3. Evaluates retrieved evidence for completeness and consistency.
4. Reformulates queries or halts execution based on explicit stopping criteria.

> [!IMPORTANT]
> **Methodological Disclaimer:**  
> While multi-hop reasoning failures in Chapter 7 provide theoretical motivation for adaptive control loops, **Agentic RAG was NOT evaluated as part of our audited final benchmark campaign**. We make NO empirical claim that Agentic RAG would improve accuracy beyond our frozen **90.0%** baseline. Agentic RAG introduces substantial operational trade-offs, including multi-second latency penalties, exponential token growth, and complex trajectory failure modes.

---

## 10.2 Core Mechanics of Agentic RAG

Agentic RAG replaces static execution with a dynamic **Plan-Act-Verify-Refine** loop:

```
                      AGENTIC RAG DECISION LOOP
┌────────────────────────────────────────────────────────────────────────┐
│                              User Query                                │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Question Decomposition & Sub-Query Planning                         │
│    Decompose Query Q ──► Sub-Plan P = {q_1, q_2, ..., q_k}              │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Dynamic Tool Selection & Execution                                  │
│    Route Sub-Query q_i ──► [Dense Vector / BM25 / Graph API / SQL]     │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Evidence Verification & Self-Reflection                             │
│    Verify(E_retrieved, q_i) in {SUFFICIENT, INCOMPLETE, CONTRADICTORY} │
└────────────────────────────────────────────────────────────────────────┘
                       │                     │
               NO (Incomplete)           YES (Sufficient)
                       │                     │
                       ▼                     ▼
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│ Refine Query / Invoke Tool  │   │ Accumulate Evidence & Next Sub-Query │
└─────────────────────────────┘   └──────────────────────────────────────┘
                       │                     │
                       └──────────┬──────────┘
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Deterministic Stopping Criterion Check & Final Generation           │
│    Max Steps Met OR Goal Satisfied ──► Formulate Final Output          │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 10.2: Detailed Architecture of the Adaptive Control Loop, Verification Cycle, and Tool Execution Engine*

### 10.2.1 Question Decomposition & Sub-Query Planning

When presented with a complex multi-hop query, the agent breaks the problem down into a sequence of atomic sub-questions:

$$\mathcal{P} = \{q_1, q_2, \dots, q_k\}$$

*Worked Example (HotpotQA Multi-Hop Question):*  
Query: *"In what year did the actor who played Edward Snowden act in a movie with Heath Ledger and Julia Stiles?"*

1. **Step 1 (Sub-Query $q_1$):** `"Who played Edward Snowden in the movie Snowden?"`  
   - *Tool Action:* Dense vector search over movie corpus.  
   - *Retrieved Fact:* `Joseph Gordon-Levitt`.
2. **Step 2 (Sub-Query $q_2$):** `"Which movie co-starred Joseph Gordon-Levitt, Heath Ledger, and Julia Stiles?"`  
   - *Tool Action:* Lexical BM25 search for `Joseph Gordon-Levitt AND Heath Ledger AND Julia Stiles`.  
   - *Retrieved Fact:* `10 Things I Hate About You (released in 1999)`.
3. **Step 3 (Final Synthesis):** Extract target year: `1999`.

---

### 10.2.2 Dynamic Tool Selection & Multi-Source Routing

Rather than routing every query to a single vector index, an agent selects the most appropriate tool based on query characteristics:

- **Dense Vector Search (`bge-small-en-v1.5`):** Selected for broad semantic intent and conceptual similarity.
- **Lexical Search (`BM25Okapi`):** Selected for exact keyword matching, proper nouns, part numbers, or specific entity IDs.
- **Knowledge Graph API (Neo4j):** Selected for explicit $k$-hop relational traversals across multi-entity networks (as detailed in Chapter 9).
- **SQL / Relational Database Tool:** Selected for structured numerical aggregation, filtering, or tabular data.
- **Code Execution Sandbox (Python REPL):** Selected for complex mathematical calculations or data transformation.

---

### 10.2.3 Evidence Verification & Self-Reflection

After executing a tool call, the agent evaluates the quality and completeness of retrieved evidence:

$$\text{Verify}(E_{\text{retrieved}}, q_i) \in \{\text{SUFFICIENT}, \text{INCOMPLETE}, \text{CONTRADICTORY}\}$$

- **Sufficient:** The retrieved context directly answers sub-query $q_i$. The agent stores the intermediate result and proceeds to $q_{i+1}$.
- **Incomplete:** The context lacks essential details. The agent executes a **query reformulation step** ($q_i \to q_i'$), altering search terms or switching retrieval tools.
- **Contradictory:** Retrieved sources conflict. The agent generates a disambiguation query or retrieves additional corroborating passages.

---

### 10.2.4 Deterministic Stopping Criteria & Guardrails

To prevent runaway execution loops, agentic systems enforce strict, programmatic termination rules:

1. **Goal Satisfaction:** All sub-queries in plan $\mathcal{P}$ verified as `SUFFICIENT`.
2. **Step Budget Limit ($S_{\text{max}}$):** Hard cap on maximum allowed iterations (typically $S_{\text{max}} \le 4$).
3. **Context Token Cap ($T_{\text{max}}$):** Hard cap on cumulative context tokens to prevent generator window overflow.
4. **Wall-Clock Timeout:** Maximum execution time limit (e.g., 15.0 seconds).

---

## 10.3 Taxonomy of RAG Paradigms: Fixed to Fully Agentic

### Table 10.1: Architectural Comparison Across Retrieval Paradigms

| Architectural Dimension | Fixed-Pipeline Hybrid RAG (Chapter 7 Baseline) | Iterative RAG | Planner-Based RAG | Fully Agentic RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Control Flow** | Linear, Static Graph ($1$-pass) | Loop over Fixed Pipeline ($N$-passes) | Decoupled Plan $\to$ Execute Phase | **Dynamic Adaptive Control Loop** |
| **Question Decomposition** | None (Raw query used directly) | Basic query reformulation | Static initial sub-plan | **Dynamic re-planning per step** |
| **Tool Selection** | Fixed single/hybrid retriever | Fixed retriever | Multiple fixed tools | **Dynamic multi-tool routing** |
| **Self-Reflection** | None | Simple coverage check | Basic step verification | **Full self-reflection & verification** |
| **Determinism** | **100% Deterministic** (Fixed seed, greedy) | High | Moderate | Lower (Trajectory variance) |
| **Latency Profile** | **Ultra-Fast & Predictable** (7.27s total, 45ms search) | Moderate ($2\times$--$3\times$ static latency) | Moderate ($2\times$--$4\times$ static latency) | **High & Variable** ($3\times$--$10\times$ static latency) |
| **Token Cost Profile** | **Low** ($1\times$ baseline token cost) | Moderate ($2\times$--$3\times$ token cost) | Moderate ($3\times$--$5\times$ token cost) | **Extremely High** ($5\times$--$10\times$ token cost) |
| **Primary Risk** | Multi-hop reasoning starvation | Redundant retrieval loops | Stale initial sub-plans | **Infinite loops, drift, error propagation** |

---

## 10.4 Engineering Failure Modes & Operational Risks

While agentic RAG provides dynamic adaptability, it introduces severe operational failure modes that do not exist in fixed pipelines.

```
                     AGENTIC RAG FAILURE TAXONOMY
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Infinite Retrieval Loops & Stopping Failure                         │
│    Agent fails self-reflection; loops continuously until timeout       │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Retrieval Drift & Semantic Divergence                               │
│    Intermediate sub-queries wander away from original query intent    │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Error Propagation Across Step Execution                             │
│    Hallucinated entity in Step 1 corrupts all downstream tool queries │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Context Window Explosion & Evidence Bloat                           │
│    Accumulating un-pruned tool outputs floods generator attention     │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 10.3: Primary Infrastructure Failure Modes in Agentic RAG Systems*

### 10.4.1 Detailed Failure Modes:

1. **Infinite Retrieval Loops & Self-Reflection Hallucination:**  
   If an agent's internal verifier incorrectly judges retrieved evidence as `INCOMPLETE` (even when complete), it continuously generates minor query variations, looping until hitting hard step caps or execution timeouts.
2. **Retrieval Drift & Semantic Divergence:**  
   During multi-step execution, intermediate sub-queries can gradually diverge from the original user intent. A sub-query error at Step 2 pulls irrelevant context into Step 3, leading the agent down a semantic tangent.
3. **Error Propagation Across Step Execution:**  
   If Step 1 incorrectly resolves an entity (e.g., identifying the wrong actor), all subsequent sub-queries and tool invocations inherit this hallucinated entity, making downstream recovery impossible.
4. **Context Window Explosion & Evidence Accumulation Bloat:**  
   Each iterative step appends retrieved text blocks, tool outputs, and reflection logs to the agent's context history. This exponential growth causes:
   - **High Token Costs:** Repeating long context strings across multiple LLM reasoning calls.
   - **Attention Dilution ("Lost in the Middle"):** As established in Chapter 7, overflowing generator context windows dilutes attention, degrading Exact Match accuracy.
5. **Severe Latency & Financial Cost Penalties:**  
   Executing 4 sequential agent reasoning cycles with LLM verifications multiplies online response time by $4\times$--$8\times$ (scaling from 7.27 seconds to 30+ seconds on CPU) and token expenses by $5\times$--$10\times$.

---

## 10.5 Proposed Illustrative Agentic RAG Experiment Protocol

To evaluate whether Agentic RAG delivers empirical accuracy gains over our audited Chapter 7 fixed baseline (BGE + BM25 + Cross-Encoder Reranking + `flan-t5-large`), we specify a proposed, reproducible experimental protocol.

> [!NOTE]
> **Protocol Status: PROPOSED / ILLUSTRATIVE EXPERIMENT ONLY.**  
> The protocol specification below represents a future experimental design. It was **NOT executed as part of the audited benchmark campaign in Chapter 7**, and no empirical performance claims are made regarding its results.

```
                 PROPOSED ILLUSTRATIVE AGENTIC EXPERIMENT PROTOCOL
┌────────────────────────────────────────────────────────────────────────┐
│ Protocol Identifier: PROTO_PROPOSED_AGENTIC_EVAL                       │
├────────────────────────────────────────────────────────────────────────┤
│ Corpus & Split: Leakage-Safe 10 Dev / 40 Test HotpotQA Split           │
│ Agent Framework: ReAct Adaptive Control Loop with Question Decomposition│
│ Available Tools: (1) BGE Vector Search, (2) BM25 Lexical Search        │
│ Verifier Module: LLM Verification Prompt (Threshold = 0.8)             │
│ Guardrails: Max Steps S_max = 4, Context Pruning Window = 2,000 tokens │
│ Generator: google/flan-t5-large (Greedy decoding T=0.0)                │
└────────────────────────────────────────────────────────────────────────┘
```

### Proposed Metric Evaluation Matrix:
1. **Trajectory Efficiency Ratio:** Ratio of minimal required steps to actual steps taken ($S_{\text{min}} / S_{\text{actual}}$).
2. **Sub-Query Execution Accuracy:** Accuracy of intermediate entity resolutions across step trajectories.
3. **Multi-Hop Failure Recovery Rate:** Target metric specifically measuring accuracy delta on the 3 multi-hop reasoning errors (`MH = 3`) identified in Chapter 7 test set auditing.
4. **Latency & Token Cost Overhead:** Precise measurement of additional millisecond latency and total prompt/completion tokens compared to the fixed Chapter 7 baseline (7.27s mean online latency).

---

## 10.6 Decision Framework: Fixed Pipeline vs. Agentic RAG

Deploying agentic loops introduces significant operational complexity. Engineering teams must balance query difficulty against latency, cost, and determinism requirements.

```
                     AGENTIC RAG DECISION FRAMEWORK
                               User Query
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
      Single-Pass Lookup /                     Multi-Step Complex /
      Factoid Questions                        Disparate Information
                │                                     │
                ▼                                     ▼
      Strict SLA & Latency?                  Uncertain Sub-Queries?
        │               │                        │              │
       YES              NO                      YES             NO
        │               │                        │              │
        ▼               ▼                        ▼              ▼
   Fixed Hybrid    Fixed Hybrid            Fully Agentic     Planner-Based
   (Chapter 7)     (Chapter 7)             RAG Pipeline      Iterative RAG
```
*Figure 10.4: Pragmatic Decision Framework for Evaluating Agentic RAG Deployment*

### Table 10.2: Pragmatic Infrastructure Selection Matrix

| System Criterion | Fixed-Pipeline Hybrid RAG (Chapter 7) | Planner / Iterative RAG | Fully Agentic RAG |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | Standard QA, document search, factoid lookup | Known multi-step queries (e.g., 2-hop templates) | Open-ended exploration, complex multi-source research |
| **Query Flexibility** | Fixed single-pass execution | Pre-planned sequential steps | **Dynamic adaptive step planning per query** |
| **Latency Budget** | Strict (< 3s to 8s total) | Moderate (8s to 15s) | Variable / Slow (15s to 45s+) |
| **Cost Sensitivity** | Low token consumption ($1\times$) | Moderate token consumption ($2\times$--$3\times$) | High token consumption ($5\times$--$10\times$) |
| **Determinism & Reproducibility**| **100% Deterministic** (Fixed seed, greedy) | High (Pre-planned steps) | Lower (Agent decision trajectory variance) |
| **Observability Complexity** | Low (Single-pass latency logging) | Moderate (Step-level tracing) | **High** (Full trajectory, tool logs, reflection audits) |

---

## 10.7 Observability & Telemetry Requirements for Agentic RAG

Monitoring agentic RAG in production requires telemetry capable of tracking non-linear execution trajectories:

```
                    AGENTIC TELEMETRY TRACING STACK
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Trajectory Tracing: Log Full Agent Action Graph                    │
│    [Step 1: Plan] ──► [Tool: DenseSearch] ──► [Verify: Incomplete]    │
│    ──► [Step 2: Re-Plan] ──► [Tool: BM25] ──► [Verify: Complete]      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Step-Level Latency & Token Accounting                              │
│    Track LLM reasoning overhead vs tool execution latency per step     │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Context Window Growth Telemetry                                    │
│    Monitor cumulative token bloat across reflection cycles             │
└────────────────────────────────────────────────────────────────────────┘
```
*Figure 10.5: Telemetry Architecture for Logging Agentic Execution Trajectories*

1. **Trajectory Graph Tracing:** Record the complete execution Directed Acyclic Graph (DAG) of thoughts, sub-queries, selected tools, raw tool responses, and verification decisions (e.g., using OpenTelemetry, LangSmith, or Phoenix).
2. **Tool Usage Efficiency Telemetry:** Track tool invocation success rates, empty retrieval ratios, and redundant tool calls.
3. **Loop Guardrail Alarms:** Trigger automated alerts when an agent execution exceeds step limits ($S > 4$) or encounters repeated identical sub-queries.

---

## 10.8 Concise Engineering Checklist for Agentic RAG

Use this checklist to evaluate production readiness before deploying Agentic RAG:

- [ ] **Empirical Failure Audit Completed:** Confirm via log inspection that current fixed-pipeline errors stem from inability to adapt retrieval, rather than generator parameter capacity limits (Chapter 7 showed generator scaling eliminated 87.5% of reasoning errors without agent loops).
- [ ] **Strict Hard Step Limits Enforced:** Enforce a hard cap on maximum agent steps ($S_{\text{max}} \le 4$) and total wall-clock execution timeouts to prevent infinite loops.
- [ ] **Context Pruning Active:** Implement active context summarization or state pruning between agent steps to prevent context window explosion and attention dilution.
- [ ] **Multi-Tool Interfaces Encapsulated:** Wrap tool invocations behind strict, validated schemas (Pydantic / JSON Schema) with explicit error handling for empty tool outputs.
- [ ] **Trajectory Telemetry Enabled:** Instrument full trajectory tracing to log intermediate thoughts, tool calls, and reflection outputs.
- [ ] **Deterministic Stopping Criteria Defined:** Specify clear, programmatic verifiers for stopping conditions rather than relying solely on LLM self-reflection.
- [ ] **Cost & Latency SLAs Approved:** Ensure project stakeholders approve the $3\times$--$10\times$ token cost multiplier and multi-second latency increase.

---

## 10.9 Chapter Summary

- **Fixed vs. Agentic RAG:** Fixed RAG pipelines execute a single linear pass (`Retrieve -> Rerank -> Generate`). Agentic RAG introduces dynamic control loops, enabling an LLM agent to decompose complex questions, select specialized tools, evaluate evidence, and refine queries iteratively.
- **Superficial Wrappers vs. True Adaptivity:** Wrapping a static RAG pipeline in an agent framework without empowering the agent to dynamically decompose queries, verify context completeness, and alter execution paths is not true adaptive retrieval.
- **Severe Failure Modes:** Agentic RAG introduces unique operational risks: infinite retrieval loops, retrieval drift away from core query intent, error propagation across steps, context window explosion, and substantial latency/cost penalties.
- **Empirical Perspective:** In Chapter 7, our fixed hybrid pipeline achieved 90.0% accuracy on held-out test data. The remaining 3 non-annotation errors were multi-hop reasoning failures (`MH = 3`). While these failures motivate exploring adaptive agent loops, Agentic RAG was not evaluated in our final benchmark. Given the heavy cost and latency penalties of agentic loops, engineering teams should adopt Agentic RAG only when open-ended, highly complex multi-hop queries strictly demand dynamic adaptability.
