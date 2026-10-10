import time
import random
import json
from datasets import load_dataset

print("--- Step 1: Loading Qrels & Queries ---", flush=True)
qrels = load_dataset('BeIR/hotpotqa-qrels', split='train')
queries_ds = load_dataset('BeIR/hotpotqa', 'queries', split='queries', streaming=True)

# Select 50 reproducible queries using fixed random seed
random.seed(42)
all_queries = list(iter(queries_ds.take(1000)))
sample_queries = random.sample(all_queries, 50)
selected_qids = set(q['_id'] for q in sample_queries)

print(f"Selected {len(sample_queries)} queries reproducibly with seed 42.", flush=True)

# Extract relevant qrels for selected queries
relevant_qrels = []
relevant_doc_ids = set()

for x in qrels:
    if x['query-id'] in selected_qids:
        qid = x['query-id']
        cid = str(x['corpus-id'])
        score = int(x['score'])
        relevant_qrels.append({
            "query_id": qid,
            "corpus_id": cid,
            "score": score
        })
        relevant_doc_ids.add(cid)

print(f"Found {len(relevant_qrels)} relevance pairs across {len(relevant_doc_ids)} unique corpus documents.", flush=True)
print("Sample relevance pair:", relevant_qrels[0], flush=True)

# Stream corpus documents
print("--- Step 2: Streaming Corpus to retrieve linked documents ---", flush=True)
t0 = time.time()
corpus_stream = load_dataset('BeIR/hotpotqa', 'corpus', split='corpus', streaming=True)

found_docs = {}
extra_distractor_docs = []
scanned = 0

for doc in corpus_stream:
    scanned += 1
    doc_id = str(doc['_id'])
    
    if doc_id in relevant_doc_ids and doc_id not in found_docs:
        found_docs[doc_id] = doc
    elif len(extra_distractor_docs) < 100:
        extra_distractor_docs.append(doc)
        
    if scanned % 20000 == 0:
        print(f"Scanned {scanned} docs... Found {len(found_docs)} / {len(relevant_doc_ids)} relevant docs (Elapsed: {time.time()-t0:.1f}s)", flush=True)
        
    if len(found_docs) == len(relevant_doc_ids):
        print(f"All {len(relevant_doc_ids)} relevant docs retrieved after scanning {scanned} docs in {time.time()-t0:.1f}s!", flush=True)
        break

print(f"Retrieved {len(found_docs)} relevant corpus documents.", flush=True)
