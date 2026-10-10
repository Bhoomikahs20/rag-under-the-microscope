import json
import sys

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def main():
    with open("data/processed/rag_baseline/rag_results.json", "r", encoding="utf-8") as f:
        rag_data = json.load(f)
    
    with open("data/processed/hotpotqa_pilot_qrels.json", "r", encoding="utf-8") as f:
        qrels_data = json.load(f)
        
    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # Build qrels mapping query_id -> set of doc_ids
    qrels_map = {}
    for q in qrels_data:
        qid = q["query_id"]
        if qid not in qrels_map:
            qrels_map[qid] = set()
        qrels_map[qid].add(q["corpus_id"])
        
    corpus_map = {doc["_id"]: doc for doc in corpus_data}

    incorrect_results = []
    for res in rag_data["per_question_results"]:
        if not res["correctness_assessment"]["is_correct"]:
            incorrect_results.append(res)

    print(f"Total incorrect results: {len(incorrect_results)}\n")
    
    analysis_list = []
    for i, res in enumerate(incorrect_results, 1):
        qid = res["query_id"]
        gold_relevant_docs = qrels_map.get(qid, set())
        retrieved_doc_ids = set([c["doc_id"] for c in res["retrieved_chunks"]])
        
        rel_retrieved = gold_relevant_docs.intersection(retrieved_doc_ids)
        rel_missing = gold_relevant_docs - retrieved_doc_ids
        
        item_info = {
            "fail_index": i,
            "query_index": res["query_index"],
            "query_id": qid,
            "question": res["question"],
            "gold_answer": res["gold_answer"],
            "rag_answer": res["rag_generated_answer"],
            "gold_relevant_docs": list(gold_relevant_docs),
            "retrieved_doc_ids": list(retrieved_doc_ids),
            "rel_retrieved_count": f"{len(rel_retrieved)}/{len(gold_relevant_docs)}",
            "rel_retrieved_titles": [corpus_map[d]["title"] for d in rel_retrieved if d in corpus_map],
            "rel_missing_titles": [corpus_map[d]["title"] for d in rel_missing if d in corpus_map],
            "retrieved_chunks": res["retrieved_chunks"]
        }
        analysis_list.append(item_info)
        
        print("="*80)
        print(f"FAILED ITEM {i}/{len(incorrect_results)} | Query Index: {res['query_index']}")
        print(f"Query ID: {qid}")
        print(f"Question: {res['question']}")
        print(f"Gold Answer: {res['gold_answer']}")
        print(f"RAG Answer: {res['rag_generated_answer']}")
        print(f"Relevant Docs Retrieved: {len(rel_retrieved)}/{len(gold_relevant_docs)}")
        print(f"Retrieved Titles: {[c['title'] for c in res['retrieved_chunks']]}")
        if rel_missing:
            print(f"Missing Doc Titles: {[corpus_map[d]['title'] for d in rel_missing if d in corpus_map]}")
            
        print("\nRetrieved Chunks:")
        for chunk in res["retrieved_chunks"]:
            is_rel = chunk["doc_id"] in gold_relevant_docs
            print(f"  Rank {chunk['rank']} | Score: {chunk['similarity_score']:.4f} | Title: {chunk['title']} [Relevant: {is_rel}]")
            
        print("\nFull Text of Relevant Gold Docs:")
        for doc_id in gold_relevant_docs:
            doc = corpus_map.get(doc_id, {})
            print(f"  --- [{doc.get('title')}] (Doc ID: {doc_id}) ---")
            print(f"  {doc.get('text')}")

    with open("scratch/rag_failures_detail.json", "w", encoding="utf-8") as out_f:
        json.dump(analysis_list, out_f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
