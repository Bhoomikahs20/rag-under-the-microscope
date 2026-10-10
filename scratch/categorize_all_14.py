import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with open("scratch/rag_failures_detail.json", "r", encoding="utf-8") as f:
        failures = json.load(f)
        
    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus = {d["_id"]: d for d in json.load(f)}

    print(f"Total Failures: {len(failures)}\n")

    for item in failures:
        qid = item["query_id"]
        rel_retrieved_count = item["rel_retrieved_count"]
        retrieved_titles = [c["title"] for c in item["retrieved_chunks"]]
        missing_titles = item["rel_missing_titles"]
        
        print("="*80)
        print(f"ITEM Query Index: {item['query_index']} (QID: {qid})")
        print(f"Question: {item['question']}")
        print(f"Gold Answer: '{item['gold_answer']}'")
        print(f"RAG Generated Answer: '{item['rag_answer']}'")
        print(f"Relevant Docs Retrieved: {rel_retrieved_count}")
        print(f"Retrieved Titles: {retrieved_titles}")
        print(f"Missing Gold Titles: {missing_titles}")
        
        print("\nRetrieved Context Snippets:")
        for c in item["retrieved_chunks"]:
            print(f"  [{c['title']}] (Rank {c['rank']}, Score {c['similarity_score']:.4f}): {c['snippet']}")
            
        if missing_titles:
            print("\nMissing Gold Document Content:")
            for doc_id in item["gold_relevant_docs"]:
                if corpus[doc_id]["title"] in missing_titles:
                    print(f"  [{corpus[doc_id]['title']}]: {corpus[doc_id]['text']}")
        print("\n")

if __name__ == "__main__":
    main()
