import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with open("scratch/rag_failures_detail.json", "r", encoding="utf-8") as f:
        failures = json.load(f)
        
    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus = {d["_id"]: d for d in json.load(f)}

    for item in failures:
        print("================================================================================")
        print(f"FAILED ITEM (Query Index: {item['query_index']}, Query ID: {item['query_id']})")
        print(f"Question: {item['question']}")
        print(f"Gold Answer: '{item['gold_answer']}'")
        print(f"RAG Generated Answer: '{item['rag_answer']}'")
        print(f"Gold Relevant Doc Count: {item['rel_retrieved_count']}")
        print(f"Retrieved Titles: {[c['title'] for c in item['retrieved_chunks']]}")
        print(f"Missing Gold Doc Titles: {item['rel_missing_titles']}")
        
        print("\n--- RETRIEVED CHUNKS & SCORES ---")
        context_str = ""
        for c in item['retrieved_chunks']:
            print(f"Rank {c['rank']} (Score {c['similarity_score']:.4f}) [{c['title']} - DocID: {c['doc_id']}]:")
            print(f"  {c['snippet']}")
            context_str += f"Document Title: {c['title']}\nContent: {c['snippet']}\n\n"
            
        if item['rel_missing_titles']:
            print("\n--- MISSING GOLD DOCS ---")
            for doc_id in item['gold_relevant_docs']:
                if corpus[doc_id]['title'] in item['rel_missing_titles']:
                    print(f"[{corpus[doc_id]['title']} - DocID: {doc_id}]:")
                    print(f"  {corpus[doc_id]['text']}")
        print("\n")

if __name__ == "__main__":
    main()
