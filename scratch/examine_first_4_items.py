import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with open("scratch/rag_failures_detail.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus = {d["_id"]: d for d in json.load(f)}

    target_indices = [3, 7, 13, 14]

    for item in failures:
        if item["query_index"] in target_indices:
            print("="*80)
            print(f"QUERY INDEX: {item['query_index']} | QID: {item['query_id']}")
            print(f"Question: {item['question']}")
            print(f"Gold Answer: '{item['gold_answer']}'")
            print(f"RAG Generated Answer: '{item['rag_answer']}'")
            print(f"Retrieved Titles: {[c['title'] for c in item['retrieved_chunks']]}")
            print("\nRetrieved Chunks Full Snippets:")
            for c in item['retrieved_chunks']:
                print(f"  Title: [{c['title']}] (Score: {c['similarity_score']:.4f})")
                print(f"  Snippet: {c['snippet']}")
                
            print("\nFull Text of Gold Docs:")
            for doc_id in item['gold_relevant_docs']:
                print(f"  Doc Title: [{corpus[doc_id]['title']}]")
                print(f"  Text: {corpus[doc_id]['text']}")
            print("\n")

if __name__ == "__main__":
    main()
