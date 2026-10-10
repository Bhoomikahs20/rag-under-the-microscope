import json

def print_failures():
    with open("scratch/rag_failures_detail.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    for item in failures:
        print(f"Index: {item['query_index']} | ID: {item['query_id']}")
        print(f"Q: {item['question']}")
        print(f"Gold: {item['gold_answer']}")
        print(f"RAG: {item['rag_answer']}")
        print(f"Retrieved {item['rel_retrieved_count']} gold docs")
        print(f"Retrieved Titles: {[c['title'] for c in item['retrieved_chunks']]}")
        if item['rel_missing_titles']:
            print(f"Missing Gold Titles: {item['rel_missing_titles']}")
        print("-" * 60)

if __name__ == "__main__":
    print_failures()
