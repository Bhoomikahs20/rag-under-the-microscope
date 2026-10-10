import json

def categorize():
    with open("scratch/rag_failures_detail.json", "r", encoding="utf-8") as f:
        failures = json.load(f)

    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus = {d["_id"]: d for d in json.load(f)}

    print(f"{'Idx':<4} | {'QID':<25} | {'Rel Docs':<8} | {'Retrieved Titles'}")
    print("="*100)
    for item in failures:
        ret_titles = [c['title'] for c in item['retrieved_chunks']]
        print(f"{item['query_index']:<4} | {item['query_id']:<25} | {item['rel_retrieved_count']:<8} | {ret_titles}")

if __name__ == "__main__":
    categorize()
