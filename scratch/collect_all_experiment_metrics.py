import os
import json

base_dir = "data/processed/rag_baseline"

def dump_file(label, rel_path):
    path = os.path.join(base_dir, rel_path)
    print(f"=== {label} ({path}) ===")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "summary_metrics" in d:
        print(json.dumps(d["summary_metrics"], indent=2))
    elif "metrics_summary" in d:
        print(json.dumps(d["metrics_summary"], indent=2))
    elif "k_summary" in d:
        print(json.dumps(d["k_summary"], indent=2))
    else:
        print(json.dumps({k: d[k] for k in d if k != "per_question_results" and k != "results_by_config" and k != "results_by_model" and k != "results_by_mode"}, indent=2))

dump_file("Chunking", "chunking_experiment/chunking_results.json")
dump_file("Embedding", "embedding_experiment/embedding_results.json")
dump_file("Hybrid Search", "hybrid_search_experiment/hybrid_search_results.json")
