import json

def generate_report_data():
    with open("data/processed/rag_baseline/rag_results.json", "r", encoding="utf-8") as f:
        rag_data = json.load(f)
        
    with open("data/processed/hotpotqa_pilot_qrels.json", "r", encoding="utf-8") as f:
        qrels_data = json.load(f)
        
    with open("data/processed/hotpotqa_pilot_corpus.json", "r", encoding="utf-8") as f:
        corpus = {d["_id"]: d for d in json.load(f)}

    qrels_map = {}
    for q in qrels_data:
        qid = q["query_id"]
        if qid not in qrels_map:
            qrels_map[qid] = set()
        qrels_map[qid].add(q["corpus_id"])

    # Categorization mapping
    category_map = {
        3: ("Multi-Hop Reasoning Failure", "Both gold documents retrieved. FLAN-T5 misassociated voice actor Frank Welker with 'Cooler' instead of 'Howler'."),
        5: ("Retrieval Failure", "Missing gold document 'Guadalajara' (doc 2). Model answered 'Guadalajara' (city) instead of state 'Jalisco'."),
        7: ("Multi-Hop Reasoning Failure", "Both gold docs retrieved. Distractor doc 'Only Fools and Horses' in top-3 caused LLM to output 'John Sullivan' instead of 'Jennifer Saunders'."),
        13: ("Gold-Answer / Data-Quality Issue", "Gold label is 'British' (nationality) for a city question, and neither retrieved document contains the city name (Washington D.C.)."),
        14: ("Multi-Hop Reasoning Failure", "Both gold docs retrieved. Model failed release year comparison (1981 vs 2006) and picked the wrong movie."),
        17: ("Generation Failure", "Both gold docs retrieved. FLAN-T5 repeated the prompt framing string and hit max token limit before generating answer."),
        21: ("Retrieval Failure", "Missing gold document '10 Things I Hate About You' (doc 2). Model guessed '2016'."),
        23: ("Generation Failure", "Both gold docs retrieved. Model generated full lead-in sentence and full name, exceeding exact answer boundary."),
        31: ("Multi-Hop Reasoning Failure", "Both gold docs retrieved. Model failed geographical comparison (Northern Ireland vs USA) and answered 'yes' instead of 'no'."),
        34: ("Retrieval Failure", "Missing gold document 'Tom Dolby' (doc 2). Model guessed director 'Luca Guadagnino' from doc 1."),
        35: ("Retrieval Failure", "Missing gold document 'Jennifer Aniston' (doc 2). Model picked 'Brad Pitt' from doc 1."),
        36: ("Retrieval Failure", "Missing gold document 'The Good Fight' (doc 2). Model outputted 'The Good Fight' (show name) instead of prequel 'The Good Wife'."),
        40: ("Multi-Hop Reasoning Failure", "Both gold docs retrieved. Model failed boolean conjunction (Toronto in Canada, ITU in Turkey) and answered 'yes' instead of 'no'."),
        43: ("Retrieval Failure", "Missing gold document 'Knife Edge (film)' (doc 2). Model hallucinated 'Coky Giedroyc'.")
    }

    per_error_records = []
    for res in rag_data["per_question_results"]:
        if not res["correctness_assessment"]["is_correct"]:
            idx = res["query_index"]
            qid = res["query_id"]
            gold_docs = qrels_map.get(qid, set())
            ret_doc_ids = set([c["doc_id"] for c in res["retrieved_chunks"]])
            rel_retrieved = gold_docs.intersection(ret_doc_ids)
            all_gold_retrieved = (len(rel_retrieved) == len(gold_docs))
            
            cat, cause = category_map[idx]
            
            rec = {
                "query_index": idx,
                "query_id": qid,
                "question": res["question"],
                "gold_answer": res["gold_answer"],
                "rag_answer": res["rag_generated_answer"],
                "category": cat,
                "root_cause_explanation": cause,
                "all_gold_docs_retrieved": all_gold_retrieved,
                "gold_docs_retrieved_ratio": f"{len(rel_retrieved)}/{len(gold_docs)}",
                "retrieved_chunks": [
                    {
                        "rank": c["rank"],
                        "doc_id": c["doc_id"],
                        "title": c["title"],
                        "score": round(c["similarity_score"], 4),
                        "is_relevant_gold": c["doc_id"] in gold_docs,
                        "snippet": c["snippet"]
                    }
                    for c in res["retrieved_chunks"]
                ]
            }
            per_error_records.append(rec)

    print(f"Verified {len(per_error_records)} per-error records.")
    
    # Category summary breakdown
    cat_counts = {}
    for r in per_error_records:
        c = r["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        
    print("Category Breakdown:")
    for c, cnt in cat_counts.items():
        print(f"  - {c}: {cnt} ({cnt/14*100:.1f}%)")

if __name__ == "__main__":
    generate_report_data()
