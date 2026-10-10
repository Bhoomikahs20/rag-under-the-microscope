"""
RAG Book Research Project - Linked Pilot Dataset Exploration Script
"""

import os
import sys
import json
import pandas as pd

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.manifest import verify_pilot_dataset

def explore_linked_pilot():
    manifest_path = "data/manifest.json"
    print("==========================================================")
    print("    1. VERIFYING LINKED PILOT DATASET INTEGRITY")
    print("==========================================================")
    verified = verify_pilot_dataset(manifest_path)
    if not verified:
        print("Dataset verification failed!")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    corpus_path = manifest["subsets"]["corpus"]["local_filepath"]
    queries_path = manifest["subsets"]["queries"]["local_filepath"]
    qrels_path = manifest["subsets"]["qrels"]["local_filepath"]

    print("\n==========================================================")
    print("    2. LOADING LINKED DATASET FILES")
    print("==========================================================")
    df_corpus = pd.read_json(corpus_path)
    df_queries = pd.read_json(queries_path)
    df_qrels = pd.read_json(qrels_path)

    print(f"Corpus documents loaded: {len(df_corpus)} (Columns: {list(df_corpus.columns)})")
    print(f"Queries loaded:          {len(df_queries)} (Columns: {list(df_queries.columns)})")
    print(f"Relevance links loaded:  {len(df_qrels)} (Columns: {list(df_qrels.columns)})")

    print("\n==========================================================")
    print("    3. QUERY-TO-DOCUMENT LINKING ANALYSIS")
    print("==========================================================")
    rel_per_query = df_qrels.groupby('query_id')['corpus_id'].count()
    print("Relevant Documents per Query Distribution:")
    print(rel_per_query.value_counts().to_string())

    print("\nSample Query with its Ground-Truth Relevant Documents:")
    sample_qid = df_queries.iloc[0]['_id']
    sample_qtext = df_queries.iloc[0]['text']
    rel_cids = df_qrels[df_qrels['query_id'] == sample_qid]['corpus_id'].tolist()
    rel_cids_str = [str(c) for c in rel_cids]
    rel_docs = df_corpus[df_corpus['_id'].astype(str).isin(rel_cids_str)]

    print(f"\n[Query ID]: {sample_qid}")
    print(f"[Query Text]: {sample_qtext}")
    print(f"[Ground Truth Relevant Doc IDs]: {rel_cids_str}")

    for idx, doc in rel_docs.iterrows():
        print(f"\n  -> Document [{doc['_id']}] Title: '{doc['title']}'")
        print(f"     Snippet: {doc['text'][:150]}...")

    print("\n==========================================================")
    print("    4. PILOT TEXT LENGTH STATISTICS")
    print("==========================================================")
    df_corpus['word_count'] = df_corpus['text'].apply(lambda x: len(str(x).split()))
    df_queries['word_count'] = df_queries['text'].apply(lambda x: len(str(x).split()))

    print("Corpus Document Length (words):")
    print(df_corpus['word_count'].describe().to_string())

    print("\nQuery Length (words):")
    print(df_queries['word_count'].describe().to_string())

if __name__ == "__main__":
    explore_linked_pilot()
