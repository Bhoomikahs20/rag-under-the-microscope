import sys

try:
    from rank_bm25 import BM25Okapi
    print("rank_bm25 is available!")
except ImportError:
    print("rank_bm25 is NOT installed.")
