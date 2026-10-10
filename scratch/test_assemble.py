import os

# Define paths
ROOT_DIR = r"d:\Rag book work"
PROCESSED_DIR = r"d:\Rag book work\data\processed\rag_baseline"

MANUSCRIPT_PATH_PROCESSED = os.path.join(PROCESSED_DIR, "rag_book_final_manuscript.md")
MANUSCRIPT_PATH_ROOT = os.path.join(ROOT_DIR, "rag_book_final_manuscript.md")
QA_CHECKLIST_PATH = os.path.join(PROCESSED_DIR, "rag_book_publication_qa.md")

print("Assembling RAG Book Manuscript Package...")
