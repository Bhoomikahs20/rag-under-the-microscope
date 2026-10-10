from sentence_transformers import SentenceTransformer
import time

models_to_test = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5"
]

for m_id in models_to_test:
    print(f"Testing loading '{m_id}'...")
    t0 = time.time()
    try:
        model = SentenceTransformer(m_id)
        emb = model.encode(["This is a test document."], normalize_embeddings=True)
        t_el = round(time.time() - t0, 2)
        print(f"Successfully loaded '{m_id}'! Embedding shape: {emb.shape}, load+encode time: {t_el}s")
    except Exception as e:
        print(f"Failed to load '{m_id}': {e}")
