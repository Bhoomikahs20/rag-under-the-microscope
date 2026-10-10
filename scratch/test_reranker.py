from sentence_transformers import CrossEncoder
import time

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Testing loading CrossEncoder '{model_name}'...")
t0 = time.time()
try:
    reranker = CrossEncoder(model_name)
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "London is the capital of the UK.")
    ]
    scores = reranker.predict(pairs)
    t_el = round(time.time() - t0, 2)
    print(f"Successfully loaded '{model_name}'! Scores: {scores}, time: {t_el}s")
except Exception as e:
    print(f"Failed to load '{model_name}': {e}")
