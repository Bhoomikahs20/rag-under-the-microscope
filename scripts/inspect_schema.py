import sys
import os
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import inspect_schema
from src.utils.logging_config import setup_logger

logger = setup_logger("inspect_schema_script")

def main():
    logger.info("Starting schema inspection for Hugging Face dataset BeIR/hotpotqa...")
    schema_info = inspect_schema(hf_path="BeIR/hotpotqa", subsets=["corpus", "queries"])
    
    print("\n" + "="*60)
    print("      BeIR/hotpotqa SCHEMA INSPECTION RESULT")
    print("="*60)
    print(json.dumps(schema_info, indent=2))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
