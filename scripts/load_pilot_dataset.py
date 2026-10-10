import sys
import os
import yaml

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import inspect_schema, load_linked_pilot_subset
from src.data.manifest import create_dataset_manifest, verify_pilot_dataset
from src.utils.logging_config import setup_logger

logger = setup_logger("load_pilot_script")

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/config.yaml"))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hf_path = cfg["dataset"]["hf_path"]
    qrels_hf_path = cfg["dataset"]["qrels_hf_path"]
    queries_count = cfg["dataset"]["pilot_size"]["queries_count"]
    seed = cfg["reproducibility"]["seed"]
    processed_dir = cfg["paths"]["processed_dir"]
    manifest_path = cfg["paths"]["manifest_path"]

    logger.info("Step 1: Inspecting dataset schemas (corpus, queries, qrels)...")
    schema_info = inspect_schema(hf_path=hf_path, qrels_hf_path=qrels_hf_path)

    logger.info(f"Step 2: Creating linked pilot subset (queries: {queries_count}, seed: {seed})...")
    pilot_stats = load_linked_pilot_subset(
        hf_path=hf_path,
        qrels_hf_path=qrels_hf_path,
        queries_count=queries_count,
        seed=seed,
        output_dir=processed_dir
    )

    logger.info("Step 3: Generating comprehensive dataset manifest...")
    manifest = create_dataset_manifest(
        hf_path=hf_path,
        qrels_hf_path=qrels_hf_path,
        license_name="CC BY-SA 4.0",
        schema_info=schema_info,
        pilot_stats=pilot_stats,
        output_path=manifest_path
    )

    logger.info("Step 4: Verifying linked pilot dataset...")
    success = verify_pilot_dataset(manifest_path=manifest_path)

    if success:
        logger.info("=== Linked Pilot Dataset Pipeline Completed & Verified Successfully! ===")
    else:
        logger.error("!!! Linked Pilot Dataset Verification Failed !!!")

if __name__ == "__main__":
    main()
