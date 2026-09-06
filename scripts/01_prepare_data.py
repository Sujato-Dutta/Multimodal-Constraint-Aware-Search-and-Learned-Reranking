"""
Step 1: Dataset Ingestion & Query Benchmark Generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from src.config import config
from src.data.dataset_loader import DatasetLoader
from src.data.query_generator import QueryGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 1: Preparing Catalog Data & Query Suites ===")
    loader = DatasetLoader()
    df = loader.load_and_preprocess(force_recompute=False)
    logger.info(f"Loaded catalog with {len(df)} products.")

    gen = QueryGenerator(df)
    splits = gen.generate_query_suite(num_train=300, num_val=60, num_test=60)
    logger.info(f"Successfully generated queries: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    logger.info("Step 1 complete!")

if __name__ == "__main__":
    main()
