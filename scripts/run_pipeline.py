"""
End-to-End Pipeline Automation Script.
Executes all 5 pipeline stages sequentially:
1. Data Preparation & Query Suite Generation
2. Multimodal CLIP Embedding Generation & Caching
3. Pinecone Vector Database Indexing
4. LightGBM LambdaMART Training with Hard Negatives
5. Rigorous Evaluation & Statistical Significance Reporting
"""
import sys
import time
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_all():
    t_start = time.time()
    logger.info("================================================================")
    logger.info("  STARTING END-TO-END MULTIMODAL SEARCH & RERANKING PIPELINE   ")
    logger.info("================================================================")

    # Step 1
    s1 = importlib.import_module("scripts.01_prepare_data")
    s1.main()

    # Step 2
    s2 = importlib.import_module("scripts.02_generate_embeddings")
    s2.main()

    # Step 3
    s3 = importlib.import_module("scripts.03_build_index")
    s3.main()

    # Step 4
    s4 = importlib.import_module("scripts.04_train_reranker")
    s4.main()

    # Step 5
    s5 = importlib.import_module("scripts.05_evaluate")
    s5.main()

    total_time = time.time() - t_start
    logger.info("================================================================")
    logger.info(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_time:.2f}s! ")
    logger.info("================================================================")

if __name__ == "__main__":
    run_all()
