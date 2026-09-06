"""
Step 3: Build & Index Catalog into Pinecone Vector Database.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd
from src.config import config
from src.data.dataset_loader import DatasetLoader
from src.retrieval.pinecone_client import PineconeManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 3: Building Pinecone Vector Index ===")
    loader = DatasetLoader()
    df = loader.load_and_preprocess()

    emb_file = config.dataset.embeddings_dir / "product_text_embeddings.npy"
    if not emb_file.exists():
        logger.error(f"Embeddings not found at {emb_file}. Please run 02_generate_embeddings.py first.")
        return

    text_embs = np.load(emb_file)
    pm = PineconeManager()
    pm.index_catalog(df, text_embs)
    logger.info("Step 3 complete!")

if __name__ == "__main__":
    main()
