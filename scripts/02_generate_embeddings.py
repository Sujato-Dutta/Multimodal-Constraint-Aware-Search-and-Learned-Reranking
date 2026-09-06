"""
Step 2: Precompute & Cache CLIP Multimodal Product Embeddings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd
from src.config import config
from src.data.dataset_loader import DatasetLoader
from src.embeddings.clip_encoder import CLIPEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 2: Precomputing CLIP Multimodal Embeddings ===")
    loader = DatasetLoader()
    df = loader.load_and_preprocess()

    encoder = CLIPEncoder()
    text_embs, img_embs = encoder.compute_and_cache_catalog_embeddings(df, force_recompute=False)
    logger.info(f"Embeddings ready. Text: {text_embs.shape}, Image: {img_embs.shape}")
    logger.info("Step 2 complete!")

if __name__ == "__main__":
    main()
