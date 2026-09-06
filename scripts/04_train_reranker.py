"""
Step 4: Train LightGBM LambdaMART Reranker with Hard-Negative Mining.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import config, PROJECT_ROOT
from src.data.dataset_loader import DatasetLoader
from src.embeddings.clip_encoder import CLIPEncoder
from src.query_understanding.constraint_parser import ConstraintParser
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.feature_extractor import RankingFeatureExtractor
from src.ranking.lambdamart_ranker import LambdaMARTRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_ranking_dataset(
    queries: list,
    encoder: CLIPEncoder,
    retriever: CandidateRetriever,
    parser: ConstraintParser,
    feature_extractor: RankingFeatureExtractor,
    catalog_text_embs: dict,
    catalog_img_embs: dict,
    candidate_k: int = 50
):
    X_list = []
    y_list = []
    group_sizes = []

    for q in tqdm(queries, desc="Building ranking dataset"):
        q_text = q["text"]
        q_modality = q.get("modality", "multimodal")
        gt = q.get("ground_truth_relevance", {})

        # Parse constraints
        parsed_c = parser.parse(q_text)

        # Encode query
        if q_modality == "text_only":
            q_text_emb = encoder.encode_text([q_text])[0]
            q_img_emb = None
            q_fused = q_text_emb.reshape(1, -1)
        elif q_modality == "image_only":
            ref_img = q.get("reference_image_url")
            q_img_emb = encoder.encode_image([ref_img])[0]
            q_text_emb = None
            q_fused = q_img_emb.reshape(1, -1)
        else:
            q_text_emb = encoder.encode_text([q_text])[0]
            ref_img = q.get("reference_image_url")
            q_img_emb = encoder.encode_image([ref_img])[0]
            q_fused = encoder.encode_multimodal(text=q_text, image=ref_img)

        # Retrieve candidates from Pinecone index
        candidates = retriever.retrieve_candidates(q_fused, top_k=candidate_k)
        if not candidates:
            continue

        # Extract features for all candidates
        feats, _ = feature_extractor.batch_extract(
            query_constraints=parsed_c,
            query_text_emb=q_text_emb,
            query_img_emb=q_img_emb,
            query_fused_emb=q_fused,
            candidates=candidates,
            catalog_text_embeddings=catalog_text_embs,
            catalog_img_embeddings=catalog_img_embs
        )

        labels = [int(gt.get(c["product_id"], 0)) for c in candidates]

        X_list.append(feats)
        y_list.extend(labels)
        group_sizes.append(len(candidates))

    if not X_list:
        return np.zeros((0, 16)), np.zeros(0), []

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    return X, y, group_sizes

def main():
    logger.info("=== STEP 4: Training LambdaMART Reranker with Hard Negatives ===")
    
    loader = DatasetLoader()
    df = loader.load_and_preprocess()

    queries_dir = config.dataset.queries_dir
    train_q_file = queries_dir / "train_queries.json"
    val_q_file = queries_dir / "val_queries.json"

    with open(train_q_file, "r", encoding="utf-8") as f:
        train_queries = json.load(f)
    with open(val_q_file, "r", encoding="utf-8") as f:
        val_queries = json.load(f)

    logger.info(f"Loaded {len(train_queries)} train queries and {len(val_queries)} val queries.")

    # Load catalog embeddings
    emb_dir = config.dataset.embeddings_dir
    text_embs = np.load(emb_dir / "product_text_embeddings.npy")
    img_embs = np.load(emb_dir / "product_image_embeddings.npy")
    with open(emb_dir / "product_ids.json", "r", encoding="utf-8") as f:
        pids = json.load(f)

    catalog_text_embs = {pid: text_embs[i] for i, pid in enumerate(pids)}
    catalog_img_embs = {pid: img_embs[i] for i, pid in enumerate(pids)}

    encoder = CLIPEncoder()
    retriever = CandidateRetriever()
    parser = ConstraintParser()
    fe = RankingFeatureExtractor()

    logger.info("Constructing training features and mining hard negatives...")
    X_train, y_train, group_train = build_ranking_dataset(
        train_queries, encoder, retriever, parser, fe, catalog_text_embs, catalog_img_embs
    )
    logger.info(f"Train dataset shape: X={X_train.shape}, y={y_train.shape}, groups={len(group_train)}")

    logger.info("Constructing validation features...")
    X_val, y_val, group_val = build_ranking_dataset(
        val_queries, encoder, retriever, parser, fe, catalog_text_embs, catalog_img_embs
    )
    logger.info(f"Val dataset shape: X={X_val.shape}, y={y_val.shape}, groups={len(group_val)}")

    ranker = LambdaMARTRanker()
    ranker.fit(
        X_train=X_train,
        y_train=y_train,
        group_train=group_train,
        X_val=X_val,
        y_val=y_val,
        group_val=group_val,
        enable_mlflow=True
    )
    logger.info("Step 4 complete! LambdaMART model trained and persisted.")

if __name__ == "__main__":
    main()
