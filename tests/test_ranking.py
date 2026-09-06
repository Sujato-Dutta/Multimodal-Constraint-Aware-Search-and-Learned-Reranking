"""
Unit tests for ranking feature extractor and LightGBM LambdaMART ranker.
"""
import pytest
import numpy as np
from src.query_understanding.constraint_parser import ConstraintParser
from src.ranking.feature_extractor import RankingFeatureExtractor
from src.ranking.lambdamart_ranker import LambdaMARTRanker

def test_feature_extraction_dimensions():
    parser = ConstraintParser()
    fe = RankingFeatureExtractor()

    constraints = parser.parse("black running shoes under $120")
    dummy_query_emb = np.random.randn(512).astype(np.float32)
    dummy_cand_emb = np.random.randn(512).astype(np.float32)

    candidate = {
        "product_id": "ADI-001",
        "name": "Ultraboost Light",
        "category": "Footwear",
        "subcategory": "Running Shoes",
        "color": "Black",
        "selling_price": 110.0,
        "gender": "Men",
        "rating": 4.8,
        "reviews_count": 150,
        "similarity_score": 0.85
    }

    features, status = fe.extract_features(
        query_constraints=constraints,
        query_text_emb=dummy_query_emb,
        query_img_emb=None,
        query_fused_emb=dummy_query_emb,
        candidate=candidate,
        cand_text_emb=dummy_cand_emb,
        cand_img_emb=dummy_cand_emb
    )

    assert len(features) == 16
    assert status["all_satisfied"] is True
    assert status["category_ok"] is True
    assert status["color_ok"] is True
    assert status["price_ok"] is True

def test_lambdamart_training_and_reranking():
    ranker = LambdaMARTRanker()
    
    # 2 query groups with 5 candidates each
    np.random.seed(42)
    X = np.random.randn(10, 16).astype(np.float32)
    y = np.array([3, 2, 1, 0, 0, 3, 2, 2, 0, 0], dtype=np.int32)
    groups = [5, 5]

    ranker.fit(X_train=X, y_train=y, group_train=groups, enable_mlflow=False)
    assert ranker.model is not None

    candidates = [
        {"product_id": f"P{i}", "similarity_score": 0.5 + i*0.05}
        for i in range(5)
    ]
    reranked = ranker.rerank(candidates, X[:5])
    assert len(reranked) == 5
    assert "rerank_score" in reranked[0]
    assert reranked[0]["final_rank"] == 1
