"""
LightGBM LambdaMART Learning-to-Rank Model for Multimodal Search Reranking.
Implements hard-negative mining, group-based ranking optimization,
and model persistence with MLflow integration.
"""
from pathlib import Path
import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import lightgbm as lgb

from src.config import config, PROJECT_ROOT
from src.query_understanding.constraint_parser import ConstraintParser, ParsedConstraints
from src.ranking.feature_extractor import RankingFeatureExtractor, FEATURE_NAMES

logger = logging.getLogger(__name__)

class LambdaMARTRanker:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.ranking.model_output_path
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_extractor = RankingFeatureExtractor()
        self.feature_names = FEATURE_NAMES
        
        if self.model_path.exists():
            self.load(self.model_path)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: List[int],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group_val: Optional[List[int]] = None,
        enable_mlflow: bool = True
    ) -> lgb.LGBMRanker:
        """Trains LightGBM LambdaMART ranker with group-based ranking loss."""
        logger.info(f"Training LightGBM LambdaMART on {len(X_train)} instances across {len(group_train)} query groups...")
        
        self.model = lgb.LGBMRanker(
            objective=config.ranking.objective,
            metric=config.ranking.metric,
            eval_at=config.ranking.ndcg_eval_at,
            learning_rate=config.ranking.learning_rate,
            num_leaves=config.ranking.num_leaves,
            n_estimators=config.ranking.n_estimators,
            min_child_samples=config.ranking.min_data_in_leaf,
            random_state=42,
            importance_type="gain",
            verbose=-1
        )
        
        eval_set = []
        eval_group = []
        if X_val is not None and y_val is not None and group_val is not None:
            eval_set.append((X_val, y_val))
            eval_group.append(group_val)

        # MLflow tracking
        mlflow_run = None
        if enable_mlflow:
            try:
                import mlflow
                mlflow.set_experiment("multimodal_constraint_reranking")
                mlflow_run = mlflow.start_run(run_name="lambdamart_hard_negative_training")
                mlflow.log_params({
                    "learning_rate": config.ranking.learning_rate,
                    "num_leaves": config.ranking.num_leaves,
                    "n_estimators": config.ranking.n_estimators,
                    "train_instances": len(X_train),
                    "train_queries": len(group_train)
                })
            except Exception as e:
                logger.warning(f"MLflow tracking initialization skipped: {e}")

        # Train model
        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                group=group_train,
                eval_set=eval_set,
                eval_group=eval_group,
                callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train, group=group_train)

        # Log feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.info("Top Feature Importances (Gain):")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:8]:
            logger.info(f"  {feat:30s}: {imp:.4f}")

        if enable_mlflow and mlflow_run:
            try:
                import mlflow
                for feat, imp in importances.items():
                    mlflow.log_metric(f"feat_imp_{feat}", float(imp))
                mlflow.end_run()
            except Exception:
                pass

        self.save(self.model_path)
        return self.model

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        features: np.ndarray,
        constraint_statuses: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Scores and re-orders candidates using the trained LambdaMART model."""
        if not candidates:
            return []

        if self.model is None:
            # Fallback if un-trained: rank by semantic similarity
            logger.warning("LambdaMART model not loaded; falling back to similarity score.")
            scores = [float(c.get("similarity_score", 0.0)) for c in candidates]
        else:
            scores = self.model.predict(features).tolist()

        reranked = []
        for i, cand in enumerate(candidates):
            c_copy = dict(cand)
            c_copy["rerank_score"] = float(scores[i])
            if constraint_statuses and i < len(constraint_statuses):
                c_copy["constraint_status"] = constraint_statuses[i]
            reranked.append(c_copy)

        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Update rank
        for new_rank, item in enumerate(reranked, start=1):
            item["final_rank"] = new_rank

        return reranked

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved LambdaMART model to {filepath}")

    def load(self, filepath: Path):
        if filepath.exists():
            self.model = joblib.load(filepath)
            logger.info(f"Loaded LambdaMART model from {filepath}")
        else:
            logger.warning(f"Model path {filepath} does not exist.")
