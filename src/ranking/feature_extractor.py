"""
Feature Extractor for Multimodal Constraint-Aware Reranking.
Transforms (Query, Candidate) pairs into rich dense feature vectors
combining semantic similarities, structured constraint alignments, and product priors.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math

from src.query_understanding.constraint_parser import ParsedConstraints

FEATURE_NAMES = [
    "clip_text_similarity",
    "clip_image_similarity",
    "clip_multimodal_similarity",
    "category_match",
    "color_match",
    "price_compatibility",
    "gender_match",
    "waterproof_match",
    "attribute_overlap_ratio",
    "constraints_total_count",
    "constraints_satisfied_count",
    "constraints_satisfaction_ratio",
    "hard_violation_flag",
    "relative_price",
    "product_rating",
    "popularity_score"
]

class RankingFeatureExtractor:
    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def extract_features(
        self,
        query_constraints: ParsedConstraints,
        query_text_emb: Optional[np.ndarray],
        query_img_emb: Optional[np.ndarray],
        query_fused_emb: np.ndarray,
        candidate: Dict[str, Any],
        cand_text_emb: Optional[np.ndarray] = None,
        cand_img_emb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts 16 dense ranking signals and metadata for a single candidate product.
        """
        # 1. Semantic similarities
        retrieval_sim = float(candidate.get("similarity_score", 0.5))
        
        # Text similarity
        if query_text_emb is not None and cand_text_emb is not None:
            text_sim = float(np.dot(query_text_emb.flatten(), cand_text_emb.flatten()))
        else:
            text_sim = retrieval_sim

        # Image similarity
        if query_img_emb is not None and cand_img_emb is not None:
            img_sim = float(np.dot(query_img_emb.flatten(), cand_img_emb.flatten()))
        elif query_img_emb is not None and cand_text_emb is not None:
            img_sim = float(np.dot(query_img_emb.flatten(), cand_text_emb.flatten()))
        else:
            img_sim = retrieval_sim

        # Multimodal similarity
        if cand_text_emb is not None:
            mm_sim = float(np.dot(query_fused_emb.flatten(), cand_text_emb.flatten()))
        else:
            mm_sim = retrieval_sim

        # 2. Structured Constraint Matching
        cand_cat = str(candidate.get("category", ""))
        cand_subcat = str(candidate.get("subcategory", ""))
        cand_color = str(candidate.get("color", ""))
        cand_price = float(candidate.get("selling_price", 0.0))
        cand_gender = str(candidate.get("gender", "Unisex"))
        cand_wp = bool(candidate.get("is_waterproof", False))
        cand_desc = f"{candidate.get('name', '')} {cand_color} {candidate.get('description', '')}".lower()

        # Category match
        category_match = 1.0
        if query_constraints.category:
            if cand_cat.lower() == query_constraints.category.lower():
                if query_constraints.subcategory and cand_subcat.lower() == query_constraints.subcategory.lower():
                    category_match = 1.0
                elif query_constraints.subcategory:
                    category_match = 0.75
                else:
                    category_match = 1.0
            else:
                category_match = 0.0

        # Color match
        color_match = 1.0
        if query_constraints.color:
            if cand_color.lower() == query_constraints.color.lower():
                color_match = 1.0
            elif query_constraints.color.lower() in cand_color.lower() or query_constraints.color.lower() in cand_desc:
                color_match = 0.5
            else:
                color_match = 0.0

        # Price compatibility
        price_compat = 1.0
        max_p = query_constraints.max_price
        min_p = query_constraints.min_price
        
        if max_p and cand_price > max_p:
            overage = (cand_price - max_p) / max_p
            price_compat = max(0.0, 1.0 - (overage * 1.5))
        elif min_p and cand_price < min_p:
            underage = (min_p - cand_price) / min_p
            price_compat = max(0.0, 1.0 - (underage * 1.5))

        # Gender match
        gender_match = 1.0
        if query_constraints.gender:
            if cand_gender.lower() == query_constraints.gender.lower() or cand_gender.lower() == "unisex":
                gender_match = 1.0
            else:
                gender_match = 0.0

        # Waterproof match
        wp_match = 1.0
        if query_constraints.is_waterproof is not None:
            if query_constraints.is_waterproof:
                wp_match = 1.0 if cand_wp else 0.1
            else:
                wp_match = 1.0

        # Style keyword overlap
        style_kws = query_constraints.style_keywords
        if style_kws:
            matched_kws = sum(1 for kw in style_kws if kw in cand_desc)
            attr_overlap = matched_kws / len(style_kws)
        else:
            attr_overlap = 1.0

        # Constraint counting
        total_active = 0
        satisfied_count = 0
        hard_violated = False

        if query_constraints.category:
            total_active += 1
            if category_match >= 0.75:
                satisfied_count += 1
            else:
                hard_violated = True

        if query_constraints.color:
            total_active += 1
            if color_match >= 0.5:
                satisfied_count += 1
            else:
                hard_violated = True

        if max_p or min_p:
            total_active += 1
            if price_compat >= 0.8:
                satisfied_count += 1
            elif price_compat < 0.5:
                hard_violated = True

        if query_constraints.gender:
            total_active += 1
            if gender_match >= 1.0:
                satisfied_count += 1
            else:
                hard_violated = True

        if query_constraints.is_waterproof is not None and query_constraints.is_waterproof:
            total_active += 1
            if wp_match >= 0.9:
                satisfied_count += 1
            else:
                hard_violated = True

        if total_active == 0:
            total_active = 1
            satisfied_count = 1

        sat_ratio = satisfied_count / total_active
        hard_flag = 1.0 if hard_violated else 0.0

        # 3. Product priors
        rel_price = min(cand_price / 250.0, 1.5)
        rating_score = float(candidate.get("rating", 4.5)) / 5.0
        rev_count = float(candidate.get("reviews_count", 100))
        pop_score = math.log1p(rev_count) / 10.0

        features = np.array([
            text_sim,
            img_sim,
            mm_sim,
            category_match,
            color_match,
            price_compat,
            gender_match,
            wp_match,
            attr_overlap,
            float(total_active),
            float(satisfied_count),
            sat_ratio,
            hard_flag,
            rel_price,
            rating_score,
            pop_score
        ], dtype=np.float32)

        constraint_status = {
            "all_satisfied": bool(satisfied_count == total_active and not hard_violated),
            "category_ok": bool(category_match >= 0.75),
            "color_ok": bool(color_match >= 0.5),
            "price_ok": bool(price_compat >= 0.8),
            "gender_ok": bool(gender_match >= 1.0),
            "waterproof_ok": bool(wp_match >= 0.9),
            "satisfaction_ratio": float(sat_ratio),
            "hard_violated": bool(hard_violated)
        }

        return features, constraint_status

    def batch_extract(
        self,
        query_constraints: ParsedConstraints,
        query_text_emb: Optional[np.ndarray],
        query_img_emb: Optional[np.ndarray],
        query_fused_emb: np.ndarray,
        candidates: List[Dict[str, Any]],
        catalog_text_embeddings: Optional[Dict[str, np.ndarray]] = None,
        catalog_img_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extracts features for a batch of candidates for a single query."""
        feature_matrix = []
        statuses = []
        
        for cand in candidates:
            pid = str(cand["product_id"])
            c_text_emb = catalog_text_embeddings.get(pid) if catalog_text_embeddings else None
            c_img_emb = catalog_img_embeddings.get(pid) if catalog_img_embeddings else None
            
            feat_vec, status = self.extract_features(
                query_constraints=query_constraints,
                query_text_emb=query_text_emb,
                query_img_emb=query_img_emb,
                query_fused_emb=query_fused_emb,
                candidate=cand,
                cand_text_emb=c_text_emb,
                cand_img_emb=c_img_emb
            )
            feature_matrix.append(feat_vec)
            statuses.append(status)
            
        return np.vstack(feature_matrix).astype(np.float32), statuses
