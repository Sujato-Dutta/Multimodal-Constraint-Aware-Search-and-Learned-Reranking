"""
Standard IR and Constraint Evaluation Metrics:
NDCG@K, Recall@K, MRR, Constraint Satisfaction Rate, and Violation Rates.
"""
import math
from typing import List, Dict, Any, Set, Optional
import numpy as np

def dcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Calculates Discounted Cumulative Gain at K."""
    relevance_scores = list(relevance_scores)[:k]
    if not relevance_scores:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved_relevances: List[float], all_ground_truth_relevances: List[float], k: int = 10) -> float:
    """Calculates Normalized Discounted Cumulative Gain at K."""
    actual_dcg = dcg_at_k(retrieved_relevances, k=k)
    ideal_scores = sorted(all_ground_truth_relevances, reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_scores, k=k)
    if ideal_dcg <= 0.0:
        return 1.0 if actual_dcg <= 0.0 else 0.0
    return min(1.0, actual_dcg / ideal_dcg)

def recall_at_k(retrieved_ids: List[str], ground_truth_relevant_ids: Set[str], k: int = 10) -> float:
    """Calculates Recall at K over relevant items (relevance >= 2)."""
    if not ground_truth_relevant_ids:
        return 1.0
    top_k_ids = set(retrieved_ids[:k])
    matched = len(top_k_ids.intersection(ground_truth_relevant_ids))
    return matched / min(k, len(ground_truth_relevant_ids))

def mrr_at_k(retrieved_ids: List[str], ground_truth_relevant_ids: Set[str], k: int = 10) -> float:
    """Calculates Mean Reciprocal Rank at K (rank of first relevant item)."""
    if not ground_truth_relevant_ids:
        return 1.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in ground_truth_relevant_ids:
            return 1.0 / i
    return 0.0

def evaluate_constraint_satisfaction(
    retrieved_items: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluates what proportion of top-K items strictly satisfy constraints
    vs violate one or more hard constraints.
    """
    top_items = retrieved_items[:k]
    if not top_items:
        return {"satisfaction_rate": 0.0, "violation_rate": 1.0, "perfect_query_success": 0.0}

    cat_req = constraints.get("category")
    subcat_req = constraints.get("subcategory")
    col_req = constraints.get("color")
    max_p = constraints.get("max_price")
    min_p = constraints.get("min_price")
    gen_req = constraints.get("gender")
    wp_req = constraints.get("is_waterproof")

    total_satisfied_items = 0
    total_violating_items = 0

    for item in top_items:
        p_cat = item.get("category", "")
        p_subcat = item.get("subcategory", "")
        p_col = item.get("color", "")
        p_price = float(item.get("selling_price", 0.0))
        p_gen = item.get("gender", "Unisex")
        p_wp = bool(item.get("is_waterproof", False))

        is_valid = True
        
        # Category check
        if cat_req and p_cat.lower() != cat_req.lower():
            is_valid = False
        # Color check
        if col_req and p_col.lower() != col_req.lower():
            is_valid = False
        # Price check
        if max_p and p_price > max_p:
            is_valid = False
        if min_p and p_price < min_p:
            is_valid = False
        # Gender check
        if gen_req and p_gen.lower() not in [gen_req.lower(), "unisex"]:
            is_valid = False
        # Waterproof check
        if wp_req is not None and wp_req and not p_wp:
            is_valid = False

        if is_valid:
            total_satisfied_items += 1
        else:
            total_violating_items += 1

    satisfaction_rate = total_satisfied_items / len(top_items)
    violation_rate = total_violating_items / len(top_items)
    # Perfect query success = at least 80% of top-10 satisfy all constraints
    perfect_success = 1.0 if satisfaction_rate >= 0.8 else 0.0

    return {
        "satisfaction_rate": round(satisfaction_rate, 4),
        "violation_rate": round(violation_rate, 4),
        "perfect_query_success": perfect_success
    }
