"""
Unit tests for ranking and constraint evaluation metrics.
"""
import pytest
import numpy as np
from src.evaluation.metrics import ndcg_at_k, recall_at_k, mrr_at_k, evaluate_constraint_satisfaction
from src.evaluation.statistical_tests import compute_bootstrap_ci, run_paired_significance_tests

def test_ndcg_at_k_perfect_and_zero():
    # Perfect order
    retrieved = [3, 2, 1, 0]
    ideal = [3, 2, 1, 0]
    assert ndcg_at_k(retrieved, ideal, k=4) == pytest.approx(1.0, rel=1e-3)

    # Inverted order
    retrieved_bad = [0, 0, 1, 3]
    score_bad = ndcg_at_k(retrieved_bad, ideal, k=4)
    assert 0.0 <= score_bad < 0.7

def test_recall_and_mrr():
    gold = {"docA", "docB", "docC"}
    retrieved = ["docX", "docA", "docY", "docB"]
    
    assert recall_at_k(retrieved, gold, k=4) == pytest.approx(2 / 3, rel=1e-3)
    assert mrr_at_k(retrieved, gold, k=4) == pytest.approx(1 / 2, rel=1e-3)

def test_constraint_satisfaction_eval():
    items = [
        {"category": "Footwear", "color": "Black", "selling_price": 100.0},
        {"category": "Footwear", "color": "Black", "selling_price": 150.0}, # price violates max 120
    ]
    constraints = {"category": "Footwear", "color": "Black", "max_price": 120.0}
    res = evaluate_constraint_satisfaction(items, constraints, k=2)
    assert res["satisfaction_rate"] == 0.5
    assert res["violation_rate"] == 0.5

def test_bootstrap_ci():
    diffs = np.array([0.15, 0.20, 0.18, 0.22, 0.19, 0.25, 0.16])
    ci = compute_bootstrap_ci(diffs, num_bootstrap=500)
    assert ci["ci_lower"] > 0.10
    assert ci["ci_upper"] < 0.30
    assert ci["significant"] is True
