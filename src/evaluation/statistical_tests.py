"""
Statistical Significance Testing and Bootstrap Confidence Intervals for IR evaluation.
"""
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

def compute_bootstrap_ci(
    differences: np.ndarray,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Computes empirical bootstrap confidence interval for mean performance difference.
    """
    np.random.seed(random_seed)
    n = len(differences)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    boot_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(differences))

    return {
        "mean_diff": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "significant": bool(ci_lower > 0 or ci_upper < 0)
    }

def run_paired_significance_tests(
    baseline_scores: np.ndarray,
    proposed_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Runs Paired Student's t-test and Wilcoxon signed-rank test.
    """
    diffs = proposed_scores - baseline_scores
    n = len(diffs)
    
    # 1. Paired t-test
    t_stat, t_pval = stats.ttest_rel(proposed_scores, baseline_scores)
    
    # 2. Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(proposed_scores, baseline_scores)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    bootstrap_res = compute_bootstrap_ci(diffs)

    return {
        "n_samples": int(n),
        "mean_baseline": round(float(np.mean(baseline_scores)), 4),
        "mean_proposed": round(float(np.mean(proposed_scores)), 4),
        "absolute_gain": round(float(np.mean(diffs)), 4),
        "relative_gain_percent": round(float((np.mean(diffs) / max(np.mean(baseline_scores), 1e-6)) * 100), 2),
        "paired_ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(t_pval),
            "is_significant_p05": bool(t_pval < 0.05),
            "is_significant_p01": bool(t_pval < 0.01)
        },
        "wilcoxon_test": {
            "w_statistic": round(float(w_stat), 4),
            "p_value": float(w_pval),
            "is_significant": bool(w_pval < 0.05)
        },
        "bootstrap_95_ci": bootstrap_res
    }
