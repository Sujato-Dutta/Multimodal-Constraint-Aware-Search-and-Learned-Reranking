"""
Step 5: Run Full Evaluation Suite & Significance Testing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import pandas as pd

from src.config import config
from src.data.dataset_loader import DatasetLoader
from src.embeddings.clip_encoder import CLIPEncoder
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.lambdamart_ranker import LambdaMARTRanker
from src.evaluation.evaluator import SearchEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 5: Running End-to-End Evaluation Benchmark ===")
    
    loader = DatasetLoader()
    df = loader.load_and_preprocess()

    test_q_file = config.dataset.queries_dir / "test_queries.json"
    with open(test_q_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    logger.info(f"Loaded {len(test_queries)} test queries.")

    encoder = CLIPEncoder()
    retriever = CandidateRetriever()
    ranker = LambdaMARTRanker()

    evaluator = SearchEvaluator(
        catalog_df=df,
        clip_encoder=encoder,
        retriever=retriever,
        ranker=ranker
    )

    results = evaluator.evaluate_query_suite(test_queries, top_k=10, candidate_k=50)

    print("\n" + "=" * 70)
    print("           MULTIMODAL SEARCH EVALUATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Test Queries Evaluated: {results['num_test_queries']}")
    print("-" * 70)
    print(f"{'Metric':<30} | {'Baseline (CLIP)':<16} | {'Proposed Reranker':<16} | {'Gain (%)':<10}")
    print("-" * 70)
    print(f"{'NDCG@10':<30} | {results['baseline']['ndcg@10']:<16.4f} | {results['proposed_system']['ndcg@10']:<16.4f} | {results['percentage_improvements']['ndcg@10']:>+8.2f}%")
    print(f"{'Recall@10':<30} | {results['baseline']['recall@10']:<16.4f} | {results['proposed_system']['recall@10']:<16.4f} | {results['percentage_improvements']['recall@10']:>+8.2f}%")
    print(f"{'MRR':<30} | {results['baseline']['mrr']:<16.4f} | {results['proposed_system']['mrr']:<16.4f} | {results['percentage_improvements']['mrr']:>+8.2f}%")
    print(f"{'Constraint Satisfaction':<30} | {results['baseline']['constraint_satisfaction_rate']:<16.4f} | {results['proposed_system']['constraint_satisfaction_rate']:<16.4f} | {results['percentage_improvements']['constraint_satisfaction']:>+8.2f}%")
    print(f"{'Multi-Constraint Success':<30} | {results['baseline']['multi_constraint_success_rate']:<16.4f} | {results['proposed_system']['multi_constraint_success_rate']:<16.4f} | {results['percentage_improvements']['multi_constraint_success']:>+8.2f}%")
    print("-" * 70)
    print(f"{'p50 Latency (ms)':<30} | {results['baseline']['latency_ms']['p50']:<16.2f} | {results['proposed_system']['latency_ms']['p50']:<16.2f} |")
    print(f"{'p95 Latency (ms)':<30} | {results['baseline']['latency_ms']['p95']:<16.2f} | {results['proposed_system']['latency_ms']['p95']:<16.2f} |")
    print(f"{'Throughput (QPS)':<30} | {'-':<16} | {results['proposed_system']['latency_ms']['throughput_qps']:<16.2f} |")
    print("=" * 70)
    
    stat_test = results["statistical_significance"]["ndcg@10"]
    print(f"Paired t-test p-value: {stat_test['paired_ttest']['p_value']:.6e} (Significant: {stat_test['paired_ttest']['is_significant_p01']})")
    ci = stat_test["bootstrap_95_ci"]
    print(f"Bootstrap 95% CI on NDCG@10 Gain: [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] (Mean: {ci['mean_diff']:+.4f})")
    print("=" * 70 + "\n")

    logger.info("Step 5 complete!")

if __name__ == "__main__":
    main()
