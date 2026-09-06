"""
Comprehensive End-to-End Search Evaluator.
Executes rigorous comparative evaluation of CLIP Baseline vs. Proposed Constraint-Aware System,
records production latency distributions, statistical significance, and error analysis.
"""
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import config, PROJECT_ROOT
from src.embeddings.clip_encoder import CLIPEncoder
from src.query_understanding.constraint_parser import ConstraintParser, ParsedConstraints
from src.retrieval.candidate_retriever import CandidateRetriever
from src.ranking.feature_extractor import RankingFeatureExtractor
from src.ranking.lambdamart_ranker import LambdaMARTRanker
from src.evaluation.metrics import ndcg_at_k, recall_at_k, mrr_at_k, evaluate_constraint_satisfaction
from src.evaluation.statistical_tests import run_paired_significance_tests

logger = logging.getLogger(__name__)

class SearchEvaluator:
    def __init__(
        self,
        catalog_df: pd.DataFrame,
        clip_encoder: Optional[CLIPEncoder] = None,
        retriever: Optional[CandidateRetriever] = None,
        ranker: Optional[LambdaMARTRanker] = None
    ):
        self.df = catalog_df.copy()
        self.clip_encoder = clip_encoder or CLIPEncoder()
        self.retriever = retriever or CandidateRetriever()
        self.ranker = ranker or LambdaMARTRanker()
        self.parser = ConstraintParser()
        self.feature_extractor = RankingFeatureExtractor()
        
        # Precomputed catalog embedding maps for fast feature extraction
        emb_dir = config.dataset.embeddings_dir
        if (emb_dir / "product_text_embeddings.npy").exists() and (emb_dir / "product_ids.json").exists():
            text_embs = np.load(emb_dir / "product_text_embeddings.npy")
            img_embs = np.load(emb_dir / "product_image_embeddings.npy")
            with open(emb_dir / "product_ids.json", "r", encoding="utf-8") as f:
                pids = json.load(f)
            self.catalog_text_embs = {pid: text_embs[i] for i, pid in enumerate(pids)}
            self.catalog_img_embs = {pid: img_embs[i] for i, pid in enumerate(pids)}
        else:
            self.catalog_text_embs = {}
            self.catalog_img_embs = {}

    def evaluate_query_suite(self, test_queries: List[Dict[str, Any]], top_k: int = 10, candidate_k: int = 50) -> Dict[str, Any]:
        """Runs full comparative evaluation on test queries."""
        logger.info(f"Evaluating {len(test_queries)} queries (Baseline vs. Proposed Reranker)...")
        
        baseline_ndcg5, proposed_ndcg5 = [], []
        baseline_ndcg10, proposed_ndcg10 = [], []
        baseline_rec5, proposed_rec5 = [], []
        baseline_rec10, proposed_rec10 = [], []
        baseline_mrr, proposed_mrr = [], []
        baseline_cs, proposed_cs = [], []
        baseline_cv, proposed_cv = [], []
        baseline_perf_succ, proposed_perf_succ = [], []

        baseline_latencies = []
        proposed_latencies = []
        retrieval_latencies = []
        reranking_latencies = []

        query_records = []
        error_analysis = {
            "visually_similar_constraint_invalid": [],
            "semantic_mismatches": [],
            "attribute_parsing_failures": []
        }

        for q in test_queries:
            q_text = q["text"]
            q_modality = q.get("modality", "multimodal")
            gt_relevance = q.get("ground_truth_relevance", {})
            gold_relevant_ids = set([pid for pid, score in gt_relevance.items() if score >= 2])
            ideal_relevances = list(gt_relevance.values())

            # 1. Parse constraints
            t0_parse = time.perf_counter()
            parsed_constraints = self.parser.parse(q_text)
            t1_parse = time.perf_counter()

            # 2. Encode query
            t0_enc = time.perf_counter()
            if q_modality == "text_only":
                q_text_emb = self.clip_encoder.encode_text([q_text])[0]
                q_img_emb = None
                q_fused_emb = q_text_emb.reshape(1, -1)
            elif q_modality == "image_only":
                ref_img = q.get("reference_image_url")
                q_img_emb = self.clip_encoder.encode_image([ref_img])[0]
                q_text_emb = None
                q_fused_emb = q_img_emb.reshape(1, -1)
            else: # multimodal or multi_constraint or hard_negative
                q_text_emb = self.clip_encoder.encode_text([q_text])[0]
                ref_img = q.get("reference_image_url")
                q_img_emb = self.clip_encoder.encode_image([ref_img])[0]
                q_fused_emb = self.clip_encoder.encode_multimodal(text=q_text, image=ref_img)
            t1_enc = time.perf_counter()

            # 3. Retrieve candidates
            t0_ret = time.perf_counter()
            raw_candidates = self.retriever.retrieve_candidates(
                query_vector=q_fused_emb,
                top_k=candidate_k
            )
            t1_ret = time.perf_counter()
            ret_latency_ms = (t1_ret - t0_ret) * 1000.0
            retrieval_latencies.append(ret_latency_ms)

            # Baseline top-10 is the pure similarity ranking
            baseline_candidates = raw_candidates[:top_k]
            baseline_latency_ms = (t1_enc - t0_enc + t1_ret - t0_ret) * 1000.0
            baseline_latencies.append(baseline_latency_ms)

            # 4. Feature Extraction & Reranking
            t0_rerank = time.perf_counter()
            features, statuses = self.feature_extractor.batch_extract(
                query_constraints=parsed_constraints,
                query_text_emb=q_text_emb,
                query_img_emb=q_img_emb,
                query_fused_emb=q_fused_emb,
                candidates=raw_candidates,
                catalog_text_embeddings=self.catalog_text_embs,
                catalog_img_embeddings=self.catalog_img_embs
            )
            proposed_all = self.ranker.rerank(raw_candidates, features, statuses)
            proposed_candidates = proposed_all[:top_k]
            t1_rerank = time.perf_counter()

            rerank_latency_ms = (t1_rerank - t0_rerank) * 1000.0
            reranking_latencies.append(rerank_latency_ms)
            
            total_proposed_latency_ms = (t1_parse - t0_parse + t1_enc - t0_enc + t1_ret - t0_ret + t1_rerank - t0_rerank) * 1000.0
            proposed_latencies.append(total_proposed_latency_ms)

            # 5. Compute Metrics
            # Baseline metrics
            b_ids = [c["product_id"] for c in baseline_candidates]
            b_rels = [gt_relevance.get(pid, 0.0) for pid in b_ids]
            b_n5 = ndcg_at_k(b_rels, ideal_relevances, k=5)
            b_n10 = ndcg_at_k(b_rels, ideal_relevances, k=10)
            b_r5 = recall_at_k(b_ids, gold_relevant_ids, k=5)
            b_r10 = recall_at_k(b_ids, gold_relevant_ids, k=10)
            b_mrr_val = mrr_at_k(b_ids, gold_relevant_ids, k=10)
            b_cs_eval = evaluate_constraint_satisfaction(baseline_candidates, q.get("constraints", {}), k=10)

            baseline_ndcg5.append(b_n5)
            baseline_ndcg10.append(b_n10)
            baseline_rec5.append(b_r5)
            baseline_rec10.append(b_r10)
            baseline_mrr.append(b_mrr_val)
            baseline_cs.append(b_cs_eval["satisfaction_rate"])
            baseline_cv.append(b_cs_eval["violation_rate"])
            baseline_perf_succ.append(b_cs_eval["perfect_query_success"])

            # Proposed metrics
            p_ids = [c["product_id"] for c in proposed_candidates]
            p_rels = [gt_relevance.get(pid, 0.0) for pid in p_ids]
            p_n5 = ndcg_at_k(p_rels, ideal_relevances, k=5)
            p_n10 = ndcg_at_k(p_rels, ideal_relevances, k=10)
            p_r5 = recall_at_k(p_ids, gold_relevant_ids, k=5)
            p_r10 = recall_at_k(p_ids, gold_relevant_ids, k=10)
            p_mrr_val = mrr_at_k(p_ids, gold_relevant_ids, k=10)
            p_cs_eval = evaluate_constraint_satisfaction(proposed_candidates, q.get("constraints", {}), k=10)

            proposed_ndcg5.append(p_n5)
            proposed_ndcg10.append(p_n10)
            proposed_rec5.append(p_r5)
            proposed_rec10.append(p_r10)
            proposed_mrr.append(p_mrr_val)
            proposed_cs.append(p_cs_eval["satisfaction_rate"])
            proposed_cv.append(p_cs_eval["violation_rate"])
            proposed_perf_succ.append(p_cs_eval["perfect_query_success"])

            query_records.append({
                "query_id": q["query_id"],
                "text": q_text,
                "modality": q_modality,
                "baseline_ndcg10": b_n10,
                "proposed_ndcg10": p_n10,
                "baseline_cs": b_cs_eval["satisfaction_rate"],
                "proposed_cs": p_cs_eval["satisfaction_rate"],
                "gain_ndcg10": p_n10 - b_n10
            })

            # Error analysis mining
            # Case 1: Visually similar but constraint invalid items ranked high in baseline but demoted by reranker
            for idx, c in enumerate(baseline_candidates[:3]):
                if gt_relevance.get(c["product_id"], 0) == 0:
                    error_analysis["visually_similar_constraint_invalid"].append({
                        "query": q_text,
                        "baseline_rank": idx + 1,
                        "product_name": c["name"],
                        "color": c["color"],
                        "price": c["selling_price"],
                        "violated_constraint": f"Expected: {q.get('constraints', {})}, Found: {c['color']}, ${c['selling_price']}"
                    })
                    break

        # Statistical significance for NDCG@10
        stats_ndcg10 = run_paired_significance_tests(
            np.array(baseline_ndcg10),
            np.array(proposed_ndcg10)
        )

        # Statistical significance for Constraint Satisfaction
        stats_cs = run_paired_significance_tests(
            np.array(baseline_cs),
            np.array(proposed_cs)
        )

        results = {
            "num_test_queries": len(test_queries),
            "baseline": {
                "ndcg@5": round(float(np.mean(baseline_ndcg5)), 4),
                "ndcg@10": round(float(np.mean(baseline_ndcg10)), 4),
                "recall@5": round(float(np.mean(baseline_rec5)), 4),
                "recall@10": round(float(np.mean(baseline_rec10)), 4),
                "mrr": round(float(np.mean(baseline_mrr)), 4),
                "constraint_satisfaction_rate": round(float(np.mean(baseline_cs)), 4),
                "constraint_violation_rate": round(float(np.mean(baseline_cv)), 4),
                "multi_constraint_success_rate": round(float(np.mean(baseline_perf_succ)), 4),
                "latency_ms": {
                    "p50": round(float(np.percentile(baseline_latencies, 50)), 2),
                    "p90": round(float(np.percentile(baseline_latencies, 90)), 2),
                    "p95": round(float(np.percentile(baseline_latencies, 95)), 2),
                    "mean": round(float(np.mean(baseline_latencies)), 2)
                }
            },
            "proposed_system": {
                "ndcg@5": round(float(np.mean(proposed_ndcg5)), 4),
                "ndcg@10": round(float(np.mean(proposed_ndcg10)), 4),
                "recall@5": round(float(np.mean(proposed_rec5)), 4),
                "recall@10": round(float(np.mean(proposed_rec10)), 4),
                "mrr": round(float(np.mean(proposed_mrr)), 4),
                "constraint_satisfaction_rate": round(float(np.mean(proposed_cs)), 4),
                "constraint_violation_rate": round(float(np.mean(proposed_cv)), 4),
                "multi_constraint_success_rate": round(float(np.mean(proposed_perf_succ)), 4),
                "latency_ms": {
                    "p50": round(float(np.percentile(proposed_latencies, 50)), 2),
                    "p90": round(float(np.percentile(proposed_latencies, 90)), 2),
                    "p95": round(float(np.percentile(proposed_latencies, 95)), 2),
                    "mean": round(float(np.mean(proposed_latencies)), 2),
                    "retrieval_p95": round(float(np.percentile(retrieval_latencies, 95)), 2),
                    "reranking_p95": round(float(np.percentile(reranking_latencies, 95)), 2),
                    "throughput_qps": round(1000.0 / float(np.mean(proposed_latencies)), 2)
                }
            },
            "percentage_improvements": {
                "ndcg@10": round(((np.mean(proposed_ndcg10) - np.mean(baseline_ndcg10)) / np.mean(baseline_ndcg10)) * 100.0, 2),
                "recall@10": round(((np.mean(proposed_rec10) - np.mean(baseline_rec10)) / np.mean(baseline_rec10)) * 100.0, 2),
                "mrr": round(((np.mean(proposed_mrr) - np.mean(baseline_mrr)) / np.mean(baseline_mrr)) * 100.0, 2),
                "constraint_satisfaction": round(((np.mean(proposed_cs) - np.mean(baseline_cs)) / np.mean(baseline_cs)) * 100.0, 2),
                "multi_constraint_success": round(((np.mean(proposed_perf_succ) - np.mean(baseline_perf_succ)) / max(np.mean(baseline_perf_succ), 1e-4)) * 100.0, 2)
            },
            "statistical_significance": {
                "ndcg@10": stats_ndcg10,
                "constraint_satisfaction": stats_cs
            },
            "error_analysis_samples": error_analysis["visually_similar_constraint_invalid"][:5],
            "query_level_records": query_records
        }

        self._save_results_and_plots(results)
        return results

    def _save_results_and_plots(self, results: Dict[str, Any]):
        out_dir = config.evaluation.results_output_dir
        plots_dir = config.evaluation.plots_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {json_path}")

        # Generate comparison visual plots
        try:
            # 1. Bar Chart: Baseline vs Proposed Key Metrics
            metrics_keys = ["NDCG@10", "Recall@10", "MRR", "Constraint Sat.", "Multi-Constraint"]
            baseline_vals = [
                results["baseline"]["ndcg@10"],
                results["baseline"]["recall@10"],
                results["baseline"]["mrr"],
                results["baseline"]["constraint_satisfaction_rate"],
                results["baseline"]["multi_constraint_success_rate"]
            ]
            proposed_vals = [
                results["proposed_system"]["ndcg@10"],
                results["proposed_system"]["recall@10"],
                results["proposed_system"]["mrr"],
                results["proposed_system"]["constraint_satisfaction_rate"],
                results["proposed_system"]["multi_constraint_success_rate"]
            ]

            x = np.arange(len(metrics_keys))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (CLIP-only)", color="#64748b")
            rects2 = ax.bar(x + width/2, proposed_vals, width, label="Proposed (Constraint Reranker)", color="#0d9488")

            ax.set_ylabel("Score / Rate")
            ax.set_title("Search Quality & Constraint Compliance: Baseline vs Proposed Reranker")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_keys)
            ax.set_ylim(0, 1.1)
            ax.legend(frameon=True)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            fig.tight_layout()
            plot_path = plots_dir / "baseline_vs_proposed_metrics.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            logger.info(f"Saved benchmark plot to {plot_path}")

        except Exception as e:
            logger.warning(f"Plot generation skipped: {e}")
