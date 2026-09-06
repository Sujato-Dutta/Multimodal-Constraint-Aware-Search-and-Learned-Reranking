"""
System-wide Configuration Management for Multimodal Constraint-Aware Search & Reranking.
"""
from pathlib import Path
import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DatasetConfig(BaseModel):
    kaggle_dataset_id: str = "thedevastator/adidas-fashion-retail-products-dataset-9300-prod"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    queries_dir: Path = PROJECT_ROOT / "data" / "queries"
    embeddings_dir: Path = PROJECT_ROOT / "data" / "embeddings"
    max_items: int = 9300
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

class EmbeddingsConfig(BaseModel):
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    batch_size: int = 64
    embedding_dim: int = 512
    normalize: bool = True

class RetrievalConfig(BaseModel):
    top_k_candidates: int = 50
    pinecone_index_name: str = "multimodal-constraint-search"
    metric: str = "cosine"
    dimension: int = 512
    use_local_fallback: bool = True
    pinecone_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))

class RankingConfig(BaseModel):
    model_type: str = "lightgbm"
    objective: str = "lambdarank"
    metric: str = "ndcg"
    ndcg_eval_at: List[int] = [5, 10]
    learning_rate: float = 0.05
    num_leaves: int = 31
    n_estimators: int = 150
    min_data_in_leaf: int = 10
    hard_negatives_per_query: int = 5
    model_output_path: Path = PROJECT_ROOT / "models" / "lambdamart_reranker.joblib"

class EvaluationConfig(BaseModel):
    k_values: List[int] = [5, 10]
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    results_output_dir: Path = PROJECT_ROOT / "experiments"
    plots_output_dir: Path = PROJECT_ROOT / "experiments" / "plots"

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

class AppConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    
    if not config_path.exists():
        return AppConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def resolve_paths(d, base=PROJECT_ROOT):
        for k, v in d.items():
            if isinstance(v, dict):
                resolve_paths(v, base)
            elif isinstance(v, str) and ("/" in v or "\\" in v) and not v.startswith("http") and not v.startswith("thedevastator") and not v.startswith("openai"):
                d[k] = base / v

    resolve_paths(data)

    dataset_cfg = DatasetConfig(**data.get("dataset", {}))
    emb_cfg = EmbeddingsConfig(**data.get("embeddings", {}))
    ret_cfg = RetrievalConfig(**data.get("retrieval", {}))
    rank_cfg = RankingConfig(**data.get("ranking", {}))
    eval_cfg = EvaluationConfig(**data.get("evaluation", {}))
    srv_cfg = ServerConfig(**data.get("server", {}))

    return AppConfig(
        dataset=dataset_cfg,
        embeddings=emb_cfg,
        retrieval=ret_cfg,
        ranking=rank_cfg,
        evaluation=eval_cfg,
        server=srv_cfg
    )

config = load_config()
