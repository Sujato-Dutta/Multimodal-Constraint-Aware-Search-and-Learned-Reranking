"""
Pydantic schemas for Search API requests, responses, and pipeline metadata.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ConstraintOverride(BaseModel):
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    gender: Optional[str] = None
    is_waterproof: Optional[bool] = None

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=50)
    constraints: Optional[ConstraintOverride] = None

class ImageSearchRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)
    constraints: Optional[ConstraintOverride] = None

class MultimodalSearchRequest(BaseModel):
    query: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight between text (alpha) and image (1-alpha)")
    constraints: Optional[ConstraintOverride] = None

class ProductCard(BaseModel):
    product_id: str
    name: str
    category: str
    subcategory: str
    color: str
    selling_price: float
    original_price: float
    brand: str
    gender: str
    rating: float
    is_waterproof: bool
    image_url: str
    similarity_score: float
    rerank_score: float
    initial_rank: int
    final_rank: int
    constraint_status: Dict[str, Any]

class PipelineTimings(BaseModel):
    constraint_parsing_ms: float
    clip_encoding_ms: float
    pinecone_retrieval_ms: float
    feature_extraction_ms: float
    lambdamart_reranking_ms: float
    total_ms: float

class QueryEvaluationMetrics(BaseModel):
    ndcg_at_10: float
    recall_at_10: float
    constraint_satisfaction_rate: float
    p95_latency_s: float
    vs_baseline_ndcg_gain_pct: float
    vs_baseline_recall_gain_pct: float
    vs_baseline_cs_gain_pct: float
    vs_baseline_latency_gain_pct: float

class BaselineComparisonCard(BaseModel):
    baseline_ndcg_at_10: float
    baseline_recall_at_10: float
    baseline_latency_s: float
    final_ndcg_at_10: float
    final_recall_at_10: float
    final_latency_s: float

class SearchResponse(BaseModel):
    query: str
    extracted_constraints: Dict[str, Any]
    results: List[ProductCard]
    total_candidates_retrieved: int
    timings: PipelineTimings
    evaluation: QueryEvaluationMetrics
    baseline_comparison: BaselineComparisonCard

class HealthResponse(BaseModel):
    status: str
    version: str
    catalog_items_count: int
    model_loaded: bool
    index_loaded: bool
    device: str

class SystemInfoResponse(BaseModel):
    app_name: str
    version: str
    description: str
    dataset_info: Dict[str, Any]
    models_info: Dict[str, Any]
    evaluation_summary: Dict[str, Any]
