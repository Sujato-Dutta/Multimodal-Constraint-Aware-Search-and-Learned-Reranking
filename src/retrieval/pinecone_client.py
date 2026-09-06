"""
Pinecone Vector Database Client Integration with Local In-Memory Fallback.
Provides unified candidate retrieval, metadata filtering, and index management.
"""
from pathlib import Path
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class LocalPineconeIndex:
    """
    Self-contained, in-memory Vector Index implementing Pinecone-compatible
    cosine similarity query, upsert, and metadata filtering ($eq, $in, $lte, $gte).
    """
    def __init__(self, dimension: int = 512, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []
        self._dirty = False

    def upsert(self, vectors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upserts list of {'id': str, 'values': List[float], 'metadata': dict}"""
        for item in vectors:
            v_id = str(item["id"])
            vec = np.array(item["values"], dtype=np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.vectors[v_id] = vec
            self.metadata[v_id] = item.get("metadata", {})
            
        self._dirty = True
        return {"upserted_count": len(vectors)}

    def _sync_matrix(self):
        if self._dirty or self._matrix is None:
            self._id_list = list(self.vectors.keys())
            if self._id_list:
                self._matrix = np.vstack([self.vectors[vid] for vid in self._id_list])
            else:
                self._matrix = np.zeros((0, self.dimension), dtype=np.float32)
            self._dirty = False

    def query(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 50,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Executes cosine similarity query with optional metadata filtering.
        """
        self._sync_matrix()
        if len(self._id_list) == 0:
            return {"matches": []}

        q_vec = np.array(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Compute cosine similarities
        scores = np.dot(self._matrix, q_vec)

        # Apply metadata filters if supplied
        candidate_indices = np.arange(len(self._id_list))
        if filter:
            candidate_indices = [
                i for i in candidate_indices
                if self._matches_filter(self.metadata[self._id_list[i]], filter)
            ]
            if not candidate_indices:
                return {"matches": []}
            candidate_indices = np.array(candidate_indices)
            candidate_scores = scores[candidate_indices]
        else:
            candidate_scores = scores

        # Top-K selection
        k = min(top_k, len(candidate_indices))
        top_sub_idx = np.argpartition(-candidate_scores, k - 1)[:k]
        sorted_sub_idx = top_sub_idx[np.argsort(-candidate_scores[top_sub_idx])]

        matches = []
        for idx in sorted_sub_idx:
            orig_idx = candidate_indices[idx]
            doc_id = self._id_list[orig_idx]
            match_dict = {
                "id": doc_id,
                "score": float(scores[orig_idx]),
            }
            if include_metadata:
                match_dict["metadata"] = self.metadata.get(doc_id, {})
            matches.append(match_dict)

        return {"matches": matches}

    def _matches_filter(self, doc_meta: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Matches Pinecone-style filter expressions."""
        for field, condition in filter_dict.items():
            val = doc_meta.get(field)
            if isinstance(condition, dict):
                for op, op_val in condition.items():
                    if op == "$eq" and val != op_val:
                        return False
                    elif op == "$ne" and val == op_val:
                        return False
                    elif op == "$in" and val not in op_val:
                        return False
                    elif op == "$nin" and val in op_val:
                        return False
                    elif op == "$lte" and (val is None or val > op_val):
                        return False
                    elif op == "$gte" and (val is None or val < op_val):
                        return False
                    elif op == "$lt" and (val is None or val >= op_val):
                        return False
                    elif op == "$gt" and (val is None or val <= op_val):
                        return False
            else:
                # Direct equality match
                if val != condition:
                    return False
        return True

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._sync_matrix()
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "id_list": self._id_list,
            "metadata": self.metadata,
        }
        np.savez_compressed(filepath, matrix=self._matrix, **data)
        logger.info(f"Saved LocalPineconeIndex with {len(self._id_list)} vectors to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "LocalPineconeIndex":
        logger.info(f"Loading LocalPineconeIndex from {filepath}...")
        loader = np.load(filepath, allow_pickle=True)
        idx = cls(dimension=int(loader["dimension"]), metric=str(loader["metric"]))
        idx._matrix = loader["matrix"]
        idx._id_list = list(loader["id_list"])
        idx.metadata = loader["metadata"].item()
        idx.vectors = {vid: idx._matrix[i] for i, vid in enumerate(idx._id_list)}
        idx._dirty = False
        logger.info(f"Loaded LocalPineconeIndex with {len(idx._id_list)} vectors.")
        return idx


class PineconeManager:
    """
    Manager for Pinecone Cloud Index with automatic LocalPineconeIndex fallback.
    """
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or config.retrieval.pinecone_api_key
        self.index_name = index_name or config.retrieval.pinecone_index_name
        self.local_index_path = config.dataset.embeddings_dir / "pinecone_local_index.npz"
        self.pinecone_client = None
        self.index = None
        self.is_cloud = False
        
        self._initialize()

    def _initialize(self):
        if self.api_key and self.api_key != "YOUR_PINECONE_API_KEY":
            try:
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=self.api_key)
                # Check existing indexes
                existing = [idx["name"] for idx in pc.list_indexes()]
                if self.index_name not in existing:
                    logger.info(f"Creating Pinecone Cloud Index '{self.index_name}'...")
                    pc.create_index(
                        name=self.index_name,
                        dimension=config.retrieval.dimension,
                        metric=config.retrieval.metric,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                self.index = pc.Index(self.index_name)
                self.is_cloud = True
                logger.info(f"Connected to Pinecone Cloud Index: {self.index_name}")
                return
            except Exception as e:
                logger.warning(f"Pinecone Cloud connection failed: {e}. Switching to LocalPineconeIndex fallback.")

        # Fallback to local index
        if self.local_index_path.exists():
            self.index = LocalPineconeIndex.load(self.local_index_path)
        else:
            self.index = LocalPineconeIndex(dimension=config.retrieval.dimension, metric=config.retrieval.metric)
        self.is_cloud = False
        logger.info("Operating on LocalPineconeIndex.")

    def index_catalog(self, df: pd.DataFrame, embeddings: np.ndarray, batch_size: int = 250):
        """Indexes products and metadata into Pinecone or LocalPineconeIndex."""
        logger.info(f"Indexing {len(df)} catalog items into {'Pinecone Cloud' if self.is_cloud else 'LocalPineconeIndex'}...")
        
        records = []
        for idx, row in df.iterrows():
            item_id = str(row["product_id"])
            meta = {
                "name": str(row["name"]),
                "category": str(row["category"]),
                "subcategory": str(row.get("subcategory", "")),
                "color": str(row["color"]),
                "selling_price": float(row["selling_price"]),
                "original_price": float(row.get("original_price", row["selling_price"])),
                "brand": str(row.get("brand", "Adidas")),
                "gender": str(row.get("gender", "Unisex")),
                "rating": float(row.get("rating", 4.5)),
                "is_waterproof": bool(row.get("is_waterproof", False)),
                "is_running": bool(row.get("is_running", False)),
                "image_url": str(row.get("image_url", ""))
            }
            records.append({
                "id": item_id,
                "values": embeddings[idx].tolist(),
                "metadata": meta
            })

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(batch)

        if not self.is_cloud:
            self.index.save(self.local_index_path)
            
        logger.info(f"Indexing complete! Total items indexed: {len(records)}")

    def query(
        self,
        vector: Union[List[float], np.ndarray],
        top_k: int = 50,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries the vector index and returns top-K match dictionaries."""
        if isinstance(vector, np.ndarray):
            vector = vector.flatten().tolist()

        if self.is_cloud:
            res = self.index.query(vector=vector, top_k=top_k, filter=filter_dict, include_metadata=True)
            return res.to_dict().get("matches", [])
        else:
            res = self.index.query(vector=vector, top_k=top_k, filter=filter_dict, include_metadata=True)
            return res.get("matches", [])

if __name__ == "__main__":
    pm = PineconeManager()
    dummy_vec = np.random.randn(512).astype(np.float32)
    res = pm.query(dummy_vec, top_k=5)
    print(f"Query returned {len(res)} results.")
