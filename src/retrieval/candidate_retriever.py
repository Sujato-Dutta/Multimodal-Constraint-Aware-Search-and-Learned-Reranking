"""
Candidate Retriever for Pinecone Vector Index.
Fetches top candidate products with similarity scores and joined metadata.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from src.config import config
from src.retrieval.pinecone_client import PineconeManager

class CandidateRetriever:
    def __init__(self, pinecone_mgr: Optional[PineconeManager] = None):
        self.pinecone_mgr = pinecone_mgr or PineconeManager()

    def retrieve_candidates(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K candidates from Pinecone index matching the query vector.
        """
        matches = self.pinecone_mgr.query(
            vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        candidates = []
        for rank, match in enumerate(matches, start=1):
            meta = match.get("metadata", {})
            raw_img = str(meta.get("image_url", "")).strip()
            clean_img = raw_img.split("~")[0].strip() if raw_img else ""
            candidate = {
                "product_id": match["id"],
                "similarity_score": float(match["score"]),
                "initial_rank": rank,
                "name": meta.get("name", ""),
                "category": meta.get("category", ""),
                "subcategory": meta.get("subcategory", ""),
                "color": meta.get("color", ""),
                "selling_price": float(meta.get("selling_price", 0.0)),
                "original_price": float(meta.get("original_price", 0.0)),
                "brand": meta.get("brand", "Adidas"),
                "gender": meta.get("gender", "Unisex"),
                "rating": float(meta.get("rating", 4.5)),
                "is_waterproof": bool(meta.get("is_waterproof", False)),
                "is_running": bool(meta.get("is_running", False)),
                "image_url": clean_img
            }
            candidates.append(candidate)
            
        return candidates
