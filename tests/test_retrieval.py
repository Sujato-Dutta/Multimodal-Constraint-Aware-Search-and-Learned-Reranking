"""
Unit tests for LocalPineconeIndex and retrieval mechanics.
"""
import pytest
import numpy as np
from src.retrieval.pinecone_client import LocalPineconeIndex, PineconeManager

def test_local_pinecone_index_upsert_and_query():
    idx = LocalPineconeIndex(dimension=4, metric="cosine")
    
    # Create 3 test vectors
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    v3 = [0.7071, 0.7071, 0.0, 0.0]

    idx.upsert([
        {"id": "doc1", "values": v1, "metadata": {"category": "Footwear", "price": 100}},
        {"id": "doc2", "values": v2, "metadata": {"category": "Apparel", "price": 50}},
        {"id": "doc3", "values": v3, "metadata": {"category": "Footwear", "price": 80}},
    ])

    # Query with vector close to doc1
    res = idx.query(vector=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(res["matches"]) == 2
    assert res["matches"][0]["id"] == "doc1"
    assert res["matches"][0]["score"] > 0.8

def test_local_pinecone_metadata_filtering():
    idx = LocalPineconeIndex(dimension=4, metric="cosine")
    idx.upsert([
        {"id": "shoe1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "Footwear", "selling_price": 120.0}},
        {"id": "jacket1", "values": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "Apparel", "selling_price": 80.0}},
        {"id": "shoe2", "values": [0.8, 0.2, 0.0, 0.0], "metadata": {"category": "Footwear", "selling_price": 70.0}},
    ])

    # Query with category=Footwear and price <= 100
    res = idx.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        filter={"category": "Footwear", "selling_price": {"$lte": 100.0}}
    )
    assert len(res["matches"]) == 1
    assert res["matches"][0]["id"] == "shoe2"
