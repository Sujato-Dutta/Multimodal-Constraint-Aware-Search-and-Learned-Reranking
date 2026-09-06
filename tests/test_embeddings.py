"""
Unit tests for CLIP multimodal embedding encoder and caching.
"""
import pytest
import numpy as np
from PIL import Image
from src.embeddings.clip_encoder import CLIPEncoder

@pytest.fixture(scope="module")
def encoder():
    return CLIPEncoder()

def test_text_encoding_shape_and_norm(encoder):
    texts = ["black running shoes", "waterproof outdoor jacket"]
    embs = encoder.encode_text(texts)
    assert embs.shape == (2, 512)
    for i in range(2):
        norm = np.linalg.norm(embs[i])
        assert norm == pytest.approx(1.0, rel=1e-3)

def test_image_encoding_shape_and_norm(encoder):
    dummy_img = Image.new("RGB", (224, 224), color=(30, 30, 30))
    embs = encoder.encode_image([dummy_img])
    assert embs.shape == (1, 512)
    norm = np.linalg.norm(embs[0])
    assert norm == pytest.approx(1.0, rel=1e-3)

def test_multimodal_fusion(encoder):
    dummy_img = Image.new("RGB", (224, 224), color=(10, 50, 100))
    fused = encoder.encode_multimodal(text="blue trail shoes", image=dummy_img, alpha=0.6)
    assert fused.shape == (1, 512)
    norm = np.linalg.norm(fused[0])
    assert norm == pytest.approx(1.0, rel=1e-3)
