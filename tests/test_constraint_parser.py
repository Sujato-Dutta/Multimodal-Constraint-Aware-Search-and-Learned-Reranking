"""
Unit tests for deterministic constraint parser.
"""
import pytest
from src.query_understanding.constraint_parser import ConstraintParser

@pytest.fixture
def parser():
    return ConstraintParser()

def test_price_parsing_max(parser):
    q = "black running shoes under $120"
    res = parser.parse(q)
    assert res.max_price == 120.0
    assert res.color == "Black"
    assert res.category == "Footwear"
    assert res.subcategory == "Running Shoes"

def test_price_parsing_range(parser):
    q = "men red athletic jacket between $50 and $100"
    res = parser.parse(q)
    assert res.min_price == 50.0
    assert res.max_price == 100.0
    assert res.color == "Red"
    assert res.gender == "Men"
    assert res.category == "Apparel"
    assert res.subcategory == "Jackets"

def test_waterproof_and_gender(parser):
    q = "women waterproof trail running shoes under $150 in grey"
    res = parser.parse(q)
    assert res.gender == "Women"
    assert res.is_waterproof is True
    assert res.color == "Grey"
    assert res.max_price == 150.0
    assert res.category == "Footwear"

def test_empty_and_generic_queries(parser):
    res = parser.parse("")
    assert res.category is None
    assert res.color is None
    assert res.max_price is None

    res2 = parser.parse("awesome kicks")
    assert res2.category == "Footwear"
