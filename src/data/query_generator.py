"""
Deterministic Query Suite Generator for Multimodal Constraint-Aware Search & Reranking.
Generates comprehensive Train, Validation, and Test search benchmarks covering:
- Text queries
- Image queries
- Multimodal (Image + Text) queries
- Multi-constraint compositional queries
- Adversarial hard-negative test cases (visually similar but constraint violating)
"""
from pathlib import Path
import json
import random
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from src.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Query Template Definitions
QUERY_TEMPLATES = [
    {"text": "similar black running shoes under $120", "category": "Footwear", "subcategory": "Running Shoes", "color": "Black", "max_price": 120.0, "gender": None, "type": "multimodal"},
    {"text": "black running shoes under $100", "category": "Footwear", "subcategory": "Running Shoes", "color": "Black", "max_price": 100.0, "gender": None, "type": "text_only"},
    {"text": "white ultraboost running shoes under $180", "category": "Footwear", "subcategory": "Running Shoes", "color": "White", "max_price": 180.0, "gender": None, "type": "text_only"},
    {"text": "women lightweight running shoes under $80", "category": "Footwear", "subcategory": "Running Shoes", "color": None, "max_price": 80.0, "gender": "Women", "type": "multi_constraint"},
    {"text": "men black cushioned marathon running shoes under $130", "category": "Footwear", "subcategory": "Running Shoes", "color": "Black", "max_price": 130.0, "gender": "Men", "type": "multi_constraint"},
    {"text": "blue athletic gym trainers under $75", "category": "Footwear", "subcategory": "Running Shoes", "color": "Blue", "max_price": 75.0, "gender": None, "type": "text_only"},
    {"text": "red performance running shoes under $90", "category": "Footwear", "subcategory": "Running Shoes", "color": "Red", "max_price": 90.0, "gender": None, "type": "multimodal"},
    {"text": "grey durable daily road shoes under $70", "category": "Footwear", "subcategory": "Running Shoes", "color": "Grey", "max_price": 70.0, "gender": None, "type": "text_only"},
    {"text": "waterproof trail running shoes under $150", "category": "Footwear", "subcategory": "Running Shoes", "color": None, "max_price": 150.0, "is_waterproof": True, "type": "multi_constraint"},
    {"text": "men waterproof outdoor terrex shoes under $200", "category": "Footwear", "subcategory": "Running Shoes", "color": None, "max_price": 200.0, "gender": "Men", "is_waterproof": True, "type": "hard_negative_adversarial"},
    {"text": "women waterproof hiking shoes in black under $160", "category": "Footwear", "subcategory": "Running Shoes", "color": "Black", "max_price": 160.0, "gender": "Women", "is_waterproof": True, "type": "multi_constraint"},
    {"text": "white classic leather sneakers under $100", "category": "Footwear", "subcategory": "General Footwear", "color": "White", "max_price": 100.0, "type": "text_only"},
    {"text": "black retro street lifestyle sneakers under $110", "category": "Footwear", "subcategory": "General Footwear", "color": "Black", "max_price": 110.0, "type": "multimodal"},
    {"text": "women casual court shoes under $90 in white", "category": "Footwear", "subcategory": "General Footwear", "color": "White", "max_price": 90.0, "gender": "Women", "type": "multi_constraint"},
    {"text": "black wind-resistant running jacket under $80", "category": "Apparel", "subcategory": "Jackets", "color": "Black", "max_price": 80.0, "type": "multi_constraint"},
    {"text": "women weather-resistant outdoor jacket under $170", "category": "Apparel", "subcategory": "Jackets", "color": None, "max_price": 170.0, "gender": "Women", "type": "text_only"},
    {"text": "men blue sports track top jacket under $70", "category": "Apparel", "subcategory": "Jackets", "color": "Blue", "max_price": 70.0, "gender": "Men", "type": "multimodal"},
    {"text": "red full zip athletic jacket under $75", "category": "Apparel", "subcategory": "Jackets", "color": "Red", "max_price": 75.0, "type": "hard_negative_adversarial"},
    {"text": "black breathable running shorts under $45", "category": "Apparel", "subcategory": "Shorts", "color": "Black", "max_price": 45.0, "type": "text_only"},
    {"text": "women workout gym shorts under $40 in blue", "category": "Apparel", "subcategory": "Shorts", "color": "Blue", "max_price": 40.0, "gender": "Women", "type": "multi_constraint"},
    {"text": "men training tee shirt in grey under $50", "category": "Apparel", "subcategory": "T-Shirts", "color": "Grey", "max_price": 50.0, "gender": "Men", "type": "text_only"},
    {"text": "white moisture-wicking athletic t-shirt under $40", "category": "Apparel", "subcategory": "T-Shirts", "color": "White", "max_price": 40.0, "type": "text_only"},
    {"text": "black tapered soccer training pants under $55", "category": "Apparel", "subcategory": "Pants", "color": "Black", "max_price": 55.0, "gender": "Men", "type": "text_only"},
    {"text": "women high-waisted compression leggings under $65", "category": "Apparel", "subcategory": "Pants", "color": "Black", "max_price": 65.0, "gender": "Women", "type": "multi_constraint"},
    {"text": "black multi-compartment athletic backpack under $50", "category": "Accessories", "subcategory": "Accessories", "color": "Black", "max_price": 50.0, "type": "text_only"},
    {"text": "white cushioned running socks under $25", "category": "Accessories", "subcategory": "Accessories", "color": "White", "max_price": 25.0, "type": "text_only"}
]

class QueryGenerator:
    def __init__(self, catalog_df: pd.DataFrame, queries_dir: Optional[Path] = None):
        self.df = catalog_df.copy()
        self.queries_dir = queries_dir or config.dataset.queries_dir
        self.queries_dir.mkdir(parents=True, exist_ok=True)
        # Pre-convert catalog to records for sub-second relevance mapping
        self.catalog_records = self.df.to_dict(orient="records")
        
    def generate_query_suite(self, num_train: int = 300, num_val: int = 60, num_test: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Generates disjoint Train, Validation, and Test query suites with ground-truth relevance."""
        logger.info("Generating deterministic multimodal query benchmark suite...")
        random.seed(config.dataset.random_seed)
        np.random.seed(config.dataset.random_seed)
        
        all_queries = []
        q_id = 1
        
        colors = ["Black", "White", "Blue", "Red", "Grey", "Green", "Yellow", "Pink"]
        price_caps = [40.0, 55.0, 70.0, 85.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
        genders = ["Men", "Women", "Unisex"]
        
        # 1. Base templates
        for tmpl in QUERY_TEMPLATES:
            q_obj = self._create_query_object(q_id, tmpl)
            all_queries.append(q_obj)
            q_id += 1
            
        categories = [
            ("Footwear", "Running Shoes", ["running shoes", "athletic trainers", "road shoes", "jogging sneakers"]),
            ("Footwear", "General Footwear", ["casual sneakers", "retro street shoes", "lifestyle sneakers", "court shoes"]),
            ("Apparel", "Jackets", ["windbreaker jacket", "running jacket", "track jacket", "weather-resistant jacket"]),
            ("Apparel", "Shorts", ["running shorts", "athletic workout shorts", "training shorts"]),
            ("Apparel", "T-Shirts", ["workout t-shirt", "training tee", "breathable athletic top"]),
            ("Apparel", "Pants", ["track pants", "training pants", "compression leggings"]),
            ("Accessories", "Accessories", ["sports backpack", "cushioned running socks", "running cap"])
        ]
        
        modalities = ["text_only", "image_only", "multimodal", "multi_constraint", "hard_negative_adversarial"]
        
        while len(all_queries) < (num_train + num_val + num_test):
            broad_cat, subcat, phrases = random.choice(categories)
            phrase = random.choice(phrases)
            color = random.choice(colors) if random.random() > 0.15 else None
            price = random.choice(price_caps)
            gender = random.choice(genders) if random.random() > 0.4 else None
            is_waterproof = True if ("jacket" in phrase or "shoes" in phrase) and random.random() > 0.75 else False
            modality = random.choice(modalities)
            
            tokens = []
            if modality in ["multimodal", "image_only"]:
                tokens.append("similar")
            if gender and gender != "Unisex":
                tokens.append(gender.lower())
            if color:
                tokens.append(color.lower())
            if is_waterproof:
                tokens.append("waterproof")
            tokens.append(phrase)
            tokens.append(f"under ${int(price)}")
            
            query_text = " ".join(tokens)
            
            tmpl = {
                "text": query_text,
                "category": broad_cat,
                "subcategory": subcat,
                "color": color,
                "max_price": price,
                "gender": gender,
                "is_waterproof": is_waterproof,
                "type": modality
            }
            
            q_obj = self._create_query_object(q_id, tmpl)
            all_queries.append(q_obj)
            q_id += 1
            
        random.shuffle(all_queries)
        
        train_queries = all_queries[:num_train]
        val_queries = all_queries[num_train:num_train + num_val]
        test_queries = all_queries[num_train + num_val:num_train + num_val + num_test]
        
        logger.info(f"Split {len(all_queries)} queries into Train: {len(train_queries)}, Val: {len(val_queries)}, Test: {len(test_queries)}")
        
        splits = {
            "train": self._annotate_ground_truth(train_queries),
            "val": self._annotate_ground_truth(val_queries),
            "test": self._annotate_ground_truth(test_queries)
        }
        
        for split_name, q_list in splits.items():
            out_file = self.queries_dir / f"{split_name}_queries.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(q_list, f, indent=2)
            logger.info(f"Saved {len(q_list)} {split_name} queries to {out_file}")
            
        return splits
        
    def _create_query_object(self, q_id: int, tmpl: Dict[str, Any]) -> Dict[str, Any]:
        category = tmpl.get("category", "Footwear")
        subcategory = tmpl.get("subcategory", "Running Shoes")
        color = tmpl.get("color")
        max_price = tmpl.get("max_price")
        gender = tmpl.get("gender")
        is_waterproof = tmpl.get("is_waterproof", False)
        modality = tmpl.get("type", "multimodal")
        
        matching_prods = self.df[self.df["category"] == category]
        if subcategory and not matching_prods[matching_prods["subcategory"] == subcategory].empty:
            matching_prods = matching_prods[matching_prods["subcategory"] == subcategory]
            
        ref_prod_id = matching_prods.iloc[0]["product_id"] if not matching_prods.empty else "ADI-00001"
        ref_prod_img = matching_prods.iloc[0]["image_url"] if not matching_prods.empty else "https://images.unsplash.com/photo-1542291026-7eec264c27ff"
        
        return {
            "query_id": f"Q-{q_id:04d}",
            "text": tmpl["text"],
            "modality": modality,
            "reference_product_id": ref_prod_id,
            "reference_image_url": ref_prod_img,
            "constraints": {
                "category": category,
                "subcategory": subcategory,
                "color": color,
                "max_price": max_price,
                "min_price": tmpl.get("min_price"),
                "gender": gender,
                "brand": "Adidas",
                "is_waterproof": is_waterproof
            }
        }
        
    def _annotate_ground_truth(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast vectorized evaluation of ground-truth relevance across catalog records."""
        annotated = []
        
        for q in queries:
            c = q["constraints"]
            cat_req = c.get("category")
            subcat_req = c.get("subcategory")
            col_req = c.get("color")
            max_p = c.get("max_price")
            gen_req = c.get("gender")
            wp_req = c.get("is_waterproof")
            
            ground_truth = {}
            
            for prod in self.catalog_records:
                pid = prod["product_id"]
                p_cat = prod["category"]
                p_subcat = prod["subcategory"]
                p_col = prod["color"]
                p_price = prod["selling_price"]
                p_gen = prod["gender"]
                p_wp = prod.get("is_waterproof", False)
                
                if cat_req and p_cat.lower() != cat_req.lower():
                    continue
                    
                subcat_ok = (not subcat_req) or (p_subcat.lower() == subcat_req.lower())
                color_ok = (not col_req) or (p_col.lower() == col_req.lower())
                price_ok = (not max_p) or (p_price <= max_p)
                gender_ok = (not gen_req) or (p_gen in [gen_req, "Unisex"])
                wp_ok = (not wp_req) or (p_wp == wp_req)
                
                all_hard_ok = color_ok and price_ok and gender_ok and wp_ok
                
                if all_hard_ok and subcat_ok:
                    score = 3
                elif all_hard_ok:
                    score = 2
                elif color_ok and price_ok:
                    score = 1
                else:
                    score = 0
                    
                if score > 0:
                    ground_truth[pid] = score
                    
            q_copy = dict(q)
            q_copy["ground_truth_relevance"] = ground_truth
            annotated.append(q_copy)
            
        return annotated
