"""
Dataset loader for the Adidas fashion retail products dataset.
Handles downloading via kagglehub, schema normalization, missing value handling,
and offline fallback catalog generation.
"""
from pathlib import Path
import os
import re
import json
import logging
import random
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from src.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Standard color taxonomy for normalization
COLOR_MAPPINGS = {
    "black": ["black", "core black", "carbon", "night black", "triple black", "dark grey", "coal"],
    "white": ["white", "ftwr white", "cloud white", "chalk white", "cream white", "off white", "triple white"],
    "blue": ["blue", "dark blue", "navy", "legend ink", "royal blue", "cyan", "sky blue", "marine", "bright blue"],
    "red": ["red", "solar red", "scarlet", "vivid red", "burgundy", "maroon", "crimson"],
    "grey": ["grey", "gray", "dash grey", "matte silver", "silver", "halo silver", "medium grey", "charcoal"],
    "green": ["green", "olive", "semi green", "signal green", "pulse green", "mint", "sage"],
    "yellow": ["yellow", "solar yellow", "gold", "semi solar yellow", "amber"],
    "pink": ["pink", "bliss pink", "pulse pink", "lucid pink", "rose"],
    "orange": ["orange", "solar orange", "coral", "terracotta"],
    "purple": ["purple", "violet", "indigo", "lilac"]
}

CATEGORY_MAPPINGS = {
    "footwear": ["shoes", "shoe", "sneakers", "sneaker", "running shoes", "slides", "cleats", "boots", "sandals", "trainers", "footwear"],
    "apparel": ["jacket", "jackets", "t-shirt", "tee", "shirt", "hoodie", "sweatshirt", "shorts", "pants", "tracksuit", "bra", "tights", "leggings", "jersey", "tank"],
    "accessories": ["bag", "backpack", "cap", "hat", "socks", "gloves", "ball", "water bottle", "beanie", "headband"]
}

SUBCATEGORY_MAPPINGS = {
    "running shoes": ["running", "ultraboost", "supernova", "adizero", "duramo", "galaxy", "runfalcon", "pureboost", "response super", "solarglide"],
    "jackets": ["jacket", "windbreaker", "parka", "vest", "bomber", "track top"],
    "shorts": ["shorts", "short"],
    "t-shirts": ["t-shirt", "tee", "tank", "polo"],
    "hoodies": ["hoodie", "sweatshirt", "fleece", "pullover"],
    "pants": ["pants", "trousers", "joggers", "track pants", "tights", "leggings"],
    "accessories": ["bag", "backpack", "cap", "socks", "hat", "ball"]
}

def normalize_color(raw_color: Any, raw_title: str = "") -> str:
    combined = f"{str(raw_color)} {raw_title}".lower()
    for standard_col, variants in COLOR_MAPPINGS.items():
        for variant in variants:
            if re.search(r'\b' + re.escape(variant) + r'\b', combined):
                return standard_col.capitalize()
    return "Black" if "black" in combined else "Multi"

def normalize_category(raw_cat: Any, raw_title: str = "") -> Tuple[str, str]:
    text = f"{str(raw_cat)} {raw_title}".lower()
    
    # Subcategory
    subcat = "General Footwear"
    for sc, kw_list in SUBCATEGORY_MAPPINGS.items():
        if any(kw in text for kw in kw_list):
            subcat = sc.title()
            break
            
    # Broad Category
    broad_cat = "Footwear"
    for cat, kw_list in CATEGORY_MAPPINGS.items():
        if any(kw in text for kw in kw_list):
            broad_cat = cat.title()
            break
            
    return broad_cat, subcat

def normalize_gender(raw_gender: Any, raw_title: str = "") -> str:
    text = f"{str(raw_gender)} {raw_title}".lower()
    if re.search(r'\b(women|woman|womens|female)\b', text):
        return "Women"
    elif re.search(r'\b(kids|kid|children|infant|toddler|boy|girl)\b', text):
        return "Kids"
    elif re.search(r'\b(men|man|mens|male)\b', text):
        return "Men"
    return "Unisex"

def clean_price(price_val: Any) -> float:
    if pd.isna(price_val):
        return 75.0
    if isinstance(price_val, (int, float)):
        return max(15.0, float(price_val))
    # String cleaning
    cleaned = re.sub(r'[^\d.]', '', str(price_val))
    try:
        val = float(cleaned)
        # If dataset uses INR (e.g. 5000), convert to approx USD for display standard ($65) or keep scaled
        if val > 1000:
            val = round(val / 80.0, 2)
        return max(15.0, val)
    except ValueError:
        return 75.0

class DatasetLoader:
    def __init__(self, raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or config.dataset.raw_dir
        self.processed_dir = processed_dir or config.dataset.processed_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(self) -> Path:
        """Downloads the Kaggle dataset via kagglehub or creates processed catalog."""
        logger.info("Attempting dataset download via kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download(config.dataset.kaggle_dataset_id)
            logger.info(f"Kaggle dataset downloaded to: {path}")
            return Path(path)
        except Exception as e:
            logger.warning(f"Kagglehub download failed or not configured: {e}. Checking local raw files...")
            return self.raw_dir

    def load_and_preprocess(self, force_recompute: bool = False) -> pd.DataFrame:
        """Loads raw dataset files, normalizes fields, and saves to processed_dir."""
        processed_path = self.processed_dir / "products_catalog.parquet"
        processed_csv = self.processed_dir / "products_catalog.csv"
        
        if not force_recompute and processed_path.exists():
            logger.info(f"Loading existing processed catalog from {processed_path}")
            df = pd.read_parquet(processed_path)
            return df
        
        download_path = self.download_kaggle_dataset()
        
        # Look for csv files in downloaded path or raw_dir
        csv_files = list(download_path.glob("**/*.csv")) + list(self.raw_dir.glob("**/*.csv"))
        
        if csv_files:
            logger.info(f"Found CSV dataset file: {csv_files[0]}")
            raw_df = pd.read_csv(csv_files[0])
            df = self._normalize_dataframe(raw_df)
        else:
            logger.info("No raw CSV found; generating comprehensive deterministic Adidas catalog...")
            df = self._generate_synthetic_adidas_catalog(n_items=config.dataset.max_items)
            
        logger.info(f"Processed {len(df)} catalog items successfully.")
        df.to_parquet(processed_path, index=False)
        df.to_csv(processed_csv, index=False)
        return df

    def _normalize_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Maps diverse raw schemas into standardized catalog schema."""
        df = raw_df.copy()
        
        # Determine column names
        col_map = {c.lower(): c for c in df.columns}
        
        id_col = col_map.get("sku") or col_map.get("product_id") or col_map.get("id") or df.columns[0]
        name_col = col_map.get("name") or col_map.get("title") or col_map.get("product_name") or df.columns[1]
        desc_col = col_map.get("description") or col_map.get("product_description") or name_col
        price_col = col_map.get("selling_price") or col_map.get("price") or col_map.get("discounted_price") or col_map.get("current_price")
        orig_price_col = col_map.get("original_price") or col_map.get("mrp") or col_map.get("retail_price") or price_col
        color_col = col_map.get("color") or col_map.get("colour") or col_map.get("color_family")
        cat_col = col_map.get("category") or col_map.get("product_type") or col_map.get("breadcrumbs")
        gender_col = col_map.get("gender") or col_map.get("department") or col_map.get("division")
        img_col = col_map.get("images") or col_map.get("image_url") or col_map.get("image") or col_map.get("url")
        rating_col = col_map.get("rating") or col_map.get("average_rating") or col_map.get("reviews_rating")
        
        records = []
        for idx, row in df.iterrows():
            prod_id = f"ADI-{row[id_col] if pd.notna(row[id_col]) else idx:05d}" if isinstance(row[id_col], int) else f"ADI-{str(row[id_col])[:10]}"
            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else f"Adidas Performance Product {idx}"
            desc = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else name
            
            raw_c = row[color_col] if color_col and pd.notna(row[color_col]) else ""
            color = normalize_color(raw_c, name)
            
            raw_k = row[cat_col] if cat_col and pd.notna(row[cat_col]) else ""
            broad_cat, subcat = normalize_category(raw_k, name)
            
            raw_g = row[gender_col] if gender_col and pd.notna(row[gender_col]) else ""
            gender = normalize_gender(raw_g, name)
            
            price = clean_price(row[price_col] if price_col and pd.notna(row[price_col]) else 75.0)
            orig_price = clean_price(row[orig_price_col] if orig_price_col and pd.notna(row[orig_price_col]) else price * 1.2)
            orig_price = max(orig_price, price)
            
            rating = float(row[rating_col]) if rating_col and pd.notna(row[rating_col]) and str(row[rating_col]).replace('.','',1).isdigit() else round(random.uniform(4.1, 4.9), 1)
            
            img_url = str(row[img_col]) if img_col and pd.notna(row[img_col]) else f"https://assets.adidas.com/images/w_600,f_auto,q_auto/{prod_id}.jpg"
            if img_url.startswith("[") and "'" in img_url:
                # Handle list representation
                try:
                    img_url = json.loads(img_url.replace("'", '"'))[0]
                except Exception:
                    pass

            records.append({
                "product_id": str(prod_id),
                "name": name,
                "description": desc,
                "category": broad_cat,
                "subcategory": subcat,
                "color": color,
                "color_details": str(raw_c) if raw_c else color,
                "selling_price": price,
                "original_price": orig_price,
                "brand": "Adidas",
                "gender": gender,
                "rating": rating,
                "reviews_count": random.randint(15, 850),
                "image_url": img_url,
                "is_waterproof": bool("waterproof" in desc.lower() or "gore-tex" in desc.lower() or "trail" in desc.lower()),
                "is_running": bool("running" in desc.lower() or "boost" in desc.lower() or "run" in name.lower()),
                "material": "Primeknit / Boost" if "boost" in name.lower() else "Recycled Polyester / Textile"
            })
            
        return pd.DataFrame(records)

    def _generate_synthetic_adidas_catalog(self, n_items: int = 9300) -> pd.DataFrame:
        """Generates realistic structured Adidas catalog covering footwear, apparel and accessories."""
        random.seed(42)
        np.random.seed(42)
        
        footwear_models = [
            ("Ultraboost Light", "High-performance responsive running shoes with light BOOST midsole and Primeknit+ upper.", 180.0, 200.0, ["running", "boost", "cushioning", "marathon"]),
            ("Adizero SL", "Lightweight elite competition training running shoes with Lightstrike cushioning.", 120.0, 130.0, ["running", "speed", "lightweight", "racing"]),
            ("Supernova Rise", "Engineered comfort daily running shoes with Dreamstrike+ superfoam.", 140.0, 150.0, ["running", "comfort", "daily trainer"]),
            ("Duramo SL 2", "Versatile lightweight running and gym trainers with breathable mesh.", 65.0, 75.0, ["running", "gym", "mesh", "lightweight"]),
            ("Galaxy 6", "Plush cushioned everyday athletic shoes with Cloudfoam midsole.", 60.0, 70.0, ["running", "cloudfoam", "daily"]),
            ("Pureboost 23", "Urban energy return running shoes with responsive full-length Boost.", 130.0, 140.0, ["running", "boost", "city"]),
            ("Runfalcon 5", "Durable all-day support running shoes with EVA cushioning and rubber outsole.", 55.0, 65.0, ["running", "support", "durable"]),
            ("Response Super 3.0", "Supportive fitness running shoes with Bounce and Boost hybrid heel insert.", 80.0, 90.0, ["running", "bounce", "fitness"]),
            ("Terrex Free Hiker 2 GORE-TEX", "Waterproof rugged trail running and hiking shoes with GORE-TEX membrane and Continental Rubber.", 200.0, 230.0, ["trail", "waterproof", "hiking", "gore-tex"]),
            ("Terrex Agravic Flow 2", "All-terrain waterproof trail running shoes with rock protection and Continental grip.", 140.0, 160.0, ["trail", "waterproof", "grip", "outdoor"]),
            ("Samba Classic", "Iconic indoor leather soccer and lifestyle street sneakers with gum sole.", 90.0, 100.0, ["lifestyle", "leather", "classic", "retro"]),
            ("Gazelle", "Heritage low-profile suede casual sneakers with contrast 3-Stripes.", 100.0, 110.0, ["lifestyle", "suede", "classic", "retro"]),
            ("Stan Smith", "Timeless clean minimalist tennis sneakers made with sustainable materials.", 100.0, 105.0, ["lifestyle", "leather", "minimalist", "clean"]),
            ("Forum Low", "Retro 80s basketball-inspired lifestyle sneakers with ankle strap detailing.", 110.0, 120.0, ["lifestyle", "basketball", "retro"]),
            ("Adilette Comfort Slides", "Ultra-soft contoured footbed slides for post-workout recovery.", 35.0, 40.0, ["slides", "recovery", "comfort"])
        ]
        
        apparel_models = [
            ("Own the Run Jacket", "Wind-resistant breathable running jacket with reflective 3-Stripes and zip pockets.", 75.0, 85.0, ["running", "jacket", "windbreaker", "reflective"]),
            ("Terrex Multi Light Down Jacket", "Insulated packable outdoor weather-resistant thermal jacket.", 160.0, 190.0, ["jacket", "outdoor", "insulated", "waterproof"]),
            ("Marathon 20 Running Shorts", "Moisture-wicking AEROREADY breathable lightweight running shorts with inner brief.", 35.0, 40.0, ["running", "shorts", "aeroready", "lightweight"]),
            ("Own the Run Shorts", "Classic athletic training shorts with sweat-guard zip pocket and 360 reflectivity.", 40.0, 45.0, ["running", "shorts", "training"]),
            ("Designed for Training Tee", "High-mobility gym workout t-shirt with flatlock anti-chafing seams.", 45.0, 50.0, ["training", "t-shirt", "gym", "aeroready"]),
            ("Essentials 3-Stripes Track Top", "Classic full-zip tricot heritage track jacket with ribbed cuffs.", 65.0, 75.0, ["jacket", "tracksuit", "classic"]),
            ("Tiro 24 Training Pants", "Tapered leg breathable football training track pants with ankle zips.", 50.0, 55.0, ["pants", "training", "tracksuit", "football"]),
            ("Optime Train Leggings", "High-waisted compressive workout tights with four-way stretch LYCRA.", 60.0, 70.0, ["pants", "leggings", "training", "yoga"])
        ]
        
        accessory_models = [
            ("Power Graphic Backpack", "Ergonomic multi-compartment athletic laptop backpack with air-mesh straps.", 45.0, 50.0, ["bag", "backpack", "gym", "storage"]),
            ("Superlite Linear 6-Pack Socks", "Moisture-wicking cushioned performance low-cut athletic running socks.", 22.0, 26.0, ["socks", "cushioned", "running"]),
            ("Superlite Trainer Cap", "UPF 50 sun protection breathable lightweight running cap with hook-and-loop backstrap.", 25.0, 30.0, ["cap", "hat", "running", "sun protection"])
        ]
        
        colors = ["Black", "White", "Blue", "Red", "Grey", "Green", "Yellow", "Pink", "Orange", "Purple"]
        genders = ["Men", "Women", "Unisex"]
        
        records = []
        item_idx = 1
        
        all_templates = (
            [(m, "Footwear") for m in footwear_models] * 120 +
            [(m, "Apparel") for m in apparel_models] * 80 +
            [(m, "Accessories") for m in accessory_models] * 40
        )
        random.shuffle(all_templates)
        
        for (model_name, desc_tmpl, base_price, base_orig, tags), broad_cat in all_templates:
            if item_idx > n_items:
                break
            
            color = random.choice(colors)
            gender = random.choice(genders)
            
            # Subcategory
            subcat = "General " + broad_cat
            for sc in ["Running Shoes", "Jackets", "Shorts", "T-Shirts", "Pants", "Accessories"]:
                if sc.lower() in model_name.lower() or any(sc.lower()[:-1] in t for t in tags):
                    subcat = sc
                    break
            
            # Variant name
            variant_suffix = f"{gender}'s " if gender != "Unisex" else ""
            color_desc = f"{color} / Core Black" if color == "Black" else f"{color} / Cloud White" if color in ["White", "Blue", "Red"] else f"{color} / Grey"
            full_name = f"{variant_suffix}{model_name}"
            
            # Slight price variation per colorway/gender
            price_mult = random.uniform(0.85, 1.15)
            price = round(base_price * price_mult, 2)
            orig_price = round(max(price, base_orig * price_mult), 2)
            
            prod_id = f"ADI-{item_idx:05d}"
            rating = round(random.uniform(4.0, 4.9), 1)
            
            is_waterproof = "waterproof" in desc_tmpl.lower() or "gore-tex" in desc_tmpl.lower() or "terrex" in model_name.lower()
            is_running = "running" in desc_tmpl.lower() or "boost" in model_name.lower() or "run" in model_name.lower()
            
            records.append({
                "product_id": prod_id,
                "name": full_name,
                "description": f"{full_name} in {color_desc}. {desc_tmpl} Features: {', '.join(tags)}.",
                "category": broad_cat,
                "subcategory": subcat,
                "color": color,
                "color_details": color_desc,
                "selling_price": price,
                "original_price": orig_price,
                "brand": "Adidas",
                "gender": gender,
                "rating": rating,
                "reviews_count": random.randint(25, 950),
                "image_url": f"https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=600&q=80" if broad_cat == "Footwear" else "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?auto=format&fit=crop&w=600&q=80",
                "is_waterproof": is_waterproof,
                "is_running": is_running,
                "material": "Primeknit / Boost" if "Boost" in model_name else "Recycled Polyester / Mesh"
            })
            item_idx += 1
            
        return pd.DataFrame(records)

if __name__ == "__main__":
    loader = DatasetLoader()
    df = loader.load_and_preprocess(force_recompute=True)
    print(f"Loaded {len(df)} products:")
    print(df[["product_id", "name", "category", "color", "selling_price"]].head(10))
