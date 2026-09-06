"""
Deterministic Rule-Based Query Understanding & Constraint Parser.
Extracts structured constraints (Category, Color, Price Range, Gender, Features, Style)
from natural language queries without requiring external or paid LLM APIs.
"""
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

COLOR_PATTERNS = {
    "Black": [r"\bblack\b", r"\bcore black\b", r"\bcarbon\b", r"\btriple black\b", r"\bcoal\b", r"\bdark grey\b"],
    "White": [r"\bwhite\b", r"\bcloud white\b", r"\bftwr white\b", r"\bchalk white\b", r"\btriple white\b", r"\boff white\b"],
    "Blue": [r"\bblue\b", r"\bnavy\b", r"\blegend ink\b", r"\broyal blue\b", r"\bcyan\b", r"\bsky blue\b"],
    "Red": [r"\bred\b", r"\bsolar red\b", r"\bscarlet\b", r"\bvivid red\b", r"\bburgundy\b", r"\bmaroon\b", r"\bcrimson\b"],
    "Grey": [r"\bgrey\b", r"\bgray\b", r"\bsilver\b", r"\bhalo silver\b", r"\bmatte silver\b", r"\bcharcoal\b"],
    "Green": [r"\bgreen\b", r"\bolive\b", r"\bsignal green\b", r"\bpulse green\b", r"\bmint\b", r"\bsage\b"],
    "Yellow": [r"\byellow\b", r"\bsolar yellow\b", r"\bgold\b", r"\bamber\b"],
    "Pink": [r"\bpink\b", r"\bbliss pink\b", r"\bpulse pink\b", r"\brose\b"],
    "Orange": [r"\borange\b", r"\bsolar orange\b", r"\bcoral\b"],
    "Purple": [r"\bpurple\b", r"\bviolet\b", r"\bindigo\b", r"\blilac\b"]
}

CATEGORY_PATTERNS = {
    "Footwear": {
        "Running Shoes": [r"\brunning\s+shoes?\b", r"\brunning\s+sneakers?\b", r"\btrail\s+running\b", r"\broad\s+shoes?\b", r"\bmarathon\b", r"\bjogging\s+shoes?\b", r"\brunning\b", r"\btrainers?\b"],
        "General Footwear": [r"\bshoes?\b", r"\bsneakers?\b", r"\bfootwear\b", r"\bcleats?\b", r"\bboots?\b", r"\bslides?\b", r"\bsandals?\b", r"\bkicks\b", r"\bcourt\s+shoes?\b"]
    },
    "Apparel": {
        "Jackets": [r"\bjackets?\b", r"\bwindbreakers?\b", r"\bouterwear\b", r"\bparkas?\b", r"\bvests?\b", r"\btrack\s+top\b", r"\bhoodies?\b", r"\bsweatshirts?\b"],
        "Shorts": [r"\bshorts?\b", r"\brunning\s+shorts?\b"],
        "T-Shirts": [r"\bt-shirts?\b", r"\btees?\b", r"\btanks?\b", r"\btops?\b", r"\bpolos?\b", r"\bjerseys?\b"],
        "Pants": [r"\bpants?\b", r"\btrousers?\b", r"\bjoggers?\b", r"\btrack\s+pants?\b", r"\bleggings?\b", r"\btights?\b"]
    },
    "Accessories": {
        "Accessories": [r"\bbackpacks?\b", r"\bbags?\b", r"\bsocks?\b", r"\bcaps?\b", r"\bhats?\b", r"\bbeanies?\b", r"\bgloves?\b", r"\bwater\s+bottles?\b"]
    }
}

GENDER_PATTERNS = {
    "Women": [r"\bwomen'?s?\b", r"\bwoman\b", r"\bfemale\b", r"\bladies\b"],
    "Men": [r"\bmen'?s?\b", r"\bman\b", r"\bmale\b"],
    "Kids": [r"\bkids?\b", r"\bchildren\b", r"\binfant\b", r"\btoddler\b", r"\bboys?\b", r"\bgirls?\b"]
}

STYLE_KEYWORDS = {
    "waterproof": [r"\bwaterproof\b", r"\bgore-tex\b", r"\bweather-resistant\b", r"\bwater-resistant\b", r"\brain\b"],
    "running": [r"\brunning\b", r"\brun\b", r"\bmarathon\b", r"\bjogging\b", r"\bboost\b"],
    "trail": [r"\btrail\b", r"\bhiking\b", r"\boutdoor\b", r"\ball-terrain\b", r"\bterrex\b"],
    "lightweight": [r"\blightweight\b", r"\blight\b", r"\bbreathable\b", r"\baeroready\b"],
    "cushioning": [r"\bcushioned\b", r"\bcushioning\b", r"\bcomfort\b", r"\bcloudfoam\b", r"\bboost\b", r"\bbounce\b"],
    "leather": [r"\bleather\b", r"\bsuede\b", r"\bclassic\b", r"\bretro\b"]
}

@dataclass
class ParsedConstraints:
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    gender: Optional[str] = None
    brand: Optional[str] = "Adidas"
    is_waterproof: Optional[bool] = None
    style_keywords: List[str] = field(default_factory=list)
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_constraints(self) -> bool:
        return any([
            self.category is not None,
            self.subcategory is not None,
            self.color is not None,
            self.max_price is not None,
            self.min_price is not None,
            self.gender is not None,
            self.is_waterproof is not None,
            len(self.style_keywords) > 0
        ])

    @property
    def active_constraints_count(self) -> int:
        count = 0
        if self.category: count += 1
        if self.color: count += 1
        if self.max_price or self.min_price: count += 1
        if self.gender: count += 1
        if self.is_waterproof is not None: count += 1
        if self.subcategory and self.subcategory != self.category: count += 1
        return max(count, 1)

class ConstraintParser:
    def __init__(self):
        pass

    def parse(self, query: str) -> ParsedConstraints:
        """Parses free-form query string and extracts all structured constraints."""
        if not query or not isinstance(query, str):
            return ParsedConstraints(raw_query="")

        text = query.lower().strip()
        parsed = ParsedConstraints(raw_query=query)

        # 1. Price Parsing
        # Range: "between $50 and $100" or "$50 - $100"
        range_match = re.search(r'(?:between\s+[\$₹]?\s*(\d+(?:\.\d+)?)\s+(?:and|to|-)\s+[\$₹]?\s*(\d+(?:\.\d+)?))|(?:[\$₹]?\s*(\d+(?:\.\d+)?)\s*-\s*[\$₹]?\s*(\d+(?:\.\d+)?))', text)
        if range_match:
            g = range_match.groups()
            p1 = float(g[0] or g[2])
            p2 = float(g[1] or g[3])
            # Check INR scale
            if p1 > 1000: p1 = round(p1 / 80.0, 2)
            if p2 > 1000: p2 = round(p2 / 80.0, 2)
            parsed.min_price = min(p1, p2)
            parsed.max_price = max(p1, p2)
        else:
            # Max price: "under $120", "below 100", "< 150", "less than $80", "under ₹5000"
            max_match = re.search(r'(?:under|below|less\s+than|max|up\s+to|<\s*=?)\s*[\$₹]?\s*(\d+(?:\.\d+)?)', text)
            if max_match:
                p = float(max_match.group(1))
                if p > 1000: p = round(p / 80.0, 2)
                parsed.max_price = p

            # Min price: "above $50", "over 40", "> 30", "at least $60"
            min_match = re.search(r'(?:above|over|more\s+than|min|at\s+least|>\s*=?)\s*[\$₹]?\s*(\d+(?:\.\d+)?)', text)
            if min_match:
                p = float(min_match.group(1))
                if p > 1000: p = round(p / 80.0, 2)
                parsed.min_price = p

        # 2. Color Parsing
        for standard_color, patterns in COLOR_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                parsed.color = standard_color
                break

        # 3. Category & Subcategory Parsing
        cat_found = False
        for broad_cat, subcats in CATEGORY_PATTERNS.items():
            for subcat_name, patterns in subcats.items():
                if any(re.search(p, text) for p in patterns):
                    parsed.category = broad_cat
                    parsed.subcategory = subcat_name
                    cat_found = True
                    break
            if cat_found:
                break
                
        # Default category heuristic if shoes/apparel keyword found
        if not parsed.category:
            if "shoe" in text or "sneaker" in text or "runner" in text or "cleat" in text:
                parsed.category = "Footwear"
                parsed.subcategory = "General Footwear"
            elif "jacket" in text or "shirt" in text or "pant" in text or "short" in text or "hoodie" in text:
                parsed.category = "Apparel"
            elif "bag" in text or "sock" in text or "cap" in text:
                parsed.category = "Accessories"

        # 4. Gender Parsing
        for g_name, patterns in GENDER_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                parsed.gender = g_name
                break

        # 5. Feature & Style Parsing
        for style_tag, patterns in STYLE_KEYWORDS.items():
            if any(re.search(p, text) for p in patterns):
                parsed.style_keywords.append(style_tag)
                if style_tag == "waterproof":
                    parsed.is_waterproof = True

        return parsed

if __name__ == "__main__":
    parser = ConstraintParser()
    test_queries = [
        "similar black running shoes under $120",
        "find something visually similar, black, waterproof, under ₹5,000",
        "women trail running shoes under $150 in grey",
        "men red athletic jacket between $50 and $100",
        "white classic sneakers over $80"
    ]
    for q in test_queries:
        res = parser.parse(q)
        print(f"\nQuery: '{q}'")
        print(f"Parsed: {res.to_dict()}")
