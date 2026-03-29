import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import KFold
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")

# ══════════════════════════════════════════
#  BLOCK 1 — DROP LEAKAGE COLUMNS
# ══════════════════════════════════════════

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that cause target leakage or are useless for modeling.

    Target Leakage = a feature that is derived from the target variable (price).
    If included, model 'cheats' during training but fails on unseen data.

    Columns dropped:
    - price_per_sqft : mathematically = price / area → direct leakage
    - property_name  : free text, high cardinality, leaks target
    - link           : URL, zero predictive value
    - description    : raw text, noisy, needs NLP pipeline (future scope)
    - property_id    : row identifier, not a feature
    - address        : raw string — sector already extracted separately
    - society        : will be target-encoded separately, raw string dropped
    """
    cols_to_drop = [
        'price_per_sqft', 'rate', 'areaWithType', 'property_name', 'link',
        'description', 'property_id', 'address',
    ]
    # Only drop columns that actually exist (safe for both houses + flats)
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"Dropped {len(existing)} leakage/noise columns. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════
#  BLOCK 2 — AMENITY SCORE
# ══════════════════════════════════════════

# Weighted amenity dictionary
# Luxury amenities score higher — they directly command price premium
AMENITY_WEIGHTS = {
    'swimming pool': 3, 'gym': 2, 'fitness centre / gym': 2,
    'club house / community center': 2, 'security personnel': 1,
    'lift(s)': 1, 'park': 1, 'intercom facility': 1,
    'power back-up': 1, 'visitor parking': 1,
    'maintenance staff': 1, 'piped-gas': 1,
    'rain water harvesting': 1, 'shopping centre': 2,
    'security / fire alarm': 1, 'water storage': 1,
}

def _safe_parse_list(val) -> list:
    """Safely parse a stringified Python list to actual list."""
    if pd.isna(val) or val == 'not available':
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

def build_amenity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw 'features' string-list into a weighted amenity_score (0-10).

    Strategy: Weighted scoring — luxury amenities (pool, gym) score higher
    than basic ones (lift, park). Score is capped at 10 for consistency.

    WHY weighted over binary columns?
    With 30+ amenities, binary OHE creates 30 sparse columns.
    A single dense score is more stable for tree splits and less prone
    to overfitting on rare amenities.
    """
    def score_amenities(val):
        amenities = _safe_parse_list(val)
        amenities_lower = [a.lower().strip() for a in amenities]
        score = sum(AMENITY_WEIGHTS.get(a, 0) for a in amenities_lower)
        return min(score, 10)  # cap at 10

    df['amenity_score'] = df['features'].apply(score_amenities)
    df = df.drop(columns=['features'])
    print(f"amenity_score built. Mean: {df['amenity_score'].mean():.2f}")
    return df


# ══════════════════════════════════════════
#  BLOCK 3 — FURNISH SCORE
# ══════════════════════════════════════════

def build_furnish_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts 'furnishDetails' string-list into furnish_score (0-1).

    Logic: Count appliances PRESENT (not 'No X') / total possible appliances.
    furnish_score = 1.0 means fully furnished, 0.0 means bare shell.

    Example:
        ['3 Fan', '2 AC', 'No Bed', 'No TV'] → 2 present / 10 possible = 0.2
    """
    POSSIBLE_APPLIANCES = [
        'ac', 'fan', 'geyser', 'light', 'bed', 'sofa', 'tv',
        'washing machine', 'fridge', 'microwave', 'wardrobe',
        'modular kitchen', 'curtains', 'dining table', 'water purifier'
    ]

    def score_furnish(val):
        items = _safe_parse_list(val)
        if not items:
            return 0.0
        # Count items that DON'T start with 'No'
        present = sum(
            1 for item in items
            if not str(item).strip().lower().startswith('no ')
            and any(ap in str(item).lower() for ap in POSSIBLE_APPLIANCES)
        )
        return round(min(present / len(POSSIBLE_APPLIANCES), 1.0), 2)

    df['furnish_score'] = df['furnishDetails'].apply(score_furnish)
    df = df.drop(columns=['furnishDetails'])
    print(f"furnish_score built. Mean: {df['furnish_score'].mean():.2f}")
    return df


# ══════════════════════════════════════════
#  BLOCK 4 — NEARBY LOCATION FEATURES
# ══════════════════════════════════════════

def build_nearby_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts binary proximity features from 'nearbyLocations' string-list.

    These binary flags capture location quality signals:
    - has_metro_nearby  → price premium for connectivity
    - has_hospital_nearby → essential infrastructure
    - has_school_nearby  → family-appeal signal
    - has_mall_nearby    → lifestyle amenity

    WHY binary instead of count?
    Presence/absence is more stable than count for sparse location data.
    """
    KEYWORDS = {
        'has_metro_nearby':    ['metro', 'metro station'],
        'has_hospital_nearby': ['hospital', 'medical', 'healthcare'],
        'has_school_nearby':   ['school', 'college', 'university', 'institute'],
        'has_mall_nearby':     ['mall', 'shopping', 'market'],
        'has_airport_nearby':  ['airport', 'intl airport'],
    }

    def check_keyword(val, keywords):
        locations = _safe_parse_list(val)
        text = ' '.join(str(l).lower() for l in locations)
        return int(any(kw in text for kw in keywords))

    for col, keywords in KEYWORDS.items():
        df[col] = df['nearbyLocations'].apply(lambda v: check_keyword(v, keywords))

    df = df.drop(columns=['nearbyLocations'])
    print(f"Nearby features built: {list(KEYWORDS.keys())}")
    return df


# ══════════════════════════════════════════
#  BLOCK 5 — RATING SCORE
# ══════════════════════════════════════════

def build_rating_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses 'rating' string-list into a single avg_rating float.

    Input:  "['Environment4 out of 5', 'Lifestyle4.5 out of 5', 'Safety4 out of 5']"
    Output: avg_rating = 4.17

    Regex extracts all numbers before 'out of 5' pattern.
    Missing rating → -1 flag (model learns 'no rating' as a signal, not noise)
    """
    import re

    def parse_rating(val):
        items = _safe_parse_list(val)
        if not items:
            return -1.0
        scores = []
        for item in items:
            match = re.search(r'(\d+\.?\d*)\s*out\s*of\s*5', str(item), re.IGNORECASE)
            if match:
                scores.append(float(match.group(1)))
        return round(np.mean(scores), 2) if scores else -1.0

    df['avg_rating'] = df['rating'].apply(parse_rating)
    df['has_rating'] = (df['avg_rating'] != -1).astype(int)
    df = df.drop(columns=['rating'])
    print(f"avg_rating built. Rated properties: {df['has_rating'].sum()}")
    return df


# ══════════════════════════════════════════
#  BLOCK 6 — AGE POSSESSION ENCODING
# ══════════════════════════════════════════

def build_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts messy agePossession strings into ordinal age_bucket (0-5).

    Ordinal encoding because there IS a meaningful order:
    Under Construction < New < 1-5yr < 5-10yr < 10yr+

    Date strings like 'Dec 2023' → parsed to approximate age category.
    'undefined' → -1 flag (handled separately, not imputed with median
    because missing might correlate with property type).
    """
    import re
    from datetime import datetime

    CURRENT_YEAR = 2025

    def map_age(val):
        if pd.isna(val) or str(val).lower() == 'undefined':
            return -1  # Missing signal — keep as -1, don't impute

        val = str(val).strip().lower()

        if 'under construction' in val:    return 0
        if 'within 3 months' in val:       return 0
        if 'within 6 months' in val:       return 0
        if '0 to 1' in val:                return 1
        if '1 to 5' in val:                return 2
        if '5 to 10' in val:               return 3
        if '10+' in val:                   return 4

        # Handle date strings like 'dec 2023', 'by 2024', 'jul 2027'
        year_match = re.search(r'(\d{4})', val)
        if year_match:
            year = int(year_match.group(1))
            diff = CURRENT_YEAR - year
            if diff <= 0:   return 0   # Future → under construction
            if diff <= 1:   return 1
            if diff <= 5:   return 2
            if diff <= 10:  return 3
            return 4

        return -1  # Fallback

    df['age_bucket'] = df['agePossession'].apply(map_age)
    df = df.drop(columns=['agePossession'])
    print(f"age_bucket distribution:\n{df['age_bucket'].value_counts().sort_index()}")
    return df


# ══════════════════════════════════════════
#  BLOCK 7 — ADDITIONAL ROOM FEATURES
# ══════════════════════════════════════════

def build_additional_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts binary flags from 'additionalRoom' column.

    'servant room, study room' → has_servant_room=1, has_study_room=1

    WHY binary flags instead of count?
    Each room type adds different value — servant room signals luxury,
    study room signals family/professional demographic.
    """
    if 'additionalRoom' not in df.columns:
        return df

    val_str = df['additionalRoom'].fillna('').str.lower()
    df['has_servant_room'] = val_str.str.contains('servant').astype(int)
    df['has_study_room']   = val_str.str.contains('study').astype(int)
    df['has_pooja_room']   = val_str.str.contains('pooja').astype(int)
    df['has_store_room']   = val_str.str.contains('store').astype(int)
    df = df.drop(columns=['additionalRoom'])
    return df


# ══════════════════════════════════════════
#  BLOCK 8 — DERIVED NUMERIC FEATURES
# ══════════════════════════════════════════

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates interaction features that are more meaningful than raw columns."""
    
    if df['bathroom'].dtype == object:
        df['bathroom'] = df['bathroom'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # 1. Relative floor
    df['relative_floor'] = np.where(
        df['total_floors'] > 0,
        df['floor_pos'] / df['total_floors'],
        0.0
    )
    # 🛠️ FIXED: Capping at 1.0 to handle dirty real estate data
    df['relative_floor'] = df['relative_floor'].clip(upper=1.0) 

    # 2. Bathroom per bedroom ratio
    df['bath_per_bed'] = np.where(
        df['bedRoom'] > 0,
        df['bathroom'] / df['bedRoom'],
        1.0
    )
    df['bath_per_bed'] = df['bath_per_bed'].clip(upper=3.0)  # cap outliers

    # 3. Log-transformed area
    df['log_area'] = np.log1p(df['area_sqft'])

    print(f"Derived features built: relative_floor, bath_per_bed, log_area")
    return df


# ══════════════════════════════════════════
#  BLOCK 9 — BALCONY ENCODING
# ══════════════════════════════════════════

def encode_balcony(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts '3+ Balconies', '2 Balconies', 'No Balcony' → integer.
    '3+' → 3, 'No Balcony' → 0
    """
    def parse_balcony(val):
        val = str(val).strip().lower()
        if 'no balcony' in val: return 0
        import re
        match = re.search(r'(\d+)', val)
        return int(match.group(1)) if match else 0

    if df['balcony'].dtype == object:
        df['balcony'] = df['balcony'].apply(parse_balcony)

    return df


# ══════════════════════════════════════════
#  BLOCK 10 — FACING ENCODING
# ══════════════════════════════════════════

def encode_facing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encoding for facing direction.

    In Indian real estate, East/North-East facing commands price premium
    (Vastu Shastra + natural light). South facing is least preferred.

    WHY ordinal over OHE?
    There IS a meaningful order in Indian market preference.
    Ordinal preserves this ranking in a single column.
    """
    FACING_MAP = {
        'east': 5, 'north-east': 5, 'north east': 5,
        'north': 4, 'north-west': 4, 'north west': 4,
        'west': 3,
        'south-east': 2, 'south east': 2,
        'south': 1, 'south-west': 1, 'south west': 1,
    }
    df['facing_score'] = (df['facing']
                          .fillna('unknown')
                          .str.lower()
                          .str.strip()
                          .map(FACING_MAP)
                          .fillna(0)  # unknown → 0 (neutral)
                         )
    df = df.drop(columns=['facing'])
    return df


# ══════════════════════════════════════════
#  BLOCK 11 — SECTOR EXTRACTION FROM ADDRESS
# ══════════════════════════════════════════

def extract_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts sector name from society/address columns for target encoding.

    Examples:
        'Sector 92 Gurgaon'    → 'sector_92'
        'DLF Phase 1'          → 'dlf_phase_1'
        'Sohna'                → 'sohna'

    WHY: society column has 500+ unique values (too high cardinality).
    sector has ~80 values — more stable for target encoding.
    """
    import re

    def parse_sector(val):
        if pd.isna(val):
            return 'unknown'
        val = str(val).lower().strip()
        # Match 'Sector XX' pattern
        match = re.search(r'sector[\s_-]?(\d+)', val)
        if match:
            return f"sector_{match.group(1)}"
        # Match DLF phases
        if 'dlf' in val:
            phase_match = re.search(r'phase[\s_-]?(\d+)', val)
            return f"dlf_phase_{phase_match.group(1)}" if phase_match else 'dlf_other'
        # Known localities
        for loc in ['sohna', 'palam vihar', 'golf course', 'cyber city',
                    'manesar', 'gwal pahari', 'rajendra park']:
            if loc in val:
                return loc.replace(' ', '_')
        return 'other'

    # Try society first, fallback to address
    if 'society' in df.columns:
        df['sector'] = df['society'].apply(parse_sector)
        df = df.drop(columns=['society'])
    return df


# ══════════════════════════════════════════
#  BLOCK 12 — TARGET ENCODING (K-FOLD SAFE)
# ══════════════════════════════════════════

def target_encode_sector(df: pd.DataFrame,
                          col: str = 'sector',
                          target: str = 'price',
                          n_splits: int = 5) -> pd.DataFrame:
    """
    K-Fold Target Encoding — prevents data leakage.

    NAIVE approach (WRONG):
        sector_mean = df.groupby('sector')['price'].mean()
        df['sector_encoded'] = df['sector'].map(sector_mean)
    Problem: Each row sees its own price when computing the mean.
    Model memorizes training data → overfitting.

    CORRECT approach (K-Fold Out-of-Fold):
        Split data into K folds.
        For each fold: compute mean from OTHER folds, apply to current fold.
        Each row's encoding is computed WITHOUT seeing its own target.

    Smoothing formula:
        encoded = (count × fold_mean + global_mean × SMOOTH) / (count + SMOOTH)
        WHY: Rare sectors (1-2 properties) get pulled toward global mean.
        Prevents extreme values from single data points.
    """
    SMOOTH = 10  # Smoothing factor — tune based on dataset size

    global_mean = df[target].mean()
    encoded_col = f"{col}_encoded"
    df[encoded_col] = global_mean  # Default: global mean

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Out-of-fold encoding
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]

        # Compute stats on training fold
        stats = (train_fold.groupby(col)[target]
                 .agg(['mean', 'count'])
                 .reset_index())
        stats.columns = [col, 'fold_mean', 'count']

        # Smoothing: blend fold_mean with global_mean
        stats[encoded_col] = (
            (stats['count'] * stats['fold_mean'] + global_mean * SMOOTH) /
            (stats['count'] + SMOOTH)
        )

        # Apply to validation fold only
        val_data = df.iloc[val_idx][[col]].merge(
            stats[[col, encoded_col]], on=col, how='left'
        ).fillna(global_mean)

        df.iloc[val_idx, df.columns.get_loc(encoded_col)] = val_data[encoded_col].values # type: ignore

    df = df.drop(columns=[col])
    print(f"Target encoded '{col}' → '{encoded_col}' using {n_splits}-Fold KFold")
    return df


# ══════════════════════════════════════════
#  BLOCK 13 — PROPERTY TYPE ENCODING
# ══════════════════════════════════════════

def encode_property_type(df: pd.DataFrame) -> pd.DataFrame:
    """Simple binary encoding: flat=0, house=1"""
    df['is_house'] = (df['property_type'] == 'house').astype(int)
    df = df.drop(columns=['property_type'])
    return df


# ══════════════════════════════════════════
#  MASTER PIPELINE
# ══════════════════════════════════════════

def run_feature_pipeline():
    """
    Orchestrates all feature engineering steps in dependency order.

    Input:  data/processed/gurgaon_properties_cleaned.parquet
    Output: data/features/train.parquet + feature_metadata.json
    """
    print("=" * 55)
    print("  Layer 4: Feature Engineering Pipeline")
    print("=" * 55)

    # Load cleaned data
    df = pd.read_parquet(PROCESSED_DIR / "gurgaon_properties_cleaned.parquet")
    print(f"\nLoaded cleaned data: {df.shape[0]} rows, {df.shape[1]} cols")

    # STEP 1: Drop leakage columns FIRST
    df = drop_leakage_columns(df)

    # STEP 2: Parse string-list columns → numeric features
    df = build_amenity_score(df)
    df = build_furnish_score(df)
    df = build_nearby_features(df)
    df = build_rating_score(df)

    # STEP 3: Encode categorical columns
    df = build_age_feature(df)
    df = build_additional_rooms(df)
    df = encode_balcony(df)
    df = encode_facing(df)
    df = encode_property_type(df)

    # STEP 4: Extract sector for target encoding
    df = extract_sector(df)

    # STEP 5: Build derived numeric features
    df = build_derived_features(df)

    # STEP 6: Target encode sector (K-Fold — must run AFTER all rows present)
    df = target_encode_sector(df, col='sector', target='price', n_splits=5)

    # STEP 7: Final cleanup — drop any remaining raw text columns
    text_cols = ['furnishDetails'] if 'furnishDetails' in df.columns else []
    if text_cols:
        df = df.drop(columns=text_cols)

    # STEP 8: Log-transform target (price is right-skewed)
    df['log_price'] = np.log1p(df['price'])

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DIR / "train.parquet"
    df.to_parquet(output_path, index=False)

    # Save feature metadata for API input validation
    import json
    feature_cols = [c for c in df.columns if c not in ['price', 'log_price']]
    metadata = {
        "feature_columns": feature_cols,
        "target": "log_price",
        "n_features": len(feature_cols),
        "n_rows": len(df),
        "note": "price is log1p transformed — use expm1 to inverse transform predictions"
    }
    with open(FEATURES_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Feature Engineering Complete!")
    print(f"  Output: {len(df)} rows x {len(feature_cols)} features")
    print(f"  Saved: {output_path}")
    print(f"{'='*55}")

    return df


if __name__ == "__main__":
    df = run_feature_pipeline()
    print("\nSample feature columns:")
    print(df.columns.tolist())
    print("\nStats:")
    print(df[['price', 'log_price', 'amenity_score',
              'furnish_score', 'avg_rating', 'relative_floor']].describe())