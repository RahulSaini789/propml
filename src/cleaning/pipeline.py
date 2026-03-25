import pandas as pd 
# import numpy as np
# from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")

# # --- Directory Setup ---
# BASE_DIR = Path(".")
# RAW_DIR = BASE_DIR / "data" / "raw"
# PROCESSED_DIR = BASE_DIR / "data" / "processed"

# def load_data(filepath: Path) -> pd.DataFrame:
#     """Reads the raw CSV file into a pandas DataFrame."""
#     df = pd.read_csv(filepath)
#     print(f"Loaded {filepath.name}: {df.shape[0]} rows, {df.shape[1]} cols")
#     return df

# def clean_houses_area(df: pd.DataFrame) -> pd.DataFrame:
#     df['area'] = df['area'].str.extract(r'(\d+\.?\d*)').astype(float)
#     df['area'] = df['area'] * 10.7639
#     df = df.rename(columns={'area': 'area_sqft'})
#     return df

# def clean_houses_price(df: pd.DataFrame) -> pd.DataFrame:
#     df['price'] = df['price'].str.extract(r'(\d+\.?\d*)').astype(float)
#     return df

# def clean_houses_floors(df: pd.DataFrame) -> pd.DataFrame:
#     df['total_floors'] = df['noOfFloor'].str.extract(r'(\d+)').astype(float)
#     df['floor_pos'] = 1.0
#     df = df.drop(columns=['noOfFloor'])
#     return df

# def clean_flats_area(df: pd.DataFrame) -> pd.DataFrame:
#     """Calculates actual area from price and rate/sq.ft."""
#     df['rate_per_sqft'] = df['area'].str.extract(r'(\d+,?\d*)').replace(',', '', regex=True).astype(float)
    
#     # Bug Fix: Handle zero or tiny rates to prevent infinity/explosion
#     df['rate_per_sqft'] = df['rate_per_sqft'].replace(0, np.nan)
    
#     df['area_sqft'] = (df['price'] * 10000000) / df['rate_per_sqft']
#     df = df.drop(columns=['area', 'rate_per_sqft'])
#     return df

# def clean_flats_price(df: pd.DataFrame) -> pd.DataFrame:
#     raw_numbers = df['price'].str.extract(r'(\d+\.?\d*)').astype(float).iloc[:, 0]
#     df['price'] = np.where(df['price'].str.contains('Lac', case=False, na=False),
#                            raw_numbers * 0.01, raw_numbers)
#     return df

# def clean_floors(df: pd.DataFrame) -> pd.DataFrame:
#     df['floorNum'] = df['floorNum'].str.replace(r'\xa0', ' ', regex=True).str.strip()
#     floors = df['floorNum'].str.extract(r'(\d+).*?(\d+)')
#     df['floor_pos'] = floors[0].astype(float)
#     df['total_floors'] = floors[1].astype(float)
#     df = df.drop(columns=['floorNum'])
#     return df

# def clean_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
#     if 'bedRoom' in df.columns:
#         df['bedRoom'] = df['bedRoom'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
#     return df

# def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
#     """Smart Outlier Removal: Grouped by property_type to save our houses!"""
#     df = df.dropna(subset=['price', 'area_sqft'])

#     def filter_iqr(sub_df):
#         # 1. Price IQR
#         Q1_p = sub_df['price'].quantile(0.25)
#         Q3_p = sub_df['price'].quantile(0.75)
#         IQR_p = Q3_p - Q1_p
#         sub_df = sub_df[(sub_df['price'] >= Q1_p - 1.5*IQR_p) & (sub_df['price'] <= Q3_p + 1.5*IQR_p)]
        
#         # 2. Area IQR (Yeh 8.7 Million sq.ft. wale monster ko maarega!)
#         Q1_a = sub_df['area_sqft'].quantile(0.25)
#         Q3_a = sub_df['area_sqft'].quantile(0.75)
#         IQR_a = Q3_a - Q1_a
#         sub_df = sub_df[(sub_df['area_sqft'] >= Q1_a - 1.5*IQR_a) & (sub_df['area_sqft'] <= Q3_a + 1.5*IQR_a)]
        
#         return sub_df

#     # Apply grouped filter!
#     df = df.groupby('property_type', group_keys=False).apply(filter_iqr)
    
#     # Cap bedrooms
#     if 'bedRoom' in df.columns:
#         df = df[df['bedRoom'] <= 10]
        
#     print(f"Outliers intelligently removed. Surviving rows: {len(df)}")
#     return df

# def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.dropna(subset=['price'])
#     median_area = df['area_sqft'].median()
#     df['area_sqft'] = df['area_sqft'].fillna(median_area)
#     df['floor_pos'] = df['floor_pos'].fillna(1.0)
#     df['total_floors'] = df['total_floors'].fillna(1.0)
#     return df

# def run_pipeline():
#     # --- PHASE 1: LOAD MATERIALS ---
#     houses = load_data(RAW_DIR / "houses.csv")
#     flats = load_data(RAW_DIR / "flats.csv")

#     # --- PHASE 2: FLATS CLEANING ---
#     flats = clean_flats_price(flats)   
#     flats = clean_flats_area(flats)
#     flats = clean_floors(flats)
#     flats = clean_bedrooms(flats)      
#     flats['property_type'] = 'flat'

#     # --- PHASE 3: HOUSES CLEANING ---
#     houses = clean_houses_price(houses)
#     houses = clean_houses_area(houses)
#     houses = clean_houses_floors(houses)
#     houses = clean_bedrooms(houses)    
#     houses['property_type'] = 'house'

#     # --- PHASE 4: MERGE & SMART CLEAN ---
#     master_df = pd.concat([houses, flats], ignore_index=True)
#     master_df = handle_missing_values(master_df) # Handle missing before IQR
#     master_df = remove_outliers(master_df) # Apply SMART IQR

#     # --- PHASE 5: EXPORT ---
#     PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#     master_df.to_parquet(PROCESSED_DIR / "gurgaon_properties_cleaned.parquet", index=False)
#     print("✅ Pipeline Success! MLOps standards met.")

# if __name__ == "__main__":
#     run_pipeline()

df = pd.read_parquet("data/processed/gurgaon_properties_cleaned.parquet")
print(df.shape)
print(df['property_type'].value_counts())
print(df[['price', 'area_sqft', 'bedRoom']].describe())