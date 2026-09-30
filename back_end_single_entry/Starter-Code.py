# import micropip
# await micropip.install("scikit-learn")
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. LOAD CSV DIRECTLY FROM JUPYTERLITE FILES
# =============================================================================
df = pd.read_csv("steam_games_2026.csv")
print(f"Dataset successfully loaded! Total games before DLC filter: {len(df)}")

# =============================================================================
# 2. PREPROCESSING & DYNAMIC COLUMN DETECTION
# =============================================================================
print("CSV Columns found:", df.columns.tolist())

# Lowercase dictionary mapping for fast detection
cols = {str(c).lower().strip().replace(" ", "_"): c for c in df.columns}

def get_column(keys, fallback_keyword=None, default=None):
    for k in keys:
        if k in cols:
            return cols[k]
    if fallback_keyword:
        for c in df.columns:
            if fallback_keyword in str(c).lower():
                return c
    return default

# Detect all required feature columns
title_col   = get_column(['title', 'name', 'game_title'], fallback_keyword='title', default=df.columns[0])
tag_col     = get_column(['tags', 'community_tags'], fallback_keyword='tag')
genre_col   = get_column(['primary_genre', 'genres', 'genre'], fallback_keyword='genre')
review_col  = get_column(['positive_review_percentage', 'review_score', 'rating'], fallback_keyword='review')
price_col   = get_column(['price', 'current_price'], fallback_keyword='price')
disc_col    = get_column(['discount_pct', 'discount', 'discount_percentage'], fallback_keyword='discount')
tot_rev_col = get_column(['total_reviews', 'review_count', 'reviews'], fallback_keyword='total')
owners_col  = get_column(['estimated_owners', 'owners'], fallback_keyword='owner')
peak_col    = get_column(['24h_peak_players', 'peak_concurrent_players', 'peak_players'], fallback_keyword='peak')

# -----------------------------------------------------------------------------
# NEW: DLC / EXPANSION FILTERING REGEX
# -----------------------------------------------------------------------------
dlc_keywords = [
    r'\bdlc\b', r'\bexpansion\b', r'\bseason pass\b', r'\badd-on\b', r'\baddon\b',
    r'\bcontent pack\b', r'\bsoundtrack\b', r'\bost\b', r'\bartbook\b',
    r'hearts of stone', r'blood and wine', r'shadow of erdtree', r'phantom liberty'
]
dlc_regex = re.compile("|".join(dlc_keywords), re.IGNORECASE)

# Filter out rows where title explicitly indicates a DLC/Expansion package
is_not_dlc = ~df[title_col].astype(str).str.contains(dlc_regex)
df = df[is_not_dlc].reset_index(drop=True)
print(f"Total games remaining after filtering explicit DLCs: {len(df)}")

print(f"Mapped Columns -> Title: '{title_col}' | Tags: '{tag_col}' | Primary Genre: '{genre_col}'")

def clean_text_vector(data):
    """Normalizes raw string or lists into space-separated string vectors."""
    if pd.isna(data):
        return ""
    cleaned = str(data).translate(str.maketrans('', '', "[]'\""))
    tags = [t.strip().replace(" ", "_").lower() for t in cleaned.split(',') if t.strip()]
    return " ".join(tags)

df['clean_tag_string'] = df[tag_col].apply(clean_text_vector) if tag_col else ""
df['clean_genre_string'] = df[genre_col].apply(clean_text_vector) if genre_col else ""

# Helper to clean numeric columns safely
def parse_numeric_series(column_name):
    if not column_name or column_name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column_name].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

s_rating   = parse_numeric_series(review_col)
s_price    = parse_numeric_series(price_col)
s_discount = parse_numeric_series(disc_col)
s_tot_rev  = parse_numeric_series(tot_rev_col)
s_owners   = parse_numeric_series(owners_col)
s_peak     = parse_numeric_series(peak_col)

# =============================================================================
# 3. MATHEMATICAL VECTORIZATION (TF-IDF MATRICES)
# =============================================================================
tfidf_tags = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_tags_matrix = tfidf_tags.fit_transform(df['clean_tag_string'])

tfidf_genre = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_genre_matrix = tfidf_genre.fit_transform(df['clean_genre_string'])

# =============================================================================
# 4. COSINE SIMILARITY MATRICES
# =============================================================================
tag_similarity_matrix = cosine_similarity(tfidf_tags_matrix, tfidf_tags_matrix)
genre_similarity_matrix = cosine_similarity(tfidf_genre_matrix, tfidf_genre_matrix)

# =============================================================================
# 5. FEATURE NORMALIZATION & LOG TRANSFORMS
# =============================================================================
scaler = MinMaxScaler()

norm_rating   = scaler.fit_transform(s_rating.values.reshape(-1, 1)).flatten()
norm_discount = scaler.fit_transform(s_discount.values.reshape(-1, 1)).flatten()

norm_price_raw = scaler.fit_transform(s_price.values.reshape(-1, 1)).flatten()
norm_price_inv = 1.0 - norm_price_raw

norm_tot_revs = scaler.fit_transform(np.log1p(s_tot_rev).values.reshape(-1, 1)).flatten()
norm_owners   = scaler.fit_transform(np.log1p(s_owners).values.reshape(-1, 1)).flatten()
norm_peak     = scaler.fit_transform(np.log1p(s_peak).values.reshape(-1, 1)).flatten()

# =============================================================================
# 6. MULTI-WEIGHTED RECOMMENDER SCORING ENGINE (WITH DEDUPLICATION)
# =============================================================================
def get_base_title(title):
    """Extracts root title before colon/hyphen/dash to deduplicate franchises."""
    raw = str(title).lower().strip()
    # Splitting on common subtitle separators
    base = re.split(r'[:\-—]', raw)[0].strip()
    return base

def recommend_games_vectorized(target_title, dataframe, top_n=10):
    match_mask = dataframe[title_col].astype(str).str.lower().str.contains(target_title.lower().strip())
    matched_indices = dataframe[match_mask].index

    if len(matched_indices) == 0:
        return None, f"❌ Game containing '{target_title}' was not found in the dataset."

    target_idx = matched_indices[0]
    target_row = dataframe.iloc[target_idx]
    target_base_name = get_base_title(target_row[title_col])

    tag_sims = tag_similarity_matrix[target_idx]
    genre_sims = genre_similarity_matrix[target_idx]

    effective_genre_sims = np.where(genre_sims == 0, tag_sims, genre_sims)

    # Weight Allocation
    w_tags     = 0.35
    w_genre    = 0.23
    w_rating   = 0.28
    w_discount = 0.02
    w_price    = 0.02
    w_tot_revs = 0.07
    w_owners   = 0.03
    w_peak     = 0.00

    # Linear Composite Multi-Objective Score Equation
    hybrid_scores = (
        (w_tags     * tag_sims) +
        (w_genre    * effective_genre_sims) +
        (w_rating   * norm_rating) +
        (w_discount * norm_discount) +
        (w_price    * norm_price_inv) +
        (w_tot_revs * norm_tot_revs) +
        (w_owners   * norm_owners) +
        (w_peak     * norm_peak)
    )

    # Exclude self-match
    hybrid_scores[target_idx] = -1.0

    # Retrieve all sorted candidate indices
    sorted_indices = np.argsort(hybrid_scores)[::-1]

    results = []
    seen_base_titles = {target_base_name}  # Seed set with target base title to prevent target franchise duplicates

    for idx in sorted_indices:
        title = dataframe.iloc[idx][title_col]
        base_title = get_base_title(title)

        # Skip if we already added a game variant from the same base root name
        if base_title in seen_base_titles:
            continue

        seen_base_titles.add(base_title)

        results.append({
            "title": title,
            "composite_score": round(float(hybrid_scores[idx]), 4),
            "tag_sim": round(float(tag_sims[idx]), 4),
            "genre_sim": round(float(effective_genre_sims[idx]), 4),
            "rating": s_rating.iloc[idx],
            "price": s_price.iloc[idx],
            "discount_pct": s_discount.iloc[idx],
            "peak_players": int(s_peak.iloc[idx])
        })

        if len(results) == top_n:
            break

    return target_row, results

# =============================================================================
# 7. EXECUTION
# =============================================================================
user_query = input("Enter a game title: ")
target_game, recommendations = recommend_games_vectorized(user_query, df, top_n=5)

if target_game is not None:
    print(f"\n🎯 TARGET GAME: {target_game[title_col]}")
    print(f"   Tags: {target_game.get(tag_col, 'N/A')}\n")
    print("=" * 110)
    print(f"{'RANK':<5} | {'GAME TITLE':<30} | {'SCORE':<8} | {'TAG SIM':<8} | {'GENRE SIM':<10} | {'RATING':<7} | {'PRICE':<7} | {'PEAK PLAYERS'}")
    print("=" * 110)

    for rank, rec in enumerate(recommendations, start=1):
        print(f"#{rank:<4} | {rec['title']:<30} | {rec['composite_score']:<8} | {rec['tag_sim']:<8} | {rec['genre_sim']:<10} | {rec['rating']}%    | ${rec['price']:<6} | {rec['peak_players']:,}")
else:
    print(recommendations)
