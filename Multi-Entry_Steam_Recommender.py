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

title_col   = get_column(
    ['title', 'name', 'game_title'],
    fallback_keyword='title',
    default=df.columns[0]
)

tag_col     = get_column(
    ['tags', 'community_tags'],
    fallback_keyword='tag'
)

genre_col   = get_column(
    ['primary_genre', 'genres', 'genre'],
    fallback_keyword='genre'
)

review_col  = get_column(
    ['positive_review_percentage', 'review_score', 'rating'],
    fallback_keyword='review'
)

price_col   = get_column(
    ['price', 'current_price'],
    fallback_keyword='price'
)

disc_col    = get_column(
    ['discount_pct', 'discount', 'discount_percentage'],
    fallback_keyword='discount'
)

tot_rev_col = get_column(
    ['total_reviews', 'review_count', 'reviews'],
    fallback_keyword='total'
)

owners_col  = get_column(
    ['estimated_owners', 'owners'],
    fallback_keyword='owner'
)

peak_col    = get_column(
    ['24h_peak_players', 'peak_concurrent_players', 'peak_players'],
    fallback_keyword='peak'
)

# =============================================================================
# 3. DLC / EXPANSION FILTERING
# =============================================================================
dlc_keywords = [
    r'\bdlc\b',
    r'\bexpansion\b',
    r'\bseason pass\b',
    r'\badd-on\b',
    r'\baddon\b',
    r'\bcontent pack\b',
    r'\bsoundtrack\b',
    r'\bost\b',
    r'\bartbook\b',
    r'hearts of stone',
    r'blood and wine',
    r'shadow of erdtree',
    r'phantom liberty'
]

dlc_regex = re.compile("|".join(dlc_keywords), re.IGNORECASE)

is_not_dlc = ~df[title_col].astype(str).str.contains(dlc_regex)

df = df[is_not_dlc].reset_index(drop=True)

print(f"Total games remaining after filtering explicit DLCs: {len(df)}")

print(
    f"Mapped Columns -> Title: '{title_col}' | "
    f"Tags: '{tag_col}' | "
    f"Primary Genre: '{genre_col}'"
)

# =============================================================================
# 4. TEXT CLEANING
# =============================================================================
def clean_text_vector(data):
    """Normalizes raw string or lists into space-separated string vectors."""
    if pd.isna(data):
        return ""

    cleaned = str(data).translate(
        str.maketrans('', '', "[]'\"")
    )

    tags = [
        t.strip().replace(" ", "_").lower()
        for t in cleaned.split(',')
        if t.strip()
    ]

    return " ".join(tags)


df['clean_tag_string'] = (
    df[tag_col].apply(clean_text_vector)
    if tag_col else ""
)

df['clean_genre_string'] = (
    df[genre_col].apply(clean_text_vector)
    if genre_col else ""
)

# =============================================================================
# 5. NUMERIC COLUMN CLEANING
# =============================================================================
def parse_numeric_series(column_name):

    if not column_name or column_name not in df.columns:
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(
        df[column_name]
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True),
        errors='coerce'
    ).fillna(0.0)


s_rating   = parse_numeric_series(review_col)
s_price    = parse_numeric_series(price_col)
s_discount = parse_numeric_series(disc_col)
s_tot_rev  = parse_numeric_series(tot_rev_col)
s_owners   = parse_numeric_series(owners_col)
s_peak     = parse_numeric_series(peak_col)

# =============================================================================
# 6. TF-IDF VECTORIZATION
# =============================================================================
tfidf_tags = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_tags_matrix = tfidf_tags.fit_transform(
    df['clean_tag_string']
)

tfidf_genre = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_genre_matrix = tfidf_genre.fit_transform(
    df['clean_genre_string']
)

# =============================================================================
# 7. COSINE SIMILARITY MATRICES
# =============================================================================
tag_similarity_matrix = cosine_similarity(
    tfidf_tags_matrix,
    tfidf_tags_matrix
)

genre_similarity_matrix = cosine_similarity(
    tfidf_genre_matrix,
    tfidf_genre_matrix
)

# =============================================================================
# 8. FEATURE NORMALIZATION & LOG TRANSFORMS
# =============================================================================
scaler = MinMaxScaler()

norm_rating = scaler.fit_transform(
    s_rating.values.reshape(-1, 1)
).flatten()

norm_discount = scaler.fit_transform(
    s_discount.values.reshape(-1, 1)
).flatten()

norm_price_raw = scaler.fit_transform(
    s_price.values.reshape(-1, 1)
).flatten()

norm_price_inv = 1.0 - norm_price_raw

norm_tot_revs = scaler.fit_transform(
    np.log1p(s_tot_rev).values.reshape(-1, 1)
).flatten()

norm_owners = scaler.fit_transform(
    np.log1p(s_owners).values.reshape(-1, 1)
).flatten()

norm_peak = scaler.fit_transform(
    np.log1p(s_peak).values.reshape(-1, 1)
).flatten()

# =============================================================================
# 9. BASE TITLE FUNCTION FOR DEDUPLICATION
# =============================================================================
def get_base_title(title):
    """
    Extracts root title before colon/hyphen/dash
    to prevent franchise duplicates.
    """
    raw = str(title).lower().strip()

    base = re.split(
        r'[:\-—]',
        raw
    )[0].strip()

    return base

# =============================================================================
# 10. THREE-GAME RECOMMENDER
# =============================================================================
def recommend_games_three_targets(
    target_titles,
    dataframe,
    top_n=10
):

    target_indices = []
    target_rows = []

    # -------------------------------------------------------------------------
    # Find all three games
    # -------------------------------------------------------------------------
    for target_title in target_titles:

        match_mask = dataframe[title_col].astype(str).str.lower().str.contains(
            target_title.lower().strip(),
            regex=False
        )

        matched_indices = dataframe[match_mask].index

        if len(matched_indices) == 0:
            return None, (
                f"❌ Game containing '{target_title}' "
                f"was not found in the dataset."
            )

        target_idx = matched_indices[0]

        target_indices.append(target_idx)
        target_rows.append(dataframe.iloc[target_idx])

    # -------------------------------------------------------------------------
    # Get similarity vectors for all three games
    # -------------------------------------------------------------------------
    tag_vectors = np.array([
        tag_similarity_matrix[idx]
        for idx in target_indices
    ])

    genre_vectors = np.array([
        genre_similarity_matrix[idx]
        for idx in target_indices
    ])

    # -------------------------------------------------------------------------
    # COMBINE THE THREE GAMES
    #
    # Each candidate receives the average similarity across the
    # three selected games.
    # -------------------------------------------------------------------------
    combined_tag_similarity = np.mean(
        tag_vectors,
        axis=0
    )

    combined_genre_similarity = np.mean(
        genre_vectors,
        axis=0
    )

    # If genre similarity is zero for a candidate, use tag similarity
    effective_genre_similarity = np.where(
        combined_genre_similarity == 0,
        combined_tag_similarity,
        combined_genre_similarity
    )

    # =============================================================================
    # SAME WEIGHTS AS ORIGINAL MODEL
    # =============================================================================
    w_tags     = 0.35
    w_genre    = 0.20
    w_rating   = 0.30
    w_discount = 0.02
    w_price    = 0.02
    w_tot_revs = 0.07
    w_owners   = 0.03
    w_peak     = 0.00

    # =============================================================================
    # COMPOSITE SCORE
    # =============================================================================
    hybrid_scores = (
        (w_tags * combined_tag_similarity) +
        (w_genre * effective_genre_similarity) +
        (w_rating * norm_rating) +
        (w_discount * norm_discount) +
        (w_price * norm_price_inv) +
        (w_tot_revs * norm_tot_revs) +
        (w_owners * norm_owners) +
        (w_peak * norm_peak)
    )

    # -------------------------------------------------------------------------
    # Exclude all three input games
    # -------------------------------------------------------------------------
    for idx in target_indices:
        hybrid_scores[idx] = -1.0

    # -------------------------------------------------------------------------
    # Sort candidates from highest to lowest
    # -------------------------------------------------------------------------
    sorted_indices = np.argsort(
        hybrid_scores
    )[::-1]

    # -------------------------------------------------------------------------
    # Prevent duplicate franchises
    # -------------------------------------------------------------------------
    seen_base_titles = set()

    for row in target_rows:
        seen_base_titles.add(
            get_base_title(row[title_col])
        )

    results = []

    # -------------------------------------------------------------------------
    # Retrieve Top N
    # -------------------------------------------------------------------------
    for idx in sorted_indices:

        title = dataframe.iloc[idx][title_col]

        base_title = get_base_title(title)

        # Skip franchise duplicates
        if base_title in seen_base_titles:
            continue

        seen_base_titles.add(base_title)

        results.append({
            "title": title,
            "composite_score": round(
                float(hybrid_scores[idx]),
                4
            ),
            "tag_sim": round(
                float(combined_tag_similarity[idx]),
                4
            ),
            "genre_sim": round(
                float(effective_genre_similarity[idx]),
                4
            ),
            "rating": s_rating.iloc[idx],
            "price": s_price.iloc[idx],
            "discount_pct": s_discount.iloc[idx],
            "peak_players": int(s_peak.iloc[idx])
        })

        if len(results) == top_n:
            break

    return target_rows, results

# =============================================================================
# 11. EXECUTION
# =============================================================================

print("\nEnter THREE games to build a combined recommendation profile.")

game_1 = input("Enter game 1: ")
game_2 = input("Enter game 2: ")
game_3 = input("Enter game 3: ")

target_games, recommendations = recommend_games_three_targets(
    [game_1, game_2, game_3],
    df,
    top_n=10
)

# =============================================================================
# 12. DISPLAY RESULTS
# =============================================================================
if target_games is not None:

    print("\n" + "=" * 110)
    print("🎮 THREE-GAME RECOMMENDATION PROFILE")
    print("=" * 110)

    for i, game in enumerate(target_games, start=1):
        print(f"{i}. {game[title_col]}")

    print("\n")
    print("The recommendations are based on the combined mathematical")
    print("similarity profile of all three selected games.")

    print("\n" + "=" * 110)

    print(
        f"{'RANK':<5} | "
        f"{'GAME TITLE':<30} | "
        f"{'SCORE':<8} | "
        f"{'TAG SIM':<8} | "
        f"{'GENRE SIM':<10} | "
        f"{'RATING':<7} | "
        f"{'PRICE':<7} | "
        f"{'PEAK PLAYERS'}"
    )

    print("=" * 110)

    for rank, rec in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"#{rank:<4} | "
            f"{rec['title']:<30} | "
            f"{rec['composite_score']:<8} | "
            f"{rec['tag_sim']:<8} | "
            f"{rec['genre_sim']:<10} | "
            f"{rec['rating']}%    | "
            f"${rec['price']:<6} | "
            f"{rec['peak_players']:,}"
        )

else:

    print(recommendations)
