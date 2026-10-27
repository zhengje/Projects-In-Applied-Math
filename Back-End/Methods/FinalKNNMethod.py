import pandas as pd
import numpy as np
import re
import ast

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Global variables (used everywhere in the code)
title_col = "title"

tag_col = "tags"
genre_col = "genres"
category_col = "categories"
filtered_tag_col = "filtered_tags"

rating_col = "rating"
positive_col = "positive_ratio"
reviews_col = "user_reviews"
price_col = "price_final"

# =============================================================================
# 1. LOAD DATASET
# =============================================================================
def load_data():
    
    df = pd.read_csv("Data_Objective.csv")

    print(f"Games loaded: {len(df)}")
    print("CSV Columns found:", df.columns.tolist())


    # =============================================================================
    # 2. COLUMN MAPPING
    # =============================================================================



    # =============================================================================
    # 3. REMOVE DLC / EXPANSIONS
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

    dlc_regex = re.compile(
        "|".join(dlc_keywords),
        re.IGNORECASE
    )

    df = df[
        ~df[title_col].astype(str).str.contains(
            dlc_regex,
            na=False
        )
    ].reset_index(drop=True)

    print(f"After DLC filtering: {len(df)}")


    # =============================================================================
    # 4. MINIMUM 10,000 USER REVIEWS
    # =============================================================================
    # Games with fewer than 10,000 user reviews are excluded.
    # Filtering occurs before TF-IDF, normalization,
    # and KNN calculations.
    # =============================================================================

    print(f"Games before review filter: {len(df)}")

    df[reviews_col] = pd.to_numeric(
        df[reviews_col],
        errors="coerce"
    ).fillna(0)

    df = df[
        df[reviews_col] >= 10000
    ].reset_index(drop=True)

    print(f"Games after 10,000-review filtering: {len(df)}")

    return df

df = load_data()


# =============================================================================
# 5. CLEAN TEXT AND PARSE LISTS
# =============================================================================
#
# Converts list-like, comma-separated, or semicolon-separated
# attributes into individual normalized tokens.
#
# Example:
# ['Card Game', 'Roguelike Deckbuilder']
# becomes:
# ['card_game', 'roguelike_deckbuilder']
# =============================================================================

def parse_attributes(value):

    if isinstance(value, (list, tuple, set)):
        items = list(value)

    elif pd.isna(value) or str(value).strip() in ("", "[]"):
        return []

    else:

        raw = str(value).strip()

        try:

            parsed = ast.literal_eval(raw)

            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)
            else:
                items = re.split(r"[,;]", raw)

        except (SyntaxError, ValueError):
            items = re.split(r"[,;]", raw)

    normalized = []

    for item in items:

        token = re.sub(
            r"[^\w]+",
            "_",
            str(item).lower().strip()
        ).strip("_")

        if token and token not in normalized:
            normalized.append(token)

    return normalized


df["parsed_tags"] = df[tag_col].apply(
    parse_attributes
)

df["parsed_genres"] = df[genre_col].apply(
    parse_attributes
)

df["parsed_categories"] = df[category_col].apply(
    parse_attributes
)

df["parsed_filtered_tags"] = df[filtered_tag_col].apply(
    parse_attributes
)


# =============================================================================
# 6. OPTIONAL METADATA CORRECTIONS
# =============================================================================
#
# Some games may have incomplete tags in the dataset.
#
# Add selected gameplay tags in memory without modifying
# the original CSV.
#
# Only tags already present in the dataset's vocabulary
# are accepted.
#
# These additions are manually specified and should be
# checked for accuracy.
#
# Set USE_METADATA_CORRECTIONS = False to disable.
# =============================================================================

USE_METADATA_CORRECTIONS = True

metadata_corrections = {
    "inscryption": [
        "Card Game",
        "Deckbuilding",
        "Roguelike Deckbuilder",
        "Card Battler",
        "Strategy"
    ]
}

existing_vocabulary = set()

for column in ("parsed_tags", "parsed_filtered_tags"):

    for tags in df[column]:
        existing_vocabulary.update(tags)


if USE_METADATA_CORRECTIONS:

    for game_name, additions in metadata_corrections.items():

        matches = df.index[
            df[title_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(game_name.lower())
        ]

        for idx in matches:

            for col in ("parsed_tags", "parsed_filtered_tags"):

                tags = list(df.at[idx, col])

                for tag in additions:

                    token = parse_attributes([tag])[0]

                    if (
                        token in existing_vocabulary
                        and token not in tags
                    ):
                        tags.append(token)

                df.at[idx, col] = tags


# =============================================================================
# 7. GAMEPLAY TAG EMPHASIS AND TF-IDF FEATURES
# =============================================================================
#
# Generic tags contribute less than gameplay-specific tags.
#
# Generic tags: repeated once
# Gameplay tags: repeated three times
# Other tags: repeated twice
#
# This preserves TF-IDF vectorization while emphasizing
# meaningful gameplay attributes.
# =============================================================================

generic_tags = {
    "indie",
    "singleplayer",
    "multiplayer",
    "2d",
    "3d",
    "casual",
    "action",
    "adventure",
    "great_soundtrack",
    "atmospheric",
    "colorful",
    "stylized",
    "early_access",
    "controller"
}

gameplay_tags = {
    "card_game",
    "card_battler",
    "deckbuilding",
    "deckbuilder",
    "roguelike_deckbuilder",
    "roguelike",
    "roguelite",
    "turn_based",
    "turn_based_strategy",
    "turn_based_combat",
    "strategy",
    "tactical",
    "dungeon_crawler",
    "procedural_generation",
    "permadeath",
    "resource_management",
    "puzzle",
    "metroidvania",
    "survival",
    "crafting",
    "base_building",
    "automation",
    "tower_defense",
    "platformer",
    "bullet_hell",
    "stealth"
}


def weighted_text(tags):

    tokens = []

    for tag in tags:

        if tag in generic_tags:
            repetitions = 1

        elif tag in gameplay_tags:
            repetitions = 3

        else:
            repetitions = 2

        tokens.extend([tag] * repetitions)

    return " ".join(tokens)


def plain_text(tags):
    return " ".join(tags)


# -----------------------------------------------------------------------------
# Create separate TF-IDF matrices
# -----------------------------------------------------------------------------

text_specs = [
    ("parsed_tags", weighted_text),
    ("parsed_genres", plain_text),
    ("parsed_categories", plain_text),
    ("parsed_filtered_tags", weighted_text)
]

text_matrices = []

for col, transform in text_specs:

    strings = df[col].apply(transform)

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=col in (
            "parsed_tags",
            "parsed_filtered_tags"
        )
    )

    matrix = vectorizer.fit_transform(strings)

    text_matrices.append(matrix)


print("TF-IDF matrices successfully created.")


# =============================================================================
# 8. CLEAN AND NORMALIZE NUMERIC FEATURES
# =============================================================================
#
# Rating:
# Convert descriptive Steam ratings to a five-point scale.
#
# Positive ratio:
# Higher = better.
#
# User reviews:
# Higher = better, with log1p normalization.
#
# Price:
# Lower = better, using inverted Min-Max normalization.
# =============================================================================

rating_map = {
    "overwhelmingly negative": 1,
    "very negative": 1,
    "negative": 1,

    "mostly negative": 2,

    "mixed": 3,

    "mostly positive": 4,
    "positive": 4,

    "very positive": 5,
    "overwhelmingly positive": 5
}


s_rating = (
    df[rating_col]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .map(rating_map)
    .fillna(0)
    .astype(float)
)

s_positive = pd.to_numeric(
    df[positive_col],
    errors="coerce"
).fillna(0).clip(0, 100)

s_reviews = pd.to_numeric(
    df[reviews_col],
    errors="coerce"
).fillna(0).clip(lower=0)

s_price = pd.to_numeric(
    df[price_col],
    errors="coerce"
).fillna(0).clip(lower=0)


def normalize(series):

    values = np.asarray(
        series,
        dtype=float
    ).reshape(-1, 1)

    return MinMaxScaler().fit_transform(
        values
    ).ravel()


# Higher = better
norm_rating = normalize(
    s_rating / 5.0
)

# Higher = better
norm_positive = normalize(
    s_positive / 100.0
)

# Higher = better, log-transformed
norm_reviews = normalize(
    np.log1p(s_reviews)
)

# Lower = better
norm_price_inv = 1.0 - normalize(
    s_price
)


# =============================================================================
# 9. WEIGHTED FEATURE VECTOR FOR EUCLIDEAN KNN
# =============================================================================
#
# Same weights as the previous linear similarity model:
#
# Tags             = 0.27
# Genres           = 0.18
# Categories       = 0.05
# Filtered Tags    = 0.20
# Rating           = 0.08
# Positive Ratio   = 0.08
# User Reviews     = 0.13
# Low Price        = 0.01
#
# TOTAL            = 1.00
#
# KNN uses Euclidean distance rather than a linear
# similarity score.
#
# Multiply each feature block by sqrt(weight).
#
# This produces:
#
# D^2 = sum(weight * squared_feature_difference)
#
# Lower distance = stronger recommendation.
# =============================================================================

weights = [
    0.40,  # Tags
    0.18,  # Genres
    0.05,  # Categories
    0.12,  # Filtered Tags
    0.12,  # Rating
    0.07,  # Positive Ratio
    0.03,  # User Reviews
    0.03   # Low Price
]

assert np.isclose(
    sum(weights),
    1.0
)


# -----------------------------------------------------------------------------
# Apply weights to TF-IDF feature blocks
# -----------------------------------------------------------------------------

weighted_text_matrices = [
    matrix * np.sqrt(weight)
    for matrix, weight in zip(
        text_matrices,
        weights[:4]
    )
]


# -----------------------------------------------------------------------------
# Apply weights to normalized numeric features
# -----------------------------------------------------------------------------

numeric_features = np.column_stack([
    norm_rating,
    norm_positive,
    norm_reviews,
    norm_price_inv
])

numeric_features = (
    numeric_features
    * np.sqrt(np.array(weights[4:]))
)


# -----------------------------------------------------------------------------
# Combine all weighted features into one sparse matrix
# -----------------------------------------------------------------------------

game_features = hstack(
    weighted_text_matrices + [
        csr_matrix(numeric_features)
    ],
    format="csr"
)

print("Weighted feature matrix:", game_features.shape)


# =============================================================================
# 10. FIT EUCLIDEAN KNN
# =============================================================================
#
# KNN finds the games with the smallest Euclidean distance
# from the combined input-game feature profile.
# =============================================================================

knn_model = NearestNeighbors(
    metric="euclidean",
    algorithm="brute"
)

knn_model.fit(
    game_features
)

print("Euclidean KNN model successfully fitted.")


# =============================================================================
# 11. BASE TITLE / GAME LOOKUP
# =============================================================================

def get_base_title(title):

    return re.split(
        r"[:\-—]",
        str(title).lower().strip()
    )[0].strip()


def find_game(title):

    titles = (
        df[title_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    query = str(title).lower().strip()

    # Exact match first
    exact = df.index[
        titles.eq(query)
    ]

    if len(exact):
        return int(exact[0])

    # Partial match if no exact match
    partial = df.index[
        titles.str.contains(
            query,
            regex=False,
            na=False
        )
    ]

    if len(partial):
        return int(partial[0])

    return None


# =============================================================================
# 12. UNLIMITED-GAME KNN RECOMMENDER
# =============================================================================
#
# 1. Find all input games.
# 2. Retrieve their weighted feature vectors.
# 3. Average all vectors into one centroid.
# 4. Calculate Euclidean distance to all games.
# 5. Sort by smallest distance.
# 6. Remove input games and franchise duplicates.
# 7. Return the top 10 recommendations.
# =============================================================================

def recommend_games_knn(
    target_titles,
    dataframe,
    top_n=10
):

    if (
        len(target_titles) < 1
        or any(
            not str(x).strip()
            for x in target_titles
        )
    ):

        return None, (
            "At least one nonempty game title is required."
        )

    target_indices = []

    # -------------------------------------------------------------------------
    # Find selected games
    # -------------------------------------------------------------------------

    for title in target_titles:

        idx = find_game(title)

        if idx is None:

            return None, (
                f"Game '{title}' not found "
                f"(it may have fewer than 10,000 reviews)."
            )

        target_indices.append(idx)

    target_rows = [
        dataframe.iloc[idx]
        for idx in target_indices
    ]


    # =========================================================================
    # AVERAGE ALL FEATURE VECTORS
    # =========================================================================
    #
    # The centroid is:
    #
    # C = (V1 + V2 + ... + Vn) / n
    #
    # This generalizes the original three-game KNN method
    # to any number of selected games.
    # =========================================================================

    centroid = np.asarray(
        game_features[target_indices].mean(
            axis=0
        )
    ).ravel()


    # =========================================================================
    # FIND NEAREST NEIGHBORS
    # =========================================================================
    #
    # Retrieve all neighbors so removing duplicate franchises
    # does not reduce the final top-10 list.
    # =========================================================================

    distances, indices = knn_model.kneighbors(
        centroid.reshape(1, -1),
        n_neighbors=len(dataframe)
    )


    # =========================================================================
    # EXCLUDE INPUT GAMES AND DUPLICATE FRANCHISES
    # =========================================================================

    seen = {
        get_base_title(row[title_col])
        for row in target_rows
    }

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx in target_indices:
            continue

        title = dataframe.iloc[idx][title_col]

        base = get_base_title(title)

        if base in seen:
            continue

        seen.add(base)


        # ---------------------------------------------------------------------
        # Convert distance to a convenient display score
        #
        # KNN Score = 1 / (1 + distance)
        #
        # This does not change the KNN ranking.
        # ---------------------------------------------------------------------

        knn_score = (
            1.0 / (1.0 + float(distance))
        )


        # ---------------------------------------------------------------------
        # Store recommendation
        # ---------------------------------------------------------------------

        results.append({

            "title": title,

            "distance": float(distance),

            "knn_score": knn_score,

            "rating": dataframe.iloc[idx][rating_col],

            "positive_ratio": float(
                s_positive.iloc[idx]
            ),

            "user_reviews": int(
                s_reviews.iloc[idx]
            ),

            "price": float(
                s_price.iloc[idx]
            )
        })

        if len(results) == top_n:
            break

    return target_rows, results


# =============================================================================
# 13. USER INPUT: UNLIMITED GAMES
# =============================================================================

print(
    "\nEnter as many games as you want to create a combined "
    "KNN recommendation profile."
)

print("Press ENTER on a blank line when finished.\n")

target_titles = []

while True:

    title = input(
        f"Enter game {len(target_titles) + 1}: "
    ).strip()

    if not title:

        if target_titles:
            break

        print("Please enter at least one game.")
        continue

    if title.casefold() in {
        x.casefold() for x in target_titles
    }:

        print("Game already entered.")
        continue

    target_titles.append(title)


# =============================================================================
# 14. RUN RECOMMENDER
# =============================================================================

target_games, recommendations = recommend_games_knn(
    target_titles,
    df,
    top_n=10
)


# =============================================================================
# 15. DISPLAY RESULTS
# =============================================================================

if target_games is None:

    print(recommendations)

else:

    print("\n" + "=" * 145)

    print(
        "MULTI-GAME EUCLIDEAN KNN RECOMMENDATIONS"
    )

    print("=" * 145)

    print("\nSelected Games:")

    for i, game in enumerate(
        target_games,
        start=1
    ):

        print(
            f"{i}. {game[title_col]}"
        )

    print(
        "\nLower Euclidean distance = better match."
    )

    print(
        "Only games with at least 10,000 user reviews "
        "are included."
    )

    print("\n" + "=" * 145)

    print(
        f"{'RANK':<5} | "
        f"{'GAME TITLE':<32} | "
        f"{'DISTANCE':<9} | "
        f"{'KNN SCORE':<9} | "
        f"{'RATING':<23} | "
        f"{'POS %':<6} | "
        f"{'REVIEWS':<9} | "
        f"PRICE"
    )

    print("-" * 145)

    for rank, rec in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"#{rank:<4} | "
            f"{str(rec['title'])[:32]:<32} | "
            f"{rec['distance']:<9.4f} | "
            f"{rec['knn_score']:<9.4f} | "
            f"{str(rec['rating'])[:23]:<23} | "
            f"{rec['positive_ratio']:<6.1f} | "
            f"{rec['user_reviews']:<9,} | "
            f"${rec['price']:.2f}"
        )

    print("=" * 145)


# =============================================================================
# 16. MATHEMATICAL MODEL
# =============================================================================
#
# Weighted squared Euclidean distance:
#
# D^2 =
# 0.40*d_tags^2
# + 0.18*d_genres^2
# + 0.05*d_categories^2
# + 0.12*d_filtered_tags^2
# + 0.12*d_rating^2
# + 0.07*d_positive^2
# + 0.03*d_reviews^2
# + 0.03*d_low_price^2
#
# Smaller distance = better match.
# =============================================================================

# print("\nWeighted Squared Euclidean Distance:")

# print(
#     "D^2 = "
#     "0.40*d_tags^2 + "
#     "0.18*d_genres^2 + "
#     "0.05*d_categories^2 + "
#     "0.12*d_filtered_tags^2 + "
#     "0.12*d_rating^2 + "
#     "0.07*d_positive^2 + "
#     "0.03*d_reviews^2 + "
#     "0.03*d_low_price^2"
# )

# print(
#     "\nKNN Score = 1 / (1 + D)"
# )

# print(
#     "Lower Euclidean distance = stronger recommendation."
# )

def return_recs(user_game):
    target_game, recommendations = recommend_games_knn(user_game,df,top_n=10)

    if target_game is not None:
        
        front_array = []

        for rank, rec in enumerate(recommendations, start=1):
            # print(f"#{rank:<4} | {rec['title']:<30} | {rec['composite_score']:<8} | {rec['tag_sim']:<8} | {rec['genre_sim']:<10} | {rec['rating']}%    | ${rec['price']:<6} | {rec['peak_players']:,}")

            temp = {}

            temp["Rank"] = rank
            temp["Title"] = rec['title']
            # temp["Composite_Score"] = rec['composite_score']
            temp["tag_sim"] = rec['tag_sim']
            temp["genre_sim"] = rec['genre_sim']
            temp["rating"] = rec['rating']
            temp["price"] = rec['price']
            # dict_to_return['peak_players'] = rec['peak_players']
            front_array.append(temp)

        return front_array

    else:
        return front_array
         # print(recommendations)

# KNN_front = return_recs(["Omori","Slime Rancher","Terraria"])
