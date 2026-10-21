import pandas as pd
import numpy as np
import re
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# 1. LOAD DATASET
# =============================================================================

df = pd.read_csv("Data_FixedF.csv")

print(f"Dataset successfully loaded! Total games: {len(df)}")
print("CSV Columns found:", df.columns.tolist())


# =============================================================================
# 2. COLUMN MAPPING
# =============================================================================

title_col = "title"

category_col = "categories"
genre_col = "genres"
tag_col = "tags"
filtered_tag_col = "filtered_tags"

rating_col = "rating"
positive_ratio_col = "positive_ratio"
user_reviews_col = "user_reviews"
price_col = "price_final"


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

print(f"Games remaining after DLC filter: {len(df)}")


# =============================================================================
# 4. RATING CONVERSION
# =============================================================================
#
# Convert Steam's descriptive ratings to a five-point scale.
#
# Higher = better.
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


def convert_rating(rating):

    if pd.isna(rating):
        return 0.0

    return float(
        rating_map.get(
            str(rating).strip().lower(),
            0.0
        )
    )


df["rating_score"] = df[rating_col].apply(
    convert_rating
)



# =============================================================================
# 6. TEXT CLEANING
# =============================================================================
#
# Convert list-like and comma/semicolon-separated metadata
# into normalized individual attributes.
#
# Example:
#
# ['Card Game', 'Roguelike Deckbuilder']
#
# becomes:
#
# ['card_game', 'roguelike_deckbuilder']
#
# Multiword attributes remain single tokens.
# =============================================================================

def parse_attributes(data):

    if data is None:
        return []

    if isinstance(data, (list, tuple, set)):
        values = list(data)

    else:

        if pd.isna(data):
            return []

        raw = str(data).strip()

        if raw.lower() in ("", "[]", "nan", "none"):
            return []

        try:

            parsed = ast.literal_eval(raw)

            if isinstance(parsed, (list, tuple, set)):
                values = list(parsed)
            else:
                values = re.split(r"[,;]", raw)

        except (ValueError, SyntaxError):
            values = re.split(r"[,;]", raw)

    cleaned = []

    for value in values:

        value = str(value).strip().lower()

        value = re.sub(
            r"[^\w]+",
            "_",
            value
        ).strip("_")

        if value:
            cleaned.append(value)

    return list(dict.fromkeys(cleaned))


def make_text(attributes):
    return " ".join(attributes)


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
# 7. OPTIONAL VERIFIED METADATA CORRECTIONS
# =============================================================================
#
# Some games may have missing or incomplete tags.
#
# This section lets you add known, appropriate attributes
# WITHOUT editing the original CSV.
#
# Only attributes already present in the dataset's
# existing tag vocabulary are accepted.
#
# These corrections are illustrative and should be
# reviewed against the game's actual gameplay.
#
# To disable them, set USE_METADATA_CORRECTIONS = False.
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

existing_tag_vocabulary = set()

for attributes in df["parsed_tags"]:
    existing_tag_vocabulary.update(attributes)

for attributes in df["parsed_filtered_tags"]:
    existing_tag_vocabulary.update(attributes)


if USE_METADATA_CORRECTIONS:

    for game_title, additional_tags in metadata_corrections.items():

        match_mask = (
            df[title_col]
            .astype(str)
            .str.lower()
            .str.strip()
            .eq(game_title.lower())
        )

        for idx in df[match_mask].index:

            current_tags = list(
                df.at[idx, "parsed_tags"]
            )

            current_filtered = list(
                df.at[idx, "parsed_filtered_tags"]
            )

            for tag in additional_tags:

                normalized_tag = parse_attributes([tag])

                if not normalized_tag:
                    continue

                normalized_tag = normalized_tag[0]

                if normalized_tag not in existing_tag_vocabulary:
                    continue

                if normalized_tag not in current_tags:
                    current_tags.append(normalized_tag)

                if normalized_tag not in current_filtered:
                    current_filtered.append(normalized_tag)

            df.at[idx, "parsed_tags"] = current_tags
            df.at[idx, "parsed_filtered_tags"] = current_filtered


# =============================================================================
# 8. GAMEPLAY-SPECIFIC TAG WEIGHTING
# =============================================================================
#
# Generic tags such as Indie and Singleplayer provide
# relatively little information about gameplay.
#
# More specific tags such as Card Game, Deckbuilding,
# and Roguelike Deckbuilder are more informative.
#
# We apply a small additional boost to gameplay-related
# attributes before TF-IDF vectorization.
#
# TF-IDF still handles corpus-wide rarity automatically.
#
# The boost is implemented through repeated tokens,
# preserving the same TF-IDF/cosine similarity method.
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
    "controller",
    "steam_achievements",
    "steam_cloud",
    "partial_controller_support",
    "full_controller_support"
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
    "tactical_rpg",
    "dungeon_crawler",
    "procedural_generation",
    "perma_death",
    "permadeath",
    "resource_management",
    "puzzle",
    "puzzle_platformer",
    "metroidvania",
    "souls_like",
    "survival",
    "crafting",
    "base_building",
    "colony_sim",
    "city_builder",
    "automation",
    "management",
    "farming_sim",
    "life_sim",
    "tower_defense",
    "real_time_strategy",
    "grand_strategy",
    "simulation",
    "rhythm",
    "platformer",
    "precision_platformer",
    "bullet_hell",
    "hack_and_slash",
    "loot",
    "stealth",
    "immersive_sim",
    "first_person_shooter",
    "third_person_shooter",
    "fighting",
    "racing",
    "sports",
    "visual_novel"
}


def weighted_tag_text(attributes):

    tokens = []

    for tag in attributes:

        if tag in generic_tags:
            repetitions = 1

        elif tag in gameplay_tags:
            repetitions = 3

        else:
            repetitions = 2

        tokens.extend([tag] * repetitions)

    return " ".join(tokens)


df["clean_tag_string"] = df["parsed_tags"].apply(
    weighted_tag_text
)

df["clean_genre_string"] = df["parsed_genres"].apply(
    make_text
)

df["clean_category_string"] = df["parsed_categories"].apply(
    make_text
)

df["clean_filtered_tag_string"] = df["parsed_filtered_tags"].apply(
    weighted_tag_text
)


# =============================================================================
# 9. TF-IDF VECTORIZATION
# =============================================================================
#
# Four separate TF-IDF representations:
#
# 1. Tags
# 2. Genres
# 3. Categories
# 4. Filtered Tags
#
# We calculate cosine similarity only for selected games,
# avoiding enormous N x N matrices.
# =============================================================================

tfidf_tags = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True
)

tfidf_tags_matrix = tfidf_tags.fit_transform(
    df["clean_tag_string"]
)


tfidf_genre = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_genre_matrix = tfidf_genre.fit_transform(
    df["clean_genre_string"]
)


tfidf_category = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_category_matrix = tfidf_category.fit_transform(
    df["clean_category_string"]
)


tfidf_filtered_tags = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True
)

tfidf_filtered_tags_matrix = tfidf_filtered_tags.fit_transform(
    df["clean_filtered_tag_string"]
)

print("TF-IDF matrices successfully created.")


# =============================================================================
# 10. NUMERIC CLEANING AND NORMALIZATION
# =============================================================================
#
# Rating: higher = better
# Positive ratio: higher = better
# User reviews: higher = better, log-scaled
# Price: lower = better, inverted
# =============================================================================

def parse_numeric_series(column_name):

    return pd.to_numeric(
        df[column_name],
        errors="coerce"
    ).fillna(0.0)


s_rating = df["rating_score"].copy()

s_positive_ratio = parse_numeric_series(
    positive_ratio_col
).clip(lower=0, upper=100)

s_user_reviews = parse_numeric_series(
    user_reviews_col
).clip(lower=0)

s_price = parse_numeric_series(
    price_col
).clip(lower=0)


scaler_rating = MinMaxScaler()
scaler_positive = MinMaxScaler()
scaler_reviews = MinMaxScaler()
scaler_price = MinMaxScaler()


norm_rating = scaler_rating.fit_transform(
    (s_rating / 5.0).to_numpy().reshape(-1, 1)
).flatten()


norm_positive_ratio = scaler_positive.fit_transform(
    (s_positive_ratio / 100.0).to_numpy().reshape(-1, 1)
).flatten()


norm_user_reviews = scaler_reviews.fit_transform(
    np.log1p(
        s_user_reviews.to_numpy()
    ).reshape(-1, 1)
).flatten()


norm_price_raw = scaler_price.fit_transform(
    s_price.to_numpy().reshape(-1, 1)
).flatten()

norm_price_inv = 1.0 - norm_price_raw


# =============================================================================
# 11. BASE TITLE FUNCTION
# =============================================================================

def get_base_title(title):

    raw = str(title).lower().strip()

    return re.split(
        r"[:\-—]",
        raw
    )[0].strip()


# =============================================================================
# 12. MULTI-GAME RECOMMENDER
# =============================================================================

def recommend_games(
    target_titles,
    dataframe,
    top_n=20
):

    target_indices = []
    target_rows = []

    # -------------------------------------------------------------------------
    # Find input games
    # -------------------------------------------------------------------------

    for target_title in target_titles:

        target_title = target_title.lower().strip()

        exact_mask = (
            dataframe[title_col]
            .astype(str)
            .str.lower()
            .str.strip()
            .eq(target_title)
        )

        matched_indices = dataframe[exact_mask].index

        if len(matched_indices) == 0:

            partial_mask = (
                dataframe[title_col]
                .astype(str)
                .str.lower()
                .str.contains(
                    target_title,
                    regex=False,
                    na=False
                )
            )

            matched_indices = dataframe[partial_mask].index

        if len(matched_indices) == 0:

            return None, (
                f"Game containing '{target_title}' "
                f"was not found in the dataset."
            )

        target_idx = matched_indices[0]

        target_indices.append(target_idx)
        target_rows.append(dataframe.iloc[target_idx])

    # -------------------------------------------------------------------------
    # Cosine similarity
    # -------------------------------------------------------------------------

    tag_vectors = cosine_similarity(
        tfidf_tags_matrix[target_indices],
        tfidf_tags_matrix
    )

    genre_vectors = cosine_similarity(
        tfidf_genre_matrix[target_indices],
        tfidf_genre_matrix
    )

    category_vectors = cosine_similarity(
        tfidf_category_matrix[target_indices],
        tfidf_category_matrix
    )

    filtered_tag_vectors = cosine_similarity(
        tfidf_filtered_tags_matrix[target_indices],
        tfidf_filtered_tags_matrix
    )

    # -------------------------------------------------------------------------
    # Average similarities across all selected games
    # -------------------------------------------------------------------------

    combined_tag_similarity = np.mean(
        tag_vectors,
        axis=0
    )

    combined_genre_similarity = np.mean(
        genre_vectors,
        axis=0
    )

    combined_category_similarity = np.mean(
        category_vectors,
        axis=0
    )

    combined_filtered_tag_similarity = np.mean(
        filtered_tag_vectors,
        axis=0
    )

    # =========================================================================
    # 13. UPDATED LINEAR MODEL WEIGHTS
    # =========================================================================
    #
    # Similarity:
    #
    # Tags             = 0.40
    # Genres           = 0.18
    # Categories       = 0.05
    # Filtered Tags    = 0.12
    #
    # Total similarity = 0.75
    #
    # Other:
    #
    # Rating           = 0.12
    # Positive Ratio   = 0.07
    # User Reviews     = 0.03
    # Low Price        = 0.03
    #
    # TOTAL            = 1.00
    #
    # Tags receive the highest weight because they
    # contain more gameplay-specific information.
    #
    # Categories receive less weight because they
    # often describe Steam features rather than gameplay.
    # =========================================================================

    w_tags = 0.40
    w_genre = 0.18
    w_category = 0.05
    w_filtered_tags = 0.12

    w_rating = 0.12
    w_positive_ratio = 0.07
    w_user_reviews = 0.03
    w_price = 0.03

    # =========================================================================
    # 14. COMPOSITE LINEAR SCORE
    # =========================================================================

    hybrid_scores = (
        (w_tags * combined_tag_similarity)
        +
        (w_genre * combined_genre_similarity)
        +
        (w_category * combined_category_similarity)
        +
        (w_filtered_tags * combined_filtered_tag_similarity)
        +
        (w_rating * norm_rating)
        +
        (w_positive_ratio * norm_positive_ratio)
        +
        (w_user_reviews * norm_user_reviews)
        +
        (w_price * norm_price_inv)
    )

    # -------------------------------------------------------------------------
    # Exclude selected games
    # -------------------------------------------------------------------------

    for idx in target_indices:
        hybrid_scores[idx] = -1.0

    # -------------------------------------------------------------------------
    # Sort recommendations
    # -------------------------------------------------------------------------

    sorted_indices = np.argsort(
        hybrid_scores
    )[::-1]

    # -------------------------------------------------------------------------
    # Franchise deduplication
    # -------------------------------------------------------------------------

    seen_base_titles = set()

    for row in target_rows:

        seen_base_titles.add(
            get_base_title(row[title_col])
        )

    # -------------------------------------------------------------------------
    # Build recommendations
    # -------------------------------------------------------------------------

    results = []

    for idx in sorted_indices:

        title = dataframe.iloc[idx][title_col]

        base_title = get_base_title(title)

        if base_title in seen_base_titles:
            continue

        seen_base_titles.add(base_title)

        results.append({

            "title": title,

            "composite_score": round(
                float(hybrid_scores[idx]), 4
            ),

            "tag_sim": round(
                float(combined_tag_similarity[idx]), 4
            ),

            "genre_sim": round(
                float(combined_genre_similarity[idx]), 4
            ),

            "category_sim": round(
                float(combined_category_similarity[idx]), 4
            ),

            "filtered_tag_sim": round(
                float(combined_filtered_tag_similarity[idx]), 4
            ),

            "rating": dataframe.iloc[idx][rating_col],

            "positive_ratio": float(
                s_positive_ratio.iloc[idx]
            ),

            "user_reviews": int(
                s_user_reviews.iloc[idx]
            ),

            "price": float(
                s_price.iloc[idx]
            )
        })

        if len(results) == top_n:
            break

    return target_rows, results


# =============================================================================
# 15. USER INPUT AND EXECUTION
# =============================================================================

user_input = input(
    "\nEnter game titles separated by commas "
    "(e.g., Slay the Spire, Hades, Terraria): "
)

game_list = [
    game.strip()
    for game in user_input.split(",")
    if game.strip()
]

if len(game_list) == 0:

    print("\nPlease enter at least one game title.")

else:

    target_games, recommendations = recommend_games(
        game_list,
        df,
        top_n=20
    )

    # =========================================================================
    # 16. DISPLAY RESULTS
    # =========================================================================

    if target_games is not None:

        print("\n" + "=" * 170)

        print(
            f"MULTI-GAME RECOMMENDATION PROFILE "
            f"({len(game_list)} Selected)"
        )

        print("=" * 170)

        print("\nSelected Games:")

        for i, game in enumerate(
            target_games,
            start=1
        ):
            print(f"{i}. {game[title_col]}")

        print(
            "\nRecommendations are based on TF-IDF cosine similarity, "
            "gameplay-specific tag weighting, and normalized numerical features."
        )

        print(
            "Only games with at least 5,000 user reviews are included."
        )

        print("\n" + "=" * 170)

        print(
            f"{'RANK':<5} | "
            f"{'GAME TITLE':<30} | "
            f"{'SCORE':<7} | "
            f"{'TAGS':<6} | "
            f"{'GENRE':<6} | "
            f"{'CAT':<6} | "
            f"{'F.TAGS':<6} | "
            f"{'RATING':<23} | "
            f"{'POS %':<7} | "
            f"{'REVIEWS':<10} | "
            f"{'PRICE'}"
        )

        print("=" * 170)

        for rank, rec in enumerate(
            recommendations,
            start=1
        ):

            rating_text = str(rec["rating"])[:23]

            print(
                f"#{rank:<4} | "
                f"{str(rec['title'])[:30]:<30} | "
                f"{rec['composite_score']:<7.4f} | "
                f"{rec['tag_sim']:<6.4f} | "
                f"{rec['genre_sim']:<6.4f} | "
                f"{rec['category_sim']:<6.4f} | "
                f"{rec['filtered_tag_sim']:<6.4f} | "
                f"{rating_text:<23} | "
                f"{rec['positive_ratio']:>5.1f}% | "
                f"{rec['user_reviews']:<10,} | "
                f"${rec['price']:.2f}"
            )

        print("=" * 170)

        print("\nRecommendation Equation:")

        print(
            "Score = "
            "0.40(Tag Similarity) + "
            "0.18(Genre Similarity) + "
            "0.05(Category Similarity) + "
            "0.12(Filtered Tag Similarity) + "
            "0.12(Rating) + "
            "0.07(Positive Ratio) + "
            "0.03(User Reviews) + "
            "0.03(Low Price)"
        )

    else:

        print("\n" + recommendations)
