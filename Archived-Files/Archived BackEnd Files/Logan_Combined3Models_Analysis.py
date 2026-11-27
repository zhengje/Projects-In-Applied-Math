import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Use the script folder so the model files and CSV paths stay consistent.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import FinalKNNMethod as knn_script
import FinalSimilarityMatrixMethod as sim_script
import FinalIMFMethod as imf_script

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================
dataset = pd.read_csv(os.path.join(BASE_DIR, "Data_Objective.csv"))
print(f"Fixed metadata dataset loaded: {len(dataset)} games")

subjective_df = pd.read_csv(os.path.join(BASE_DIR, "Data_Subjective.csv"))
print(f"Subjective interaction dataset loaded: {len(subjective_df)} records")

CANDIDATES_PER_MODEL = 100
FINAL_TOP_N = 10
N_RECS = CANDIDATES_PER_MODEL
print(
    "\nEnter as many games as you want to create a combined "
    "hybrid recommendation profile."
)
print("Press ENTER on a blank line when finished.\n")

input_games = []

while True:
    title = input(
        f"Enter game {len(input_games) + 1}: "
    ).strip()

    if not title:
        if input_games:
            break
        print("Please enter at least one game.")
        continue

    if title.casefold() in {
        x.casefold() for x in input_games
    }:
        print("Game already entered.")
        continue

    input_games.append(title)

# =============================================================================
# 2. GET THE THREE RECOMMENDATION LISTS
# =============================================================================
# MODEL 1: KNN Content Model
if hasattr(knn_script, "recommend_games_knn"):
    knn_fn = knn_script.recommend_games_knn
    knn_data = getattr(knn_script, "df", dataset)
elif hasattr(knn_script, "build_knn"):
    knn_fn, knn_data = knn_script.build_knn()
else:
    raise AttributeError("Could not find 'recommend_games_knn' or 'build_knn' in FinalKNNMethod.py")

_, knn_results = knn_fn(input_games, knn_data, top_n=N_RECS)
if not isinstance(knn_results, list):
    raise RuntimeError(f"KNN Error: {knn_results}")

# MODEL 2: Cosine Similarity Matrix
if hasattr(sim_script, "recommend_games"):
    sim_fn = sim_script.recommend_games
    sim_data = getattr(sim_script, "df", dataset)
elif hasattr(sim_script, "build_similarity"):
    sim_fn, sim_data = sim_script.build_similarity()
else:
    raise AttributeError("Could not find 'recommend_games' or 'build_similarity' in FinalSimilarityMatrixMethod.py")

_, sim_results = sim_fn(input_games, sim_data, top_n=N_RECS)
if not isinstance(sim_results, list):
    raise RuntimeError(f"Similarity Error: {sim_results}")

# MODEL 3: Implicit Matrix Factorization
pkl_path = os.path.join(BASE_DIR, "implicit_mf_model.pkl")
if os.path.exists(pkl_path):
    print(f"Loading pre-trained ALS model from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    app_index_map = artifacts["app_index_map"]
    app_id_to_index_map = artifacts["app_id_to_index_map"]
    game_title = artifacts["game_title"]
    title_to_app_id_map = artifacts["title_to_app_id_map"]
else:
    print("Training Implicit ALS model via FinalIMFMethod...")
    model, _, _, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = imf_script.implicit_mf()
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": model,
            "app_index_map": app_index_map,
            "app_id_to_index_map": app_id_to_index_map,
            "game_title": game_title,
            "title_to_app_id_map": title_to_app_id_map,
        }, f)
    print(f"Saved trained ALS model to {pkl_path}")

valid_app_ids = []
for name in input_games:
    search_name = name.strip().lower()
    app_id = title_to_app_id_map.get(search_name)

    if app_id is None or app_id not in app_id_to_index_map:
        for cand_title, cand_id in title_to_app_id_map.items():
            if search_name in cand_title and cand_id in app_id_to_index_map:
                app_id = cand_id
                break

    if app_id is not None and app_id in app_id_to_index_map and app_id not in valid_app_ids:
        valid_app_ids.append(app_id)

if not valid_app_ids:
    raise RuntimeError(f"None of {input_games} matched in the Implicit ALS interaction matrix.")

id_indices = [app_id_to_index_map[aid] for aid in valid_app_ids]
user_items = csr_matrix(
    (np.ones(len(id_indices), dtype=np.float32), ([0] * len(id_indices), id_indices)),
    shape=(1, model.item_factors.shape[0])
)

try:
    ids, scores = model.recommend(
        userid=0,
        user_items=user_items,
        N=N_RECS,
        filter_already_liked_items=True,
        recalculate_user=True
    )
except TypeError:
    ids, scores = model.recommend(
        userid=0,
        user_items=user_items,
        N=N_RECS,
        filter_already_liked_items=True
    )

if isinstance(ids, np.ndarray) and ids.ndim > 1:
    ids = ids[0]

implicit_results = []
for idx in ids:
    rec_app_id = app_index_map[int(idx)]
    rec_title = game_title.get(rec_app_id)
    if rec_title:
        implicit_results.append({"title": rec_title})

print("\nSuccessfully gathered list arrays:")
print(f" -> KNN Recs Count: {len(knn_results)}")
print(f" -> Similarity Recs Count: {len(sim_results)}")
print(f" -> Implicit ALS Recs Count: {len(implicit_results)}")

# =============================================================================
# 3. COMBINE THE LISTS USING WEIGHTED RRF
# =============================================================================
def game_key(title):
    return str(title).strip().casefold()

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

def combine_recommendations(model_results, top_n=10):
    # Standard RRF smoothing constant; keep fixed for a fair baseline.
    RRF_K = 60
    names = list(model_results)
    rank_maps = {}
    display_titles = {}

    for name, recommendations in model_results.items():
        ranks = {}
        for position, rec in enumerate(recommendations, start=1):
            key = game_key(rec["title"])
            if key not in ranks:
                ranks[key] = position
                display_titles.setdefault(key, rec["title"])
        rank_maps[name] = ranks

    overlap = pd.DataFrame(
        [[jaccard(rank_maps[a], rank_maps[b]) for b in names] for a in names],
        index=names,
        columns=names,
    )

    # Reward models that bring different games to the shortlist.
    # This is a diversity heuristic, NOT an accuracy-based weight.
    raw_weights = {
        name: 1 / (1 + sum(overlap.loc[name, other] for other in names if other != name))
        for name in names
    }
    total_weight = sum(raw_weights.values())
    weights = {name: raw_weights[name] / total_weight for name in names}

    rows = []
    for key, title in display_titles.items():
        score = sum(
            weights[name] / (RRF_K + rank_maps[name][key])
            for name in names
            if key in rank_maps[name]
        )
        row = {
            "Game": title,
            "Fusion score": score,
            "Models agreeing": sum(key in rank_maps[n] for n in names),
        }
        for name in names:
            row[f"{name} rank"] = rank_maps[name].get(key, None)
        rows.append(row)

    ranking = (
        pd.DataFrame(rows)
        .sort_values(
            ["Fusion score", "Models agreeing", "Game"],
            ascending=[False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "Rank"

    return ranking, overlap, weights

# =============================================================================
# 4. BUILD THE FINAL HYBRID RANKING
# =============================================================================
model_results = {
    "KNN": knn_results,
    "Similarity": sim_results,
    "Implicit ALS": implicit_results,
}

ranking, overlap, weights = combine_recommendations(model_results, top_n=FINAL_TOP_N)

print("\n3x3 JACCARD OVERLAP MATRIX")
print(overlap.round(4).to_string())
print("\nMODEL WEIGHTS:", {k: round(v, 4) for k, v in weights.items()})
print("\nFINAL HYBRID TOP 10")
print(ranking.to_string(float_format=lambda x: f"{x:.6f}"))


# =============================================================================
#             HOW THE THREE METHODS DIFFER
# =============================================================================

from itertools import combinations
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

# Store the three recommendation lists
methods = {
    "KNN": knn_results,
    "Cosine Similarity": sim_results,
    "Implicit ALS": implicit_results
}


# -----------------------------------------------------------------------------
# 1. PREPARE RECOMMENDATION LISTS
# -----------------------------------------------------------------------------

def extract_titles(results):
    """
    Extract game titles from recommendation results.
    Supports lists, tuples, Series, NumPy arrays and DataFrames.
    """

    if isinstance(results, pd.DataFrame):
        possible_columns = ["Game", "game", "title", "Title", "name"]

        title_column = next(
            (col for col in possible_columns if col in results.columns),
            None
        )

        if title_column is None:
            raise ValueError(
                "Could not identify the game-title column in recommendation results."
            )

        return results[title_column].dropna().astype(str).tolist()

    if isinstance(results, pd.Series):
        return results.dropna().astype(str).tolist()

    if isinstance(results, np.ndarray):
        results = results.tolist()

    titles = []

    for item in results:
        if isinstance(item, str):
            titles.append(item)

        elif isinstance(item, dict):
            title = (
                item.get("Game")
                or item.get("game")
                or item.get("title")
                or item.get("Title")
            )

            if title is not None:
                titles.append(str(title))

        elif isinstance(item, (tuple, list, np.ndarray)):
            if len(item) > 0:
                titles.append(str(item[0]))

    return titles


# Build normalized recommendation lists and rank dictionaries
recommendation_lists = {}
rank_maps = {}

for method, results in methods.items():

    titles = extract_titles(results)

    recommendation_lists[method] = titles

    rank_maps[method] = {}

    for rank, title in enumerate(titles, start=1):

        key = title.strip().casefold()

        if key not in rank_maps[method]:
            rank_maps[method][key] = rank


recommendation_sets = {
    method: set(ranks.keys())
    for method, ranks in rank_maps.items()
}

all_games = set.union(*recommendation_sets.values())

display_titles = {}

for method, titles in recommendation_lists.items():
    for title in titles:
        key = title.strip().casefold()

        if key not in display_titles:
            display_titles[key] = title


# -----------------------------------------------------------------------------
# 2. RECOMMENDATION OVERLAP
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("1. RECOMMENDATION OVERLAP")
print("=" * 85)

print("\nPAIRWISE COMPARISON")

overlap_rows = []

pairwise_jaccard = []

for method1, method2 in combinations(methods.keys(), 2):

    set1 = recommendation_sets[method1]
    set2 = recommendation_sets[method2]

    shared = set1.intersection(set2)
    union = set1.union(set2)

    jaccard_score = len(shared) / len(union) if union else 0

    overlap_percentage = (
        len(shared) / min(len(set1), len(set2)) * 100
        if min(len(set1), len(set2)) > 0 else 0
    )

    pairwise_jaccard.append(jaccard_score)

    overlap_rows.append({
        "Method Comparison": f"{method1} vs {method2}",
        "Shared Games": len(shared),
        "Jaccard Similarity": round(jaccard_score, 4),
        "Overlap Percentage": round(overlap_percentage, 2)
    })

overlap_df = pd.DataFrame(overlap_rows)

print(overlap_df.to_string(index=False))

average_jaccard = np.mean(pairwise_jaccard)

print(f"\nAverage pairwise Jaccard similarity: {average_jaccard:.4f}")
print(f"Total distinct games recommended: {len(all_games)}")


# -----------------------------------------------------------------------------
# 3. RANKING AGREEMENT
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("2. RANKING AGREEMENT")
print("=" * 85)

print(
    "\nSpearman correlation measures whether two methods rank "
    "their shared recommendations similarly."
)

correlation_rows = []

for method1, method2 in combinations(methods.keys(), 2):

    shared = (
        recommendation_sets[method1]
        .intersection(recommendation_sets[method2])
    )

    ranks1 = [
        rank_maps[method1][game]
        for game in shared
    ]

    ranks2 = [
        rank_maps[method2][game]
        for game in shared
    ]

    if len(shared) >= 2:
        correlation = spearmanr(ranks1, ranks2).statistic
    else:
        correlation = np.nan

    correlation_rows.append({
        "Method Comparison": f"{method1} vs {method2}",
        "Shared Games Analysed": len(shared),
        "Spearman Correlation": (
            round(correlation, 4)
            if not np.isnan(correlation) else "N/A"
        )
    })

correlation_df = pd.DataFrame(correlation_rows)

print("\nRANKING CORRELATION RESULTS")
print(correlation_df.to_string(index=False))


# -----------------------------------------------------------------------------
# 4. RECOMMENDATION DIVERSITY AND AGREEMENT
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("3. RECOMMENDATION DIVERSITY")
print("=" * 85)

diversity_rows = []

for method in methods:

    own_games = recommendation_sets[method]

    other_games = set.union(*[
        recommendation_sets[other]
        for other in methods
        if other != method
    ])

    exclusive_games = own_games - other_games

    shared_games = own_games.intersection(other_games)

    diversity_rows.append({
        "Method": method,
        "Total Recommendations": len(own_games),
        "Exclusive Recommendations": len(exclusive_games),
        "Recommendations Shared": len(shared_games),
        "Exclusive Percentage": round(
            len(exclusive_games) / len(own_games) * 100, 2
        ) if own_games else 0
    })

diversity_df = pd.DataFrame(diversity_rows)

print("\nINDIVIDUAL METHOD CONTRIBUTIONS")
print(diversity_df.to_string(index=False))


# Count how many methods recommended each distinct game
agreement_counts = {}

for game in all_games:

    count = sum(
        game in recommendation_sets[method]
        for method in methods
    )

    agreement_counts[game] = count

all_three = sum(count == 3 for count in agreement_counts.values())
exactly_two = sum(count == 2 for count in agreement_counts.values())
only_one = sum(count == 1 for count in agreement_counts.values())

print("\nAGREEMENT ACROSS ALL THREE METHODS")
print(f"Recommended by all three methods: {all_three}")
print(f"Recommended by exactly two methods: {exactly_two}")
print(f"Recommended by only one method: {only_one}")
print(f"Total distinct recommendations: {len(all_games)}")

if all_games:
    print(
        "Percentage recommended by all three: "
        f"{all_three / len(all_games) * 100:.2f}%"
    )


# -----------------------------------------------------------------------------
# 5. HYBRID MODEL WEIGHTS
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("4. HYBRID MODEL CONTRIBUTION")
print("=" * 85)

# The weights are generated by the Jaccard-weighted fusion procedure.
# Similarity is the key used in the original weight dictionary.

weight_mapping = {
    "KNN": "KNN",
    "Cosine Similarity": "Similarity",
    "Implicit ALS": "Implicit ALS"
}

hybrid_weight_rows = []

if "weights" in globals():

    for method, weight_key in weight_mapping.items():

        weight = weights.get(weight_key)

        if weight is not None:

            hybrid_weight_rows.append({
                "Method": method,
                "Hybrid Weight": round(float(weight), 4),
                "Percentage Influence": round(float(weight) * 100, 2)
            })

    if hybrid_weight_rows:

        weight_df = pd.DataFrame(hybrid_weight_rows)

        print("\nWEIGHT ASSIGNED TO EACH RECOMMENDATION METHOD")
        print(weight_df.to_string(index=False))

    else:
        print("No matching model weights were found.")

else:
    print("The model weight dictionary was not found.")


# -----------------------------------------------------------------------------
# 6. SUMMARY OF THE RECOMMENDATION LISTS
# -----------------------------------------------------------------------------

print("\n" + "=" * 85)
print("5. WRITTEN COMPARATIVE ANALYSIS")
print("=" * 85)

# Identify the methods with the greatest overlap and ranking agreement
most_similar_pair = max(
    overlap_rows,
    key=lambda row: row["Jaccard Similarity"]
)

least_similar_pair = min(
    overlap_rows,
    key=lambda row: row["Jaccard Similarity"]
)

best_correlation = max(
    correlation_rows,
    key=lambda row: (
        row["Spearman Correlation"]
        if isinstance(row["Spearman Correlation"], float)
        else -2
    )
)

most_distinctive = diversity_df.loc[
    diversity_df["Exclusive Recommendations"].idxmax()
]

print("\nA. OVERALL RECOMMENDATION BEHAVIOUR")

print(
    f"The three recommendation methods generated "
    f"{len(all_games)} distinct games across their combined recommendation lists."
)

print(
    f"The average pairwise Jaccard similarity was "
    f"{average_jaccard:.4f}, indicating the overall degree of "
    "agreement between the methods."
)

print("\nB. SIMILARITY BETWEEN METHODS")

print(
    f"{most_similar_pair['Method Comparison']} had the greatest "
    f"recommendation overlap, with a Jaccard similarity of "
    f"{most_similar_pair['Jaccard Similarity']:.4f}."
)

print(
    f"{least_similar_pair['Method Comparison']} had the lowest "
    f"recommendation overlap, with a Jaccard similarity of "
    f"{least_similar_pair['Jaccard Similarity']:.4f}."
)

print("\nC. RANKING CONSISTENCY")

print(
    f"{best_correlation['Method Comparison']} had the strongest "
    f"Spearman rank correlation, with a value of "
    f"{best_correlation['Spearman Correlation']}."
)

print(
    "This indicates that the pair ranked their commonly recommended "
    "games more similarly than the other method pairs."
)

print("\nD. RECOMMENDATION DIVERSITY")

print(
    f"{most_distinctive['Method']} contributed the greatest number "
    f"of exclusive recommendations, introducing "
    f"{int(most_distinctive['Exclusive Recommendations'])} games "
    "that neither of the other methods recommended."
)

print(
    f"Across all methods, {all_three} games were recommended by "
    "every method, while "
    f"{only_one} games appeared in only one method's list."
)

print("\nE. HYBRID MODEL")

if hybrid_weight_rows:

    greatest_weight = max(
        hybrid_weight_rows,
        key=lambda row: row["Hybrid Weight"]
    )

    print(
        f"{greatest_weight['Method']} received the greatest hybrid "
        f"weight at {greatest_weight['Percentage Influence']:.2f}%."
    )

    print(
        "The weighting procedure therefore assigns this method the "
        "largest influence in the final Reciprocal Rank Fusion result."
    )

print(
    "\nOverall, the comparison demonstrates the complementary roles "
    "of the three recommendation approaches. KNN and Cosine Similarity "
    "use game characteristics to identify related titles, whereas "
    "Implicit ALS uses patterns in user interactions. Differences in "
    "their recommendations allow the hybrid model to combine shared "
    "recommendations with games that individual methods identify independently."
)

print("\n" + "=" * 85)
print("END OF COMPARATIVE ANALYSIS")
print("=" * 85)


# =============================================================================
#       OUT-OF-SAMPLE ACCURACY EVALUATION: THREE MODELS VS HYBRID MODEL
#
# Compares:
#   1. KNN
#   2. Cosine Similarity
#   3. Implicit ALS
#   4. Combined Hybrid (Jaccard-weighted Reciprocal Rank Fusion)
#
# Evaluation:
#   - Hold out 20% of each user's positively recommended games.
#   - Generate recommendations using the remaining 80%.
#   - Compare recommendations against the hidden games.
#
# Main evaluation metric: NDCG@10
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from itertools import combinations
from scipy.stats import wilcoxon


# =============================================================================
# 1. EVALUATION SETTINGS
# =============================================================================

OBJECTIVE_PATH = "Data_Objective.csv"
SUBJECTIVE_PATH = "Data_Subjective.csv"

MAX_EVAL_USERS = 1000
MIN_POSITIVE_GAMES = 5

TEST_FRACTION = 0.20

CANDIDATES_PER_MODEL = 100
EVALUATION_K = 10

RANDOM_SEED = 42
CHUNK_SIZE = 1_000_000

rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# 2. LOAD GAME METADATA AND DEFINE THE COMMON GAME CATALOGUE
# =============================================================================

print("\n" + "=" * 90)
print("OUT-OF-SAMPLE RECOMMENDATION ACCURACY EVALUATION")
print("=" * 90)

objective_df = pd.read_csv(
    OBJECTIVE_PATH,
    dtype={"app_id": "string"}
)

required_objective = {"app_id", "title"}

if not required_objective.issubset(objective_df.columns):
    raise ValueError(
        "Data_Objective.csv must contain app_id and title."
    )

objective_df = objective_df.dropna(
    subset=["app_id", "title"]
).copy()

# Only evaluate games that both content-based models can recommend.
# Reuse the datasets initialized above; the imported modules do not
# necessarily expose attributes named knn_df or sim_df.
if not isinstance(knn_data, pd.DataFrame) or "title" not in knn_data.columns:
    raise ValueError("The KNN dataset must be a DataFrame containing a 'title' column.")

if not isinstance(sim_data, pd.DataFrame) or "title" not in sim_data.columns:
    raise ValueError("The similarity dataset must be a DataFrame containing a 'title' column.")

knn_titles = set(
    knn_data["title"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.casefold()
)

similarity_titles = set(
    sim_data["title"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.casefold()
)

common_titles = knn_titles.intersection(similarity_titles)

if not common_titles:
    raise ValueError(
        "The KNN and Cosine Similarity datasets have no matching game titles. "
        "Check their title columns and preprocessing."
    )

objective_df["title_key"] = (
    objective_df["title"]
    .astype(str)
    .str.strip()
    .str.casefold()
)

objective_df = objective_df[
    objective_df["title_key"].isin(common_titles)
].drop_duplicates("app_id").copy()

app_id_to_title = dict(
    zip(objective_df["app_id"], objective_df["title"])
)

app_id_to_index = {
    app_id: index
    for index, app_id in enumerate(app_id_to_title)
}

index_to_app_id = {
    index: app_id
    for app_id, index in app_id_to_index.items()
}

n_items = len(app_id_to_index)

valid_app_ids = set(app_id_to_index.keys())

print(f"\nCommon games available for evaluation: {n_items}")


# =============================================================================
# 3. IDENTIFY USERS WITH POSITIVE STEAM RECOMMENDATIONS
# =============================================================================
#
# We use is_recommended == 1 as the definition of a relevant game.
#
# This means the evaluation tests whether the models can predict games
# that a user actually recommended positively.
#
# We use chunks to avoid loading all 30 million interaction records
# into memory at once.
# =============================================================================

print("\nScanning interaction dataset for eligible users...")

user_counts = Counter()

interaction_columns = [
    "user_id",
    "app_id",
    "hours",
    "is_recommended"
]


def positive_recommendation_mask(data):

    values = (
        data["is_recommended"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return values.isin(["1", "1.0", "true", "yes"])


for chunk in pd.read_csv(
    SUBJECTIVE_PATH,
    usecols=interaction_columns,
    dtype={
        "user_id": "string",
        "app_id": "string"
    },
    chunksize=CHUNK_SIZE
):

    chunk = chunk.dropna(
        subset=["user_id", "app_id"]
    )

    chunk = chunk[
        positive_recommendation_mask(chunk)
        & chunk["app_id"].isin(valid_app_ids)
    ]

    counts = chunk.groupby("user_id").size()

    user_counts.update(
        counts.to_dict()
    )


eligible_users = [
    user
    for user, count in user_counts.items()
    if count >= MIN_POSITIVE_GAMES
]

print(f"Eligible users found: {len(eligible_users):,}")

if len(eligible_users) == 0:
    raise ValueError(
        "No eligible users were found. Check the interaction data "
        "and app_id matching."
    )

selected_user_count = min(
    MAX_EVAL_USERS,
    len(eligible_users)
)

selected_users = set(
    rng.choice(
        eligible_users,
        size=selected_user_count,
        replace=False
    ).tolist()
)

print(f"Users selected for evaluation: {len(selected_users):,}")


# =============================================================================
# 4. LOAD INTERACTIONS FOR THE SELECTED USERS
# =============================================================================

print("\nLoading interactions for selected users...")

selected_interaction_parts = []

for chunk in pd.read_csv(
    SUBJECTIVE_PATH,
    usecols=interaction_columns,
    dtype={
        "user_id": "string",
        "app_id": "string"
    },
    chunksize=CHUNK_SIZE
):

    chunk = chunk.dropna(
        subset=["user_id", "app_id"]
    )

    chunk = chunk[
        chunk["user_id"].isin(selected_users)
        & chunk["app_id"].isin(valid_app_ids)
        & positive_recommendation_mask(chunk)
    ]

    if not chunk.empty:
        selected_interaction_parts.append(chunk)


if not selected_interaction_parts:
    raise ValueError(
        "No interactions were loaded for the selected users."
    )

interactions = pd.concat(
    selected_interaction_parts,
    ignore_index=True
)

interactions["hours"] = pd.to_numeric(
    interactions["hours"],
    errors="coerce"
).fillna(0).clip(lower=0)

# Keep one interaction per user/game.
# Maximum playtime is retained if duplicate records exist.
interactions = (
    interactions
    .groupby(["user_id", "app_id"], as_index=False)
    .agg(hours=("hours", "max"))
)

print(f"Unique positive user-game interactions: {len(interactions):,}")


# =============================================================================
# 5. CREATE TRAINING AND TEST SETS
# =============================================================================
#
# Each user receives:
#
# Training profile = 80% of positively recommended games
# Test set         = 20% of positively recommended games
#
# Each user must retain at least three training games.
#
# The test games are never supplied to any recommender.
# =============================================================================

user_profiles = {}
training_records = []

for user_id, group in interactions.groupby("user_id"):

    game_ids = group["app_id"].tolist()

    if len(game_ids) < MIN_POSITIVE_GAMES:
        continue

    shuffled = rng.permutation(
        len(game_ids)
    )

    n_test = max(
        1,
        int(round(len(game_ids) * TEST_FRACTION))
    )

    n_test = min(
        n_test,
        len(game_ids) - 3
    )

    test_positions = shuffled[:n_test]
    train_positions = shuffled[n_test:]

    train_ids = [
        game_ids[position]
        for position in train_positions
    ]

    test_ids = [
        game_ids[position]
        for position in test_positions
    ]

    if not train_ids or not test_ids:
        continue

    user_profiles[user_id] = {
        "train": train_ids,
        "test": test_ids
    }

    for app_id in train_ids:

        hours = float(
            group.loc[
                group["app_id"].eq(app_id),
                "hours"
            ].max()
        )

        training_records.append({
            "user_id": user_id,
            "app_id": app_id,
            "hours": hours
        })


print(f"\nUsers with valid train/test splits: {len(user_profiles):,}")

if len(user_profiles) < 10:
    raise ValueError(
        "Too few users have enough games after filtering. "
        "Try lowering MIN_POSITIVE_GAMES."
    )


# =============================================================================
# 6. TRAIN ALS USING ONLY THE TRAINING INTERACTIONS
# =============================================================================
#
# A fresh ALS model is fitted here.
#
# The original fully trained ALS model is deliberately NOT used,
# because it has already seen the held-out interactions.
#
# The confidence weighting follows the original ALS approach:
#
# interaction strength = log(1 + hours) + 1
#
# Only positively recommended games are used in this evaluation.
# =============================================================================

print("\n" + "-" * 90)
print("TRAINING A FRESH ALS MODEL USING TRAINING DATA ONLY")
print("-" * 90)


class EvaluationALS:

    def __init__(
        self,
        factors=32,
        iterations=15,
        regularization=0.05,
        alpha=40.0,
        random_state=42
    ):

        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None

    def fit(self, user_items):

        user_items = user_items.tocsr().astype(
            np.float32
        )

        n_users, n_items = user_items.shape

        random_generator = np.random.default_rng(
            self.random_state
        )

        self.user_factors = random_generator.normal(
            0,
            0.01,
            (n_users, self.factors)
        ).astype(np.float32)

        self.item_factors = random_generator.normal(
            0,
            0.01,
            (n_items, self.factors)
        ).astype(np.float32)

        identity = np.eye(
            self.factors,
            dtype=np.float32
        )

        regularization_matrix = (
            self.regularization * identity
        )

        item_matrix = user_items.tocsc()

        for iteration in range(self.iterations):

            # Update user factors
            item_gram = (
                self.item_factors.T
                @ self.item_factors
            )

            for user_index in range(n_users):

                row = user_items.getrow(user_index)

                if row.nnz == 0:
                    self.user_factors[user_index] = 0
                    continue

                item_indices = row.indices
                values = row.data

                factors = self.item_factors[item_indices]

                confidence = self.alpha * values

                A = (
                    item_gram
                    + (factors.T * confidence) @ factors
                    + regularization_matrix
                )

                b = factors.T @ (1.0 + confidence)

                self.user_factors[user_index] = np.linalg.solve(
                    A,
                    b
                )

            # Update item factors
            user_gram = (
                self.user_factors.T
                @ self.user_factors
            )

            for item_index in range(n_items):

                column = item_matrix.getcol(item_index)

                if column.nnz == 0:
                    self.item_factors[item_index] = 0
                    continue

                user_indices = column.indices
                values = column.data

                factors = self.user_factors[user_indices]

                confidence = self.alpha * values

                A = (
                    user_gram
                    + (factors.T * confidence) @ factors
                    + regularization_matrix
                )

                b = factors.T @ (1.0 + confidence)

                self.item_factors[item_index] = np.linalg.solve(
                    A,
                    b
                )

            print(
                f"ALS iteration {iteration + 1}/{self.iterations} complete"
            )

        return self

    def recommend(self, seed_indices, top_n=100):

        seed_indices = np.asarray(
            seed_indices,
            dtype=int
        )

        if len(seed_indices) == 0:
            return []

        selected_factors = self.item_factors[
            seed_indices
        ]

        confidence = np.ones(
            len(seed_indices),
            dtype=np.float32
        ) * (1.0 + self.alpha)

        A = (
            (selected_factors.T * confidence)
            @ selected_factors
        )

        A += (
            self.regularization
            * np.eye(self.factors, dtype=np.float32)
        )

        b = selected_factors.T @ confidence

        temporary_user_vector = np.linalg.solve(
            A,
            b
        )

        scores = (
            self.item_factors
            @ temporary_user_vector
        )

        # Exclude all games already in the user's profile.
        scores[seed_indices] = -np.inf

        count = min(
            top_n,
            len(scores)
        )

        best_indices = np.argpartition(
            scores,
            -count
        )[-count:]

        best_indices = best_indices[
            np.argsort(scores[best_indices])[::-1]
        ]

        return best_indices.tolist()


# Map training users and games to matrix positions
training_users = list(user_profiles.keys())

user_to_row = {
    user_id: index
    for index, user_id in enumerate(training_users)
}

als_rows = []
als_columns = []
als_values = []

for record in training_records:

    user_index = user_to_row[
        record["user_id"]
    ]

    item_index = app_id_to_index[
        record["app_id"]
    ]

    strength = (
        np.log1p(record["hours"]) + 1.0
    )

    als_rows.append(user_index)
    als_columns.append(item_index)
    als_values.append(strength)


training_matrix = csr_matrix(
    (
        np.asarray(als_values, dtype=np.float32),
        (als_rows, als_columns)
    ),
    shape=(
        len(training_users),
        n_items
    ),
    dtype=np.float32
)

training_matrix.sum_duplicates()

evaluation_als_model = EvaluationALS(
    factors=32,
    iterations=15,
    regularization=0.05,
    alpha=40.0,
    random_state=RANDOM_SEED
).fit(training_matrix)


# =============================================================================
# 7. DEFINE EVALUATION METRICS
# =============================================================================

def normalize_title(title):

    return str(title).strip().casefold()


def extract_recommendation_titles(results):

    if isinstance(results, pd.DataFrame):

        if "Game" in results.columns:
            return results["Game"].dropna().astype(str).tolist()

        if "title" in results.columns:
            return results["title"].dropna().astype(str).tolist()

    titles = []

    if isinstance(results, list):

        for item in results:

            if isinstance(item, dict):

                title = (
                    item.get("title")
                    or item.get("Game")
                    or item.get("game")
                )

                if title is not None:
                    titles.append(str(title))

            elif isinstance(item, str):
                titles.append(item)

    return titles


def calculate_metrics(recommended_titles, relevant_titles, k=10):

    recommendations = [
        normalize_title(title)
        for title in recommended_titles
    ][:k]

    relevant = {
        normalize_title(title)
        for title in relevant_titles
    }

    if not relevant:
        return None

    hits = [
        1 if game in relevant else 0
        for game in recommendations
    ]

    number_of_hits = sum(hits)

    precision = number_of_hits / k

    recall = number_of_hits / len(relevant)

    hit_rate = 1.0 if number_of_hits > 0 else 0.0

    # Discounted Cumulative Gain
    dcg = sum(
        hit / np.log2(position + 2)
        for position, hit in enumerate(hits)
    )

    # Ideal DCG
    ideal_hits = min(
        len(relevant),
        k
    )

    idcg = sum(
        1.0 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Average Precision at K
    cumulative_hits = 0
    precision_sum = 0.0

    for position, hit in enumerate(hits, start=1):

        if hit:
            cumulative_hits += 1
            precision_sum += cumulative_hits / position

    average_precision = (
        precision_sum / min(len(relevant), k)
        if relevant else 0.0
    )

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg,
        "Hit Rate@10": hit_rate,
        "MAP@10": average_precision
    }


# =============================================================================
# 8. GENERATE RECOMMENDATIONS FOR EVERY TEST USER
# =============================================================================

print("\n" + "-" * 90)
print("GENERATING AND EVALUATING RECOMMENDATIONS")
print("-" * 90)

method_names = [
    "KNN",
    "Cosine Similarity",
    "Implicit ALS",
    "Combined Hybrid"
]

evaluation_rows = []

coverage_sets = {
    method: set()
    for method in method_names
}

successful_users = 0
failed_users = 0

for user_number, (user_id, profile) in enumerate(
    user_profiles.items(),
    start=1
):

    seed_app_ids = profile["train"]
    test_app_ids = profile["test"]

    seed_titles = [
        app_id_to_title[app_id]
        for app_id in seed_app_ids
        if app_id in app_id_to_title
    ]

    test_titles = [
        app_id_to_title[app_id]
        for app_id in test_app_ids
        if app_id in app_id_to_title
    ]

    if len(seed_titles) < 3 or not test_titles:
        failed_users += 1
        continue

    try:

        # ---------------------------------------------------------------------
        # KNN
        # ---------------------------------------------------------------------

        _, knn_recommendations = knn_fn(
            seed_titles,
            knn_data,
            top_n=CANDIDATES_PER_MODEL
        )

        # ---------------------------------------------------------------------
        # COSINE SIMILARITY
        # ---------------------------------------------------------------------

        _, similarity_recommendations = sim_fn(
            seed_titles,
            sim_data,
            top_n=CANDIDATES_PER_MODEL
        )

        if not isinstance(knn_recommendations, list):
            failed_users += 1
            continue

        if not isinstance(similarity_recommendations, list):
            failed_users += 1
            continue

        # ---------------------------------------------------------------------
        # IMPLICIT ALS
        # ---------------------------------------------------------------------

        seed_indices = [
            app_id_to_index[app_id]
            for app_id in seed_app_ids
            if app_id in app_id_to_index
        ]

        als_recommended_indices = evaluation_als_model.recommend(
            seed_indices,
            top_n=CANDIDATES_PER_MODEL
        )

        als_recommendations = []

        for item_index in als_recommended_indices:

            app_id = index_to_app_id[item_index]

            title = app_id_to_title.get(app_id)

            if title is not None:

                als_recommendations.append({
                    "title": title
                })

        # ---------------------------------------------------------------------
        # COMBINED HYBRID
        # ---------------------------------------------------------------------

        model_results = {
            "KNN": knn_recommendations,
            "Similarity": similarity_recommendations,
            "Implicit ALS": als_recommendations
        }

        hybrid_ranking, _, _ = combine_recommendations(
            model_results,
            top_n=EVALUATION_K
        )

        # ---------------------------------------------------------------------
        # Evaluate all four methods against the same hidden test games
        # ---------------------------------------------------------------------

        recommendations_to_evaluate = {
            "KNN": knn_recommendations,
            "Cosine Similarity": similarity_recommendations,
            "Implicit ALS": als_recommendations,
            "Combined Hybrid": hybrid_ranking
        }

        for method, recommendations in recommendations_to_evaluate.items():

            predicted_titles = extract_recommendation_titles(
                recommendations
            )

            metric_values = calculate_metrics(
                predicted_titles,
                test_titles,
                k=EVALUATION_K
            )

            if metric_values is None:
                continue

            evaluation_rows.append({
                "User": str(user_id),
                "Method": method,
                **metric_values
            })

            for title in predicted_titles[:EVALUATION_K]:
                coverage_sets[method].add(
                    normalize_title(title)
                )

        successful_users += 1

        if user_number % 25 == 0:

            print(
                f"Processed {user_number}/{len(user_profiles)} users..."
            )

    except Exception as error:

        failed_users += 1

        print(
            f"\\nERROR ON USER {user_number} (ID: {user_id})"
        )

        import traceback
        traceback.print_exc()

        # Stop after the first failure so the traceback is readable.
        break


if not evaluation_rows:
    raise ValueError(
        "No users were successfully evaluated. "
        "Check the recommendation function names and input data."
    )

evaluation_df = pd.DataFrame(
    evaluation_rows
)

print("\nEvaluation completed.")
print(f"Successfully evaluated users: {successful_users}")
print(f"Skipped users: {failed_users}")


# =============================================================================
# 9. CALCULATE FINAL PERFORMANCE OF EACH METHOD
# =============================================================================

print("\n" + "=" * 90)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 90)

summary_rows = []

for method in method_names:

    method_data = evaluation_df[
        evaluation_df["Method"] == method
    ]

    if method_data.empty:
        continue

    summary_rows.append({

        "Method": method,

        "Users Evaluated": len(method_data),

        "Precision@10": method_data["Precision@10"].mean(),

        "Recall@10": method_data["Recall@10"].mean(),

        "NDCG@10": method_data["NDCG@10"].mean(),

        "Hit Rate@10": method_data["Hit Rate@10"].mean(),

        "MAP@10": method_data["MAP@10"].mean(),

        "Catalog Coverage": (
            len(coverage_sets[method]) / n_items
        )

    })


performance_df = pd.DataFrame(
    summary_rows
).sort_values(
    "NDCG@10",
    ascending=False
).reset_index(drop=True)

performance_df.index += 1
performance_df.index.name = "Overall Rank"

print("\nMODEL PERFORMANCE TABLE")

print(
    performance_df.to_string(
        float_format=lambda value: f"{value:.4f}"
    )
)


# =============================================================================
# 10. IDENTIFY THE BEST INDIVIDUAL MODEL
# =============================================================================

individual_methods = [
    "KNN",
    "Cosine Similarity",
    "Implicit ALS"
]

individual_results = performance_df[
    performance_df["Method"].isin(individual_methods)
]

best_individual = individual_results.iloc[0]

hybrid_results = performance_df[
    performance_df["Method"] == "Combined Hybrid"
]

if hybrid_results.empty:
    raise ValueError(
        "The combined hybrid did not produce evaluation results."
    )

hybrid_result = hybrid_results.iloc[0]

best_overall = performance_df.iloc[0]

print("\n" + "=" * 90)
print("OPTIMAL METHOD IDENTIFICATION")
print("=" * 90)

print(
    f"\nBest individual method: {best_individual['Method']}"
)

print(
    f"Best individual NDCG@10: {best_individual['NDCG@10']:.4f}"
)

print(
    f"Combined hybrid NDCG@10: {hybrid_result['NDCG@10']:.4f}"
)

if hybrid_result["NDCG@10"] > best_individual["NDCG@10"]:

    improvement = (
        hybrid_result["NDCG@10"]
        - best_individual["NDCG@10"]
    )

    percentage_improvement = (
        improvement / best_individual["NDCG@10"] * 100
        if best_individual["NDCG@10"] > 0 else np.nan
    )

    print(
        "\nRESULT: The combined hybrid outperformed every "
        "individual recommendation method."
    )

    print(
        f"NDCG improvement over the best individual: "
        f"{improvement:.4f}"
    )

    print(
        f"Relative improvement: {percentage_improvement:.2f}%"
    )

elif hybrid_result["NDCG@10"] < best_individual["NDCG@10"]:

    difference = (
        best_individual["NDCG@10"]
        - hybrid_result["NDCG@10"]
    )

    print(
        "\nRESULT: The combined hybrid did NOT outperform "
        "the best individual recommendation method."
    )

    print(
        f"The best individual method performs better by "
        f"{difference:.4f} NDCG points."
    )

else:

    print(
        "\nRESULT: The combined hybrid and the best individual "
        "method achieved equal NDCG@10."
    )

print(
    f"\nBEST OVERALL METHOD: {best_overall['Method']}"
)


# =============================================================================
# 11. DIRECT USER-BY-USER COMPARISON
# =============================================================================
#
# This determines how often the hybrid beats each individual method
# on the same users, rather than only comparing overall averages.
# =============================================================================

print("\n" + "=" * 90)
print("HYBRID VS INDIVIDUAL METHODS: PAIRED COMPARISON")
print("=" * 90)

user_ndcg = evaluation_df.pivot(
    index="User",
    columns="Method",
    values="NDCG@10"
)

paired_rows = []

for method in individual_methods:

    if method not in user_ndcg.columns:
        continue

    if "Combined Hybrid" not in user_ndcg.columns:
        continue

    comparison = user_ndcg[
        ["Combined Hybrid", method]
    ].dropna()

    differences = (
        comparison["Combined Hybrid"]
        - comparison[method]
    )

    hybrid_wins = int(
        (differences > 0).sum()
    )

    individual_wins = int(
        (differences < 0).sum()
    )

    ties = int(
        (differences == 0).sum()
    )

    try:

        if np.allclose(differences, 0):
            p_value = 1.0

        else:
            p_value = wilcoxon(
                comparison["Combined Hybrid"],
                comparison[method],
                alternative="greater"
            ).pvalue

    except Exception:
        p_value = np.nan

    paired_rows.append({

        "Comparison": f"Hybrid vs {method}",

        "Users Compared": len(comparison),

        "Hybrid Wins": hybrid_wins,

        "Individual Wins": individual_wins,

        "Ties": ties,

        "Average NDCG Difference": differences.mean(),

        "Wilcoxon p-value": p_value

    })


paired_df = pd.DataFrame(
    paired_rows
)

print(
    paired_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# =============================================================================
# 12. INTERPRET THE EVALUATION
# =============================================================================

print("\n" + "=" * 90)
print("EVALUATION CONCLUSION")
print("=" * 90)

print(
    f"\nThe evaluation used {successful_users} users, "
    "with each user's positively recommended games divided "
    "into training and testing sets."
)

print(
    f"The primary evaluation metric was NDCG@{EVALUATION_K}, "
    "which measures how effectively each model places relevant "
    "games near the top of its recommendation list."
)

print(
    f"\nThe strongest individual model was {best_individual['Method']}, "
    f"with an average NDCG@10 of {best_individual['NDCG@10']:.4f}."
)

print(
    f"The combined hybrid achieved an average NDCG@10 of "
    f"{hybrid_result['NDCG@10']:.4f}."
)

if best_overall["Method"] == "Combined Hybrid":

    print(
        "\nCONCLUSION: The combined hybrid was the strongest "
        "method according to the primary ranking-quality metric."
    )

else:

    print(
        f"\nCONCLUSION: {best_overall['Method']} was the strongest "
        "method according to the primary ranking-quality metric."
    )

print(
    "\nThe paired comparisons provide additional evidence by "
    "measuring how frequently the hybrid outperforms each "
    "individual method on the same users."
)

print(
    "\nA lower Wilcoxon p-value provides stronger evidence that "
    "the hybrid's NDCG improvements are consistent rather than "
    "being caused by a small number of users. A p-value below "
    "0.05 is commonly used as a statistical significance threshold."
)

print(
    "\nIMPORTANT: These results measure prediction of positive "
    "Steam recommendations in the sampled users. They should not "
    "be interpreted as proof of universal recommendation accuracy."
)

print("\n" + "=" * 90)
print("END OF OUT-OF-SAMPLE EVALUATION")
print("=" * 90)


# =============================================================================
# 13. SAVE RESULTS
# =============================================================================

performance_df.to_csv(
    "Model_Accuracy_Comparison.csv"
)

evaluation_df.to_csv(
    "Individual_User_Evaluation_Results.csv",
    index=False
)

paired_df.to_csv(
    "Hybrid_Paired_Comparison.csv",
    index=False
)

print("\nSaved evaluation files:")
print("  Model_Accuracy_Comparison.csv")
print("  Individual_User_Evaluation_Results.csv")
print("  Hybrid_Paired_Comparison.csv")


# -----------------------------------------------------------------------------
# Extra check: uncertainty around the main ranking metric
# -----------------------------------------------------------------------------
# This does not change any recommendation or score above. It only estimates
# how much the measured average NDCG might vary across sampled users.

print("\nNDCG@10 UNCERTAINTY (95% bootstrap intervals)")
bootstrap_rng = np.random.default_rng(2026)
bootstrap_rows = []
for method in method_names:
    values = evaluation_df.loc[
        evaluation_df["Method"] == method, "NDCG@10"
    ].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        continue
    bootstrap_means = np.array([
        bootstrap_rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(1000)
    ])
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    bootstrap_rows.append({
        "Method": method,
        "Mean NDCG@10": values.mean(),
        "95% CI lower": lower,
        "95% CI upper": upper,
    })

uncertainty_df = pd.DataFrame(bootstrap_rows)
print(uncertainty_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
uncertainty_df.to_csv("Model_NDCG_Uncertainty.csv", index=False)
print("Saved: Model_NDCG_Uncertainty.csv")
print("Note: These intervals reflect user sampling, not other sources of bias.")
