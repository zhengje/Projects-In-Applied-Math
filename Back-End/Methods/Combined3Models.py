

# =============================================================================
# 3. RUN MODEL EXECUTION AND HARVEST RANK LISTS
# =============================================================================
import FinalKNNMethod as knn_script
import FinalSimilarityMatrixMethod as sim_script
import FinalIMFMethod as imf_script
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================
# Update these paths inside Combined3Models.py to match your workspace folder layout
# DATA_OBJECTIVE_PATH = "Back-End/Methods/Data_Objective.csv"
# DATA_SUBJECTIVE_PATH = "Back-End/Methods/Data_Subjective.csv"

# Shared primary metadata dataset
dataset = pd.read_csv("Back-End/Methods/Data_Objective.csv")
print(f"Fixed metadata dataset loaded: {len(dataset)} games")

# Subjective user-interaction dataset
subjective_df = pd.read_csv("Back-End/Methods/Data_Subjective.csv")
print(f"Subjective interaction dataset loaded: {len(subjective_df)} records")

# Central tracking parameters matching Section 5 constraints
CANDIDATES_PER_MODEL = 100
FINAL_TOP_N = 10

N_RECS = CANDIDATES_PER_MODEL  # Feeds 100 deep candidates per model for high-coverage fusion


# Central tracking parameters matching Section 5 constraints
input_games = ["Omori", "Resident Evil"]
N_RECS = CANDIDATES_PER_MODEL  # Feeds 100 deep candidates per model for high-coverage fusion

# -----------------------------------------------------------------------------
# MODEL 1: KNN Content Model
# -----------------------------------------------------------------------------
# Already returns dictionaries matching [{'title': 'Name', ...}] out-of-the-box
_, knn_results = knn_script.recommend_games_knn(input_games, dataset, top_n=N_RECS)

# -----------------------------------------------------------------------------
# MODEL 2: Cosine Similarity Matrix
# -----------------------------------------------------------------------------
# Already returns dictionaries matching [{'title': 'Name', ...}] out-of-the-box
_, sim_results = sim_script.recommend_games(input_games, dataset, top_n=N_RECS)

# -----------------------------------------------------------------------------
# MODEL 3: Implicit Matrix Factorization Multi
# -----------------------------------------------------------------------------
# 1. Run initialization script to resolve matrices and weights mapping dictionaries
model, _, _, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = imf_script.implicit_mf()

# 2. Build live session sparse vector array map coordinates
valid_app_ids = []
for name in input_games:
    search_name = name.strip().lower()
    app_id = title_to_app_id_map.get(search_name)
    if app_id is not None and app_id in app_id_to_index_map:
        valid_app_ids.append(app_id)

id_indices = [app_id_to_index_map[id] for id in valid_app_ids]
cols = id_indices
rows = [0] * len(cols)
data = [1.0] * len(cols)
user_items = csr_matrix((data, (rows, cols)), shape=(1, model.item_factors.shape[0]))

# 3. Request ALS candidate coordinates
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

# 4. Wrap outputs into a structure matching {'title': rec_title} for Section 5 RRF
implicit_results = []
for idx in ids:
    rec_app_id = app_index_map[idx]
    rec_title = game_title.get(rec_app_id)
    if rec_title:
        implicit_results.append({'title': rec_title})

# Confirm matrix harvest array counts match expectations before pipeline merging
print(f"Successfully gathered list arrays:")
print(f" -> KNN Recs Count: {len(knn_results)}")
print(f" -> Similarity Recs Count: {len(sim_results)}")
print(f" -> Implicit ALS Recs Count: {len(implicit_results)}")


# =============================================================================
# COMBINE PIPELINE ENGINE FUNCTIONS
# =============================================================================
def game_key(title):
    return str(title).strip().casefold()

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0

def combine_recommendations(model_results, top_n=10):
    RRF_K = 60
    names = list(model_results)
    rank_maps = {}
    display_titles = {}

    for name, recommendations in model_results.items():
        ranks = {}
        for position, rec in enumerate(recommendations, start=1):
            key = game_key(rec['title'])
            if key not in ranks:
                ranks[key] = position
                display_titles.setdefault(key, rec['title'])
        rank_maps[name] = ranks

    # Generate 3x3 Jaccard Overlap Matrix
    overlap = pd.DataFrame(
        [[jaccard(rank_maps[a], rank_maps[b]) for b in names] for a in names],
        index=names, columns=names
    )

    raw_weights = {
        name: 1 / (1 + sum(overlap.loc[name, other] for other in names if other != name))
        for name in names
    }
    total_weight = sum(raw_weights.values())
    weights = {name: raw_weights[name] / total_weight for name in names}

    rows = []
    for key, title in display_titles.items():
        score = sum(weights[name] / (RRF_K + rank_maps[name][key]) for name in names if key in rank_maps[name])
        row = {'Game': title, 'Fusion score': score, 'Models agreeing': sum(key in rank_maps[n] for n in names)}
        for name in names:
            row[f'{name} rank'] = rank_maps[name].get(key, None)
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ['Fusion score', 'Models agreeing', 'Game'],
        ascending=[False, False, True]
    ).head(top_n).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = 'Rank'

    return ranking, overlap, weights


# =============================================================================
# MAP RECS DIRECTLY INTO YOUR COMBINE PIPELINE ENGINE
# =============================================================================
model_results = {
    'KNN': knn_results,
    'Similarity': sim_results,
    'Implicit ALS': implicit_results
}

# Run the Section 5 evaluation pipeline directly using your variables
ranking, overlap, weights = combine_recommendations(model_results, top_n=FINAL_TOP_N)

print('\n3x3 JACCARD OVERLAP MATRIX')
print(overlap.round(4).to_string())
print('\nMODEL WEIGHTS:', {k: round(v, 4) for k, v in weights.items()})
print('\nFINAL HYBRID TOP 10')
print(ranking.to_string(float_format=lambda x: f'{x:.6f}'))
