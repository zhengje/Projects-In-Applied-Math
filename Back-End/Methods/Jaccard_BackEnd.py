
import FinalKNNMethod as knn_script
import FinalSimilarityMatrixMethod as sim_script
import FinalIMFMethod as imf_script
import FinalGNNMethod as gnn_script
from scipy.sparse import csr_matrix
from itertools import islice
import numpy as np
import pandas as pd

# =============================================================================
# 1. LOAD DATASETS
# =============================================================================

dataset = pd.read_csv("Data_Objective.csv")
print(f"Fixed metadata dataset loaded: {len(dataset)} games")

subjective_df = pd.read_csv("Data_Subjective.csv")
print(f"Subjective interaction dataset loaded: {len(subjective_df)} records")


CANDIDATES_PER_MODEL = 100
FINAL_TOP_N = 10

# Set up Filters for Backend
filterKey = ['title']

N_RECS = CANDIDATES_PER_MODEL  # Feeds 100 deep candidates per model for high-coverage fusion


input_games = ["Slay the Spire","Hades","Into the Breach"]

# -----------------------------------------------------------------------------
# MODEL 1: KNN Content Model
# -----------------------------------------------------------------------------

# Already returns dictionaries matching [{'title': 'Name', ...}] out-of-the-box
_, knn_results = knn_script.recommend_games_knn(input_games, dataset, top_n=N_RECS)

# Return Top 10 KNN Titles as an Array
print("KNN Top 10 Loading")
tempKNN = knn_results[:10]
knn_top10 = []

for rank in range(FINAL_TOP_N):
    temp = {}
    for key in filterKey:
        if key in tempKNN[rank]:
            temp[key] = tempKNN[rank][key]
            knn_top10.append(temp)

print(knn_top10) # Additional Test - To Be Commented Later
print("KNN Top 10 Done")


# -----------------------------------------------------------------------------
# MODEL 2: Cosine Similarity Matrix
# -----------------------------------------------------------------------------
# Already returns dictionaries matching [{'title': 'Name', ...}] out-of-the-box
_, sim_results = sim_script.recommend_games(input_games, dataset, top_n=N_RECS)

# Return Top 10 CosSim Titles as an Array
print("SIM Top 10 Loading")
tempSIM = sim_results[:10]
sim_top10 = []

for rank in range(FINAL_TOP_N):
    temp = {}
    for key in filterKey:
        if key in tempSIM[rank]:
            temp[key] = tempSIM[rank][key]
            sim_top10.append(temp)

print(sim_top10) # Additional Test - To Be Commented Later
print("SIM Top 10 Done")


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


# Return Top 10 IMF Titles as an Array
print("IMF Top 10 Loading")
imf_top10 = implicit_results[:10]
print(imf_top10) # Additional Test - To Be Commented Later
print("IMF Top 10 Done")


# -----------------------------------------------------------------------------
# MODEL 4: Graph Neural Network (3-Hop Bipartite GNN)
# -----------------------------------------------------------------------------
gnn_results = gnn_script.get_recommendations(input_games, top_k=N_RECS)

# Return Top 10 GNN Titles as an Array
print("GNN Top 10 Loading")
gnn_top10 = gnn_results[:10]
print(gnn_top10) # Additional Test - To Be Commented Later
print("GNN Top 10 Done")


# Confirm matrix harvest array counts match expectations before pipeline merging
print(f"Successfully gathered list arrays:")
print(f" -> KNN Recs Count: {len(knn_results)}")
print(f" -> Similarity Recs Count: {len(sim_results)}")
print(f" -> Implicit ALS Recs Count: {len(implicit_results)}")
print(f" -> GNN Recs Count: {len(gnn_results)}")


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
    ranking.index = (ranking.index +1).astype('int64')
    ranking.index.name = 'Rank'


    print("JAC Top 10 Loading")
    tempJAC = ranking['Game']
    jac_top10 = []

    for key, value in tempJAC.items():
        temp = {'title': value}
        print(temp)
        jac_top10.append(temp)

    print(jac_top10)
    print("JAC Top 10 Done")
    

    rank_cols = [col for col in ranking.columns if col.endswith(' rank')]
    ranking[rank_cols] = ranking[rank_cols].astype('Int64')

    return ranking, overlap, weights


# =============================================================================
# MAP RECS DIRECTLY INTO YOUR COMBINE PIPELINE ENGINE
# =============================================================================
model_results = {
    'KNN': knn_results,
    'CosineSimilarity': sim_results,
    'Implicit Matrix Factorization': implicit_results,
    'GNN' : gnn_results
}


# Run the Section 5 evaluation pipeline directly using your variables
ranking, overlap, weights = combine_recommendations(model_results, top_n=FINAL_TOP_N)


print('\n4x4 JACCARD OVERLAP MATRIX')
print(overlap.round(4).to_string())
print('\nMODEL WEIGHTS:', {k: round(v, 4) for k, v in weights.items()})
print('\nFINAL HYBRID TOP 10')
print(ranking.to_string(float_format=lambda x: f'{x:6f}'))


# FINAL HYBRID TOP 10
#                                                Game  Fusion score  Models agreeing  KNN rank  CosineSimilarity rank  Implicit Matrix Factorization rank  GNN rank
# Rank                                                                                                                                                             
# 1                         The Witcher® 3: Wild Hunt      0.014374                4        21                      9                                   1        12
# 2     Divinity: Original Sin 2 - Definitive Edition      0.012638                4        68                     28                                   9         6
# 3                                        Dead Cells      0.011009                4        63                     47                                   4        40
# 4                                 Vampire Survivors      0.010754                3         7                      5                                   2      <NA>
# 5                                       Inscryption      0.010061                3        10                     10                                   7      <NA>
# 6                     The Binding of Isaac: Rebirth      0.010002                3         4                      4                                  19      <NA>
# 7                            FTL: Faster Than Light      0.009867                3         1                      1                                  31      <NA>
# 8                                         Grim Dawn      0.009859                3         9                      3                                  18      <NA>
# 9                                         ASTRONEER      0.009693                3        14                     13                                <NA>        17
# 10                                    Monster Train      0.008791                3         2                      2                                  74      <NA>




# Set up route to return all reccomendations (knn_top10, sim_top10, imf_top10, gnn_top10, jac_top10)
# def return_recs(games):
#     return jaccard(games)