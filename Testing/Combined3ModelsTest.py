import FinalKNNMethod as knn_script
import FinalSimilarityMatrixMethod as sim_script
import FinalIMFMethod as imf_script
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


# =============================================================================
# COMBINE PIPELINE ENGINE FUNCTIONS (Safe to import)
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

    ranking.index = (ranking.index + 1).astype('Int64')
    ranking.index.name = 'Rank'

    rank_cols = [col for col in ranking.columns if col.endswith(' rank')]
    ranking[rank_cols] = ranking[rank_cols].astype('Int64')

    return ranking, overlap, weights


# =============================================================================
# STANDALONE RUN (Only executes when running Combined3Models.py directly)
# =============================================================================
if __name__ == "__main__":
    CANDIDATES_PER_MODEL = 100
    FINAL_TOP_N = 10
    N_RECS = CANDIDATES_PER_MODEL

    input_games = ["Slay the Spire", "Hades", "Into the Breach"]

    # MODEL 1: KNN Content Model (use knn_script.df so indices align!)
    _, knn_results = knn_script.recommend_games_knn(input_games, knn_script.df, top_n=N_RECS)
    knn_results = knn_results if isinstance(knn_results, list) else []

    # MODEL 2: Cosine Similarity Matrix (use sim_script.df so indices align!)
    _, sim_results = sim_script.recommend_games(input_games, sim_script.df, top_n=N_RECS)
    sim_results = sim_results if isinstance(sim_results, list) else []

    # MODEL 3: Implicit Matrix Factorization Multi
    model, _, _, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = imf_script.implicit_mf()

    valid_app_ids = [
        title_to_app_id_map[name.strip().lower()]
        for name in input_games
        if name.strip().lower() in title_to_app_id_map
        and title_to_app_id_map[name.strip().lower()] in app_id_to_index_map
    ]

    implicit_results = []
    if valid_app_ids:
        cols = [app_id_to_index_map[aid] for aid in valid_app_ids]
        rows = [0] * len(cols)
        data = [1.0] * len(cols)
        user_items = csr_matrix((data, (rows, cols)), shape=(1, model.item_factors.shape[0]))

        try:
            ids, _ = model.recommend(
                userid=0,
                user_items=user_items,
                N=N_RECS,
                filter_already_liked_items=True,
                recalculate_user=True
            )
        except TypeError:
            ids, _ = model.recommend(
                userid=0,
                user_items=user_items,
                N=N_RECS,
                filter_already_liked_items=True
            )

        if isinstance(ids, np.ndarray) and ids.ndim > 1:
            ids = ids[0]

        for idx in ids:
            rec_title = game_title.get(app_index_map[idx])
            if rec_title:
                implicit_results.append({'title': rec_title})

    print("Successfully gathered list arrays:")
    print(f" -> KNN Recs Count: {len(knn_results)}")
    print(f" -> Similarity Recs Count: {len(sim_results)}")
    print(f" -> Implicit ALS Recs Count: {len(implicit_results)}")

    model_results = {
        'KNN': knn_results,
        'CosineSimilarity': sim_results,
        'Implicit Matrix Factorization': implicit_results
    }

    ranking, overlap, weights = combine_recommendations(model_results, top_n=FINAL_TOP_N)

    print('\n3x3 JACCARD OVERLAP MATRIX')
    print(overlap.round(4).to_string())
    print('\nMODEL WEIGHTS:', {k: round(v, 4) for k, v in weights.items()})
    print('\nFINAL HYBRID TOP 10')
    print(ranking.to_string(float_format=lambda x: f'{x:.6f}'))