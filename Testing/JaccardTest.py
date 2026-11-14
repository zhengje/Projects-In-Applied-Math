import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import FinalKNNMethod as knn_mod
import FinalSimilarityMatrixMethod as sim_mod
import FinalIMFMethod as imf_mod
from Combined3ModelsTest import combine_recommendations


def evaluate_hybrid_jaccard_80_20(
    australian_csv_path,
    model_pkl_path=r"implicit_mf_model.pkl",
    candidates_per_model=100,
    top_k=10,
    min_user_games=5,      # Minimum number of games a user must have played to be included in the evaluation
    max_users=1000,        # Set to None to evaluate every eligible user
    only_played=True,      # True = keep only games where playtime_forever > 0
    seed=42
):
    rng = np.random.default_rng(seed)

    # 1. Load IMF model & mappings ONCE
    imf_mod.MODEL_PATH = model_pkl_path
    imf_model, _, _, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = imf_mod.implicit_mf()

    # 2. Build valid title sets for each model
    knn_titles = set(knn_mod.df["title"].astype(str).str.lower().str.strip())
    sim_titles = set(sim_mod.df["title"].astype(str).str.lower().str.strip())
    imf_titles = {
        str(game_title[aid]).strip().lower()
        for aid in app_id_to_index_map
        if aid in game_title and pd.notna(game_title[aid])
    }

    # Common catalog across all 3 models so every model can generate candidates & be scored fairly
    common_titles = knn_titles & sim_titles & imf_titles

    # 3. Load and filter Australian user interactions
    interactions = pd.read_csv(australian_csv_path)

    if only_played and "playtime_forever" in interactions.columns:
        interactions = interactions[interactions["playtime_forever"] > 0]

    interactions = interactions.drop_duplicates(subset=["user_id", "item_name"])
    interactions["norm_title"] = interactions["item_name"].astype(str).str.lower().str.strip()
    interactions = interactions[interactions["norm_title"].isin(common_titles)]

    # 4. Group games per user
    user_groups = {
        uid: grp["item_name"].tolist()
        for uid, grp in interactions.groupby("user_id")
        if len(grp) >= min_user_games
    }

    eligible_users = list(user_groups.keys())
    if max_users and len(eligible_users) > max_users:
        eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))

    print(f"Evaluating Jaccard Hybrid (and all 3 base models) on {len(eligible_users)} users (80/20 split, Top-{top_k})...")

    # Track metrics for Hybrid + each individual model on the exact same users
    model_names = ["Jaccard Hybrid RRF", "KNN", "CosineSimilarity", "Implicit MF"]
    metrics = {m: {"hits": [], "precision": [], "recall": [], "ndcg": []} for m in model_names}
    avg_weights = {"KNN": [], "CosineSimilarity": [], "Implicit Matrix Factorization": []}

    def record_metrics(model_key, rec_titles, test_set):
        hit_vec = [1 if t in test_set else 0 for t in rec_titles[:top_k]]
        num_hits = sum(hit_vec)

        metrics[model_key]["hits"].append(1.0 if num_hits > 0 else 0.0)
        metrics[model_key]["precision"].append(num_hits / top_k)
        metrics[model_key]["recall"].append(num_hits / len(test_set))

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(hit_vec))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_set), top_k)))
        metrics[model_key]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

    # 5. Perform 80/20 split per user
    for uid in eligible_users:
        games = np.array(user_groups[uid])
        shuffled = rng.permutation(games)

        split_idx = max(1, int(len(shuffled) * 0.8))
        train_games = shuffled[:split_idx].tolist()
        test_games = {g.lower().strip() for g in shuffled[split_idx:]}

        if not test_games:
            continue

        # --- Model 1 (KNN) ---
        _, knn_results = knn_mod.recommend_games_knn(train_games, knn_mod.df, top_n=candidates_per_model)
        if not isinstance(knn_results, list):
            knn_results = []

        # --- Model 2 (Cosine Similarity) ---
        _, sim_results = sim_mod.recommend_games(train_games, sim_mod.df, top_n=candidates_per_model)
        if not isinstance(sim_results, list):
            sim_results = []

        # --- Model 3 (Implicit MF) ---
        valid_app_ids = [
            title_to_app_id_map[g.strip().lower()]
            for g in train_games
            if g.strip().lower() in title_to_app_id_map
            and title_to_app_id_map[g.strip().lower()] in app_id_to_index_map
        ]

        implicit_results = []
        if valid_app_ids:
            cols = [app_id_to_index_map[aid] for aid in valid_app_ids]
            rows = [0] * len(cols)
            data = [1.0] * len(cols)
            user_items = csr_matrix((data, (rows, cols)), shape=(1, imf_model.item_factors.shape[0]))

            try:
                ids, _ = imf_model.recommend(
                    userid=0,
                    user_items=user_items,
                    N=candidates_per_model,
                    filter_already_liked_items=True,
                    recalculate_user=True
                )
            except TypeError:
                ids, _ = imf_model.recommend(
                    userid=0,
                    user_items=user_items,
                    N=candidates_per_model,
                    filter_already_liked_items=True
                )

            if isinstance(ids, np.ndarray) and ids.ndim > 1:
                ids = ids[0]

            for idx in ids:
                rec_title = game_title.get(app_index_map[idx])
                if rec_title:
                    implicit_results.append({"title": rec_title})

        # Skip user if none of the models returned candidates
        if not (knn_results or sim_results or implicit_results):
            continue

        # --- Combine via Jaccard-Weighted RRF ---
        model_results = {
            "KNN": knn_results,
            "CosineSimilarity": sim_results,
            "Implicit Matrix Factorization": implicit_results
        }
        ranking_df, _, weights = combine_recommendations(model_results, top_n=top_k)

        for k, v in weights.items():
            avg_weights[k].append(v)

        # Extract Top-K titles for Hybrid and each standalone model
        hybrid_titles = [str(t).lower().strip() for t in ranking_df["Game"].tolist()]
        knn_top_titles = [r["title"].lower().strip() for r in knn_results[:top_k]]
        sim_top_titles = [r["title"].lower().strip() for r in sim_results[:top_k]]
        imf_top_titles = [r["title"].lower().strip() for r in implicit_results[:top_k]]

        record_metrics("Jaccard Hybrid RRF", hybrid_titles, test_games)
        record_metrics("KNN", knn_top_titles, test_games)
        record_metrics("CosineSimilarity", sim_top_titles, test_games)
        record_metrics("Implicit MF", imf_top_titles, test_games)

    # 6. Print Side-by-Side Comparison Table
    summary_rows = []
    for m in model_names:
        summary_rows.append({
            "Model": m,
            "Users Evaluated": len(metrics[m]["hits"]),
            f"HitRate@{top_k}": round(float(np.mean(metrics[m]["hits"])), 4),
            f"Precision@{top_k}": round(float(np.mean(metrics[m]["precision"])), 4),
            f"Recall@{top_k}": round(float(np.mean(metrics[m]["recall"])), 4),
            f"NDCG@{top_k}": round(float(np.mean(metrics[m]["ndcg"])), 4),
        })

    results_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("80/20 BENCHMARK: JACCARD HYBRID RRF VS. INDIVIDUAL MODELS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\nAverage Jaccard Independence Weights Across Users:")
    for k, vals in avg_weights.items():
        if vals:
            print(f" - {k}: {np.mean(vals):.4f}")

    return results_df


if __name__ == "__main__":
    evaluate_hybrid_jaccard_80_20(
        australian_csv_path=r"australian_users_items_filtered.csv",
        model_pkl_path=r"implicit_mf_model.pkl",
        candidates_per_model=100,
        top_k=10,
        min_user_games=5,
        max_users=1000,
        only_played=True
    )

#HitRate@10: Out of 1000 users, 0.75 means 750 users had at least one of their hidden games recommended in the top 10.
#Precision@10: On average, 0.15 of the top 10 recommendations were matched to the hidden games across all users.
#Recall@10: On average, 0.30 of the hidden games were successfully recommended in the top 10 across all users.
#NormalizedDiscountedCumulativeGain@10: Measures ranking quality, with 1.0 being perfect ranking. Higher values mean the matched games appeared closer to Rank #1.