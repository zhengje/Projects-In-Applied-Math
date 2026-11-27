import sys
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

import FinalKNNMethod as knn_mod
import FinalSimilarityMatrixMethod as sim_mod
import FinalIMFMethod as imf_mod
import FinalGNNMethod as gnn_mod
from Combined4Models import combine_recommendations


def _chunked_sage_conv(conv_layer, x_tuple, edge_index, num_target_nodes, chunk_size=2_000_000):
    """Runs SAGEConv in edge chunks to prevent 30M-edge scatter_add RuntimeError: bad allocation."""
    x_src, x_dst = x_tuple
    device = x_src.device
    hidden_dim = x_src.size(1)

    aggr_sum = torch.zeros((num_target_nodes, hidden_dim), device=device)
    deg = torch.zeros((num_target_nodes, 1), device=device)

    num_edges = edge_index.size(1)
    for start in range(0, num_edges, chunk_size):
        src_idx = edge_index[0, start : start + chunk_size]
        dst_idx = edge_index[1, start : start + chunk_size]

        aggr_sum.index_add_(0, dst_idx, x_src[src_idx])
        deg.index_add_(0, dst_idx, torch.ones((src_idx.size(0), 1), device=device))

    mean_aggr = aggr_sum / deg.clamp(min=1.0)
    out = conv_layer.lin_l(mean_aggr)
    if getattr(conv_layer, "root_weight", False) and hasattr(conv_layer, "lin_r"):
        out = out + conv_layer.lin_r(x_dst)
    return out


def load_gnn_artifacts():
    """Loads the trained GNN and computes final game embeddings in memory-safe chunks."""
    if not os.path.exists(gnn_mod.MODEL_FILE) or not os.path.exists(gnn_mod.MAPPINGS_FILE):
        gnn_mod.train_and_save_artifacts()

    with open(gnn_mod.MAPPINGS_FILE, "rb") as f:
        mappings = pickle.load(f)

    device = getattr(gnn_mod, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_apps = mappings["num_apps"]

    model = gnn_mod.SimpleBipartiteGNN(hidden_dim=gnn_mod.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(gnn_mod.MODEL_FILE, map_location=device)
    emb_rows = checkpoint["user_emb_state_dict"]["weight"].shape[0]

    user_emb = torch.nn.Embedding(emb_rows, gnn_mod.EMBEDDING_DIM).to(device)
    game_emb = torch.nn.Embedding(num_apps, gnn_mod.EMBEDDING_DIM).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    user_emb.load_state_dict(checkpoint["user_emb_state_dict"])
    game_emb.load_state_dict(checkpoint["game_emb_state_dict"])
    model.eval()

    train_edges = mappings["train_edge_index"].to(device)
    rev_edges = train_edges.flip(0)
    inverse_app_map = {v: k for k, v in mappings["app_map"].items()}

    num_u_nodes = user_emb.weight.size(0)
    num_g_nodes = game_emb.weight.size(0)

    # Chunked 3-hop forward pass (uses < 100MB RAM instead of crashing on 30.8M edges)
    with torch.no_grad():
        u_0 = user_emb.weight
        g_0 = game_emb.weight

        g_1 = _chunked_sage_conv(model.conv1_u2g, (u_0, g_0), train_edges, num_g_nodes).relu()
        u_1 = _chunked_sage_conv(model.conv1_g2u, (g_0, u_0), rev_edges, num_u_nodes).relu()

        g_2 = _chunked_sage_conv(model.conv2_u2g, (u_1, g_1), train_edges, num_g_nodes).relu()
        u_2 = _chunked_sage_conv(model.conv2_g2u, (g_1, u_1), rev_edges, num_u_nodes).relu()

        g_3 = _chunked_sage_conv(model.conv3_u2g, (u_2, g_2), train_edges, num_g_nodes).relu()

        all_game_embeddings = ((g_1 + g_2 + g_3) / 3.0).detach()

    # Free large tensors from RAM immediately
    del train_edges, rev_edges, user_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model,
        "all_game_embeddings": all_game_embeddings,
        "base_game_weights": game_emb.weight.detach(),
        "mappings": mappings,
        "inverse_app_map": inverse_app_map,
        "device": device,
    }


def predict_gnn_fast(gnn_ctx, input_game_names, top_k=100):
    """Sub-millisecond cold-start inference using only the user's played games."""
    mappings = gnn_ctx["mappings"]
    app_map = mappings["app_map"]
    title_to_app_id = mappings["title_to_app_id"]
    app_id_to_title = mappings["app_id_to_title"]
    num_apps = mappings["num_apps"]
    device = gnn_ctx["device"]

    played_game_indices = set()
    for game_name in input_game_names:
        clean_name = str(game_name).strip().lower()
        raw_id = title_to_app_id.get(clean_name)
        if raw_id is not None and raw_id in app_map:
            played_game_indices.add(app_map[raw_id])

    if not played_game_indices:
        return []

    played_list = sorted(played_game_indices)
    model = gnn_ctx["model"]
    all_game_embeddings = gnn_ctx["all_game_embeddings"]

    with torch.no_grad():
        num_played = len(played_list)
        local_game_idx = torch.arange(num_played, device=device)
        local_user_idx = torch.zeros(num_played, dtype=torch.long, device=device)

        local_u2g = torch.stack([local_user_idx, local_game_idx], dim=0)
        local_g2u = torch.stack([local_game_idx, local_user_idx], dim=0)

        u_0 = torch.zeros((1, gnn_mod.EMBEDDING_DIM), device=device)
        g_0 = gnn_ctx["base_game_weights"][played_list]

        g_1 = model.conv1_u2g((u_0, g_0), local_u2g).relu()
        u_1 = model.conv1_g2u((g_0, u_0), local_g2u).relu()

        g_2 = model.conv2_u2g((u_1, g_1), local_u2g).relu()
        u_2 = model.conv2_g2u((g_1, u_1), local_g2u).relu()

        # Fixed: conv3_g2u maps games -> user
        u_3 = model.conv3_g2u((g_2, u_2), local_g2u).relu()

        target_user_vector = ((u_1 + u_2 + u_3) / 3.0).squeeze(0)

        scores = torch.matmul(all_game_embeddings, target_user_vector)
        scores[played_list] = -float("inf")

        k = min(top_k, num_apps)
        top_indices = torch.topk(scores, k=k).indices.tolist()

    results = []
    for idx in top_indices:
        original_app_id = gnn_ctx["inverse_app_map"][idx]
        title = app_id_to_title.get(original_app_id)
        if title:
            results.append({"title": title})

    return results


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
    gnn_ctx = load_gnn_artifacts()

    # 2. Build valid title sets for each model
    knn_titles = set(knn_mod.df["title"].astype(str).str.lower().str.strip())
    sim_titles = set(sim_mod.df["title"].astype(str).str.lower().str.strip())
    imf_titles = {
        str(game_title[aid]).strip().lower()
        for aid in app_id_to_index_map
        if aid in game_title and pd.notna(game_title[aid])
    }

    gnn_mappings = gnn_ctx["mappings"]
    gnn_titles = {
        str(gnn_mappings["app_id_to_title"][aid]).strip().lower()
        for aid in gnn_mappings["app_map"]
        if aid in gnn_mappings["app_id_to_title"] and pd.notna(gnn_mappings["app_id_to_title"][aid])
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

    print(f"Evaluating Jaccard Hybrid (and all 4 base models) on {len(eligible_users)} users (80/20 split, Top-{top_k})...")

    # Track metrics for Hybrid + each individual model on the exact same users
    model_names = ["Jaccard Hybrid RRF", "KNN", "CosineSimilarity", "Implicit MF", "GNN"]
    metrics = {m: {"hits": [], "precision": [], "recall": [], "ndcg": []} for m in model_names}
    avg_weights = {"KNN": [], "CosineSimilarity": [], "Implicit Matrix Factorization": [], "GNN": []}

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

        # --- Model 4 (GNN) ---
        gnn_results = predict_gnn_fast(gnn_ctx, train_games, top_k=candidates_per_model)

        # Skip user if none of the models returned candidates
        if not (knn_results or sim_results or implicit_results or gnn_results):
            continue

        # --- Combine via Jaccard-Weighted RRF ---
        model_results = {
            "KNN": knn_results,
            "CosineSimilarity": sim_results,
            "Implicit Matrix Factorization": implicit_results,
            "GNN": gnn_results,
        }
        ranking_df, _, weights = combine_recommendations(model_results, top_n=top_k)

        for k, v in weights.items():
            avg_weights[k].append(v)

        # Extract Top-K titles for Hybrid and each standalone model
        hybrid_titles = [str(t).lower().strip() for t in ranking_df["Game"].tolist()]
        knn_top_titles = [r["title"].lower().strip() for r in knn_results[:top_k]]
        sim_top_titles = [r["title"].lower().strip() for r in sim_results[:top_k]]
        imf_top_titles = [r["title"].lower().strip() for r in implicit_results[:top_k]]
        gnn_top_titles = [r["title"].lower().strip() for r in gnn_results[:top_k]]

        record_metrics("Jaccard Hybrid RRF", hybrid_titles, test_games)
        record_metrics("KNN", knn_top_titles, test_games)
        record_metrics("CosineSimilarity", sim_top_titles, test_games)
        record_metrics("Implicit MF", imf_top_titles, test_games)
        record_metrics("GNN", gnn_top_titles, test_games)

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


# 4x4 JACCARD OVERLAP MATRIX
#                                   KNN  CosineSimilarity  Implicit Matrix Factorization     GNN
# KNN                            1.0000            0.6260                         0.1236  0.0471
# CosineSimilarity               0.6260            1.0000                         0.1364  0.0582
# Implicit Matrix Factorization  0.1236            0.1364                         1.0000  0.1236
# GNN                            0.0471            0.0582                         0.1236  1.0000

# MODEL WEIGHTS: {'KNN': np.float64(0.2106), 'CosineSimilarity': np.float64(0.2079), 'Implicit Matrix Factorization': np.float64(0.2735), 'GNN': np.float64(0.308)}

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

# ================================================================================
# 80/20 BENCHMARK: JACCARD HYBRID RRF VS. INDIVIDUAL MODELS
# ================================================================================
#              Model  Users Evaluated  HitRate@10  Precision@10  Recall@10  NDCG@10
# Jaccard Hybrid RRF             1000       0.454        0.0608     0.1258   0.1076
#                KNN             1000       0.222        0.0243     0.0574   0.0475
#   CosineSimilarity             1000       0.397        0.0486     0.1029   0.0924
#        Implicit MF             1000       0.567        0.0854     0.1584   0.1465
#                GNN             1000       0.186        0.0205     0.0399   0.0364

# Average Jaccard Independence Weights Across Users:
#  - KNN: 0.2216
#  - CosineSimilarity: 0.2167
#  - Implicit Matrix Factorization: 0.2719
#  - GNN: 0.2898