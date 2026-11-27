import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from tqdm import tqdm

MODEL_FILE = "gnn_recommender.pt"
MAPPINGS_FILE = "gnn_mappings.pkl"
EMBEDDING_DIM = 64


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# 3-Hop LightGCN Model for Bipartite User-Game Graph
class LightBipartiteGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LGConv()
        self.conv2 = LGConv()
        self.conv3 = LGConv()

    def forward(self, user_x, game_x, edge_index, edge_weight=None):
        num_users = user_x.size(0)
        num_games = game_x.size(0)

        # Combine user and game embeddings into a single bipartite node matrix
        x_0 = torch.cat([user_x, game_x], dim=0)

        # Shift game indices by num_users and build undirected edges for LGConv
        u = edge_index[0]
        g = edge_index[1] + num_users
        bi_edge_index = torch.cat([
            torch.stack([u, g], dim=0),
            torch.stack([g, u], dim=0)
        ], dim=1)

        bi_edge_weight = None
        if edge_weight is not None:
            bi_edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

        # --- 3-Hop LightGCN Propagation (No linear weights or ReLU distortion) ---
        x_1 = self.conv1(x_0, bi_edge_index, bi_edge_weight)
        x_2 = self.conv2(x_1, bi_edge_index, bi_edge_weight)
        x_3 = self.conv3(x_2, bi_edge_index, bi_edge_weight)

        # Include Layer 0 to preserve direct game identity alongside multi-hop signals
        x_final = (x_0 + x_1 + x_2 + x_3) / 4.0

        u_final, g_final = torch.split(x_final, [num_users, num_games], dim=0)
        return u_final, g_final


def train_and_save_artifacts():
    print("No saved model found. Processing CSV files and training LightGCN model...")

    df_subjective = pd.read_csv("Data_Subjective.csv")
    df_fixed = pd.read_csv("Data_Objective.csv")

    # Optional: Filter out negative reviews if a recommendation/sentiment column exists
    for rec_col in ['is_recommended', 'recommended', 'positive']:
        if rec_col in df_subjective.columns:
            df_subjective = df_subjective[df_subjective[rec_col].astype(bool)]
            break

    df_subjective['app_id'] = df_subjective['app_id'].astype(int)
    df_fixed['app_id'] = df_fixed['app_id'].astype(int)

    unique_users = sorted(df_subjective['user_id'].unique())
    unique_apps = sorted(df_subjective['app_id'].unique())

    user_map = {user: i for i, user in enumerate(unique_users)}
    app_map = {app: i for i, app in enumerate(unique_apps)}

    title_to_app_id = {}
    app_id_to_title = {}
    for _, row in df_fixed.iterrows():
        raw_id = int(row['app_id'])
        clean_title = str(row['title']).strip().lower()
        title_to_app_id[clean_title] = raw_id
        app_id_to_title[raw_id] = str(row['title'])

    # Build bipartite edge index
    user_indices = df_subjective['user_id'].map(user_map).to_numpy()
    app_indices = df_subjective['app_id'].map(app_map).to_numpy()
    edge_index = torch.tensor(np.vstack([user_indices, app_indices]), dtype=torch.long)

    # Optional: Use log-scaled playtime as edge weights if a playtime/hours column exists
    edge_weight = None
    for time_col in ['hours', 'playtime', 'playtime_forever']:
        if time_col in df_subjective.columns:
            raw_hours = pd.to_numeric(df_subjective[time_col], errors='coerce').fillna(0.0).clip(lower=0.0).to_numpy()
            # Normalize around mean 1.0 so degree normalization stays well-scaled
            weights = np.log1p(raw_hours) + 1.0
            weights = weights / weights.mean()
            edge_weight = torch.tensor(weights, dtype=torch.float32)
            break

    # Popularity-smoothed negative sampling distribution: P(i) ~ freq(i)^0.75
    app_counts = np.bincount(app_indices, minlength=len(unique_apps)).astype(np.float64)
    neg_sampling_probs = np.power(np.maximum(app_counts, 1.0), 0.75)
    neg_sampling_probs /= neg_sampling_probs.sum()
    neg_sampling_probs = torch.tensor(neg_sampling_probs, dtype=torch.float32)

    # Free DataFrames before training to save RAM
    del df_subjective, df_fixed

    # Initialize embeddings and LightGCN model
    user_emb = torch.nn.Embedding(len(unique_users) + 1, EMBEDDING_DIM)
    game_emb = torch.nn.Embedding(len(unique_apps), EMBEDDING_DIM)
    torch.nn.init.normal_(user_emb.weight, std=0.1)
    torch.nn.init.normal_(game_emb.weight, std=0.1)

    model = LightBipartiteGNN()

    # Optimize initial embeddings directly (LightGCN has no internal linear weights)
    optimizer = torch.optim.Adam(
        list(user_emb.parameters()) + list(game_emb.parameters()),
        lr=0.01
    )

    model.train()
    epochs = 200
    batch_size = 65536
    reg_weight = 1e-4
    num_edges = edge_index.size(1)

    loss_history = []
    best_loss = float("inf")
    patience = 10
    min_delta = 1e-4
    wait = 0

    print(f"Training LightGCN for up to {epochs} epochs (Total edges: {num_edges})...")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_edges)
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        u_final, g_final = model(user_emb.weight, game_emb.weight, edge_index, edge_weight)

        u_det = u_final.detach().requires_grad_(True)
        g_det = g_final.detach().requires_grad_(True)

        with tqdm(range(0, num_edges, batch_size), desc=f"Epoch {epoch:03d}/{epochs}", leave=False) as batch_bar:
            for start in batch_bar:
                batch_idx = perm[start : start + batch_size]
                pos_src = edge_index[0, batch_idx]
                pos_dst = edge_index[1, batch_idx]

                # Popularity-weighted negative sampling reduces mainstream bias
                neg_dst = torch.multinomial(neg_sampling_probs, batch_idx.size(0), replacement=True)

                pos_scores = (u_det[pos_src] * g_det[pos_dst]).sum(dim=-1)
                neg_scores = (u_det[pos_src] * g_det[neg_dst]).sum(dim=-1)

                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                # L2 regularization on initial Layer-0 embeddings
                reg_loss = reg_weight * (
                    user_emb.weight[pos_src].norm(2).pow(2) +
                    game_emb.weight[pos_dst].norm(2).pow(2) +
                    game_emb.weight[neg_dst].norm(2).pow(2)
                ) / batch_idx.size(0)

                loss = bpr_loss + reg_loss
                (loss * batch_idx.size(0) / num_edges).backward()

                total_loss += bpr_loss.item()
                num_batches += 1
                batch_bar.set_postfix(bpr_loss=f"{bpr_loss.item():.4f}")

        torch.autograd.backward([u_final, g_final], [u_det.grad, g_det.grad])
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        print(f" Epoch {epoch:03d}/{epochs} completed | Avg BPR Loss: {avg_loss:.4f}")

        # Early stopping & best checkpoint saving
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            wait = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'user_emb_state_dict': user_emb.state_dict(),
                'game_emb_state_dict': game_emb.state_dict(),
            }, MODEL_FILE)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best Loss: {best_loss:.4f})")
                break

    mappings = {
        'user_map': user_map,
        'app_map': app_map,
        'title_to_app_id': title_to_app_id,
        'app_id_to_title': app_id_to_title,
        'num_users': len(unique_users),
        'num_apps': len(unique_apps),
        'train_edge_index': edge_index,
        'train_edge_weight': edge_weight,
        'loss_history': loss_history,
    }
    with open(MAPPINGS_FILE, 'wb') as f:
        pickle.dump(mappings, f)

    print("Setup complete! Best model weights and mappings saved to disk.\n")


@torch.inference_mode()
def get_recommendations(input_game_names, top_k=10):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(MAPPINGS_FILE):
        train_and_save_artifacts()

    with open(MAPPINGS_FILE, 'rb') as f:
        mappings = pickle.load(f)

    app_map = mappings['app_map']
    title_to_app_id = mappings['title_to_app_id']
    app_id_to_title = mappings['app_id_to_title']
    num_users = mappings['num_users']
    num_apps = mappings['num_apps']

    new_user_idx = num_users
    played_game_indices = []
    user_indices = []
    app_indices = []

    for game_name in input_game_names:
        clean_name = game_name.strip().lower()
        if clean_name in title_to_app_id:
            raw_id = title_to_app_id[clean_name]
            if raw_id in app_map:
                game_graph_idx = app_map[raw_id]
                user_indices.append(new_user_idx)
                app_indices.append(game_graph_idx)
                played_game_indices.append(game_graph_idx)
            else:
                print(f"Skipping '{game_name}' - Not present in training data.")
        else:
            print(f"Warning: '{game_name}' could not be matched.")

    if not played_game_indices:
        print("No games match.")
        return []

    model = LightBipartiteGNN()
    user_emb = torch.nn.Embedding(num_users + 1, EMBEDDING_DIM)
    game_emb = torch.nn.Embedding(num_apps, EMBEDDING_DIM)

    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    user_emb.load_state_dict(checkpoint['user_emb_state_dict'])
    game_emb.load_state_dict(checkpoint['game_emb_state_dict'])
    model.eval()

    new_edges = torch.tensor([user_indices, app_indices], dtype=torch.long)
    full_inference_edge_index = torch.cat([mappings['train_edge_index'], new_edges], dim=1)

    full_inference_edge_weight = None
    if mappings.get('train_edge_weight') is not None:
        new_weights = torch.ones(len(user_indices), dtype=torch.float32)
        full_inference_edge_weight = torch.cat([mappings['train_edge_weight'], new_weights], dim=0)

    u_x = user_emb.weight.clone()
    u_x[new_user_idx] = 0.0
    g_x = game_emb.weight

    target_user_embeddings, all_game_embeddings = model(
        u_x, g_x, full_inference_edge_index, full_inference_edge_weight
    )

    # L2-normalize embeddings to eliminate residual popularity magnitude bias
    all_game_embeddings = F.normalize(all_game_embeddings, p=2, dim=-1)
    target_user_vector = F.normalize(target_user_embeddings[new_user_idx], p=2, dim=-1)

    scores = torch.matmul(all_game_embeddings, target_user_vector)
    scores[played_game_indices] = -float('inf')

    k = min(top_k, num_apps)
    top_indices = torch.topk(scores, k=k).indices.tolist()
    inverse_app_map = {v: k for k, v in app_map.items()}

    results = []
    for idx in top_indices:
        original_app_id = inverse_app_map[idx]
        title = app_id_to_title.get(original_app_id)
        if title:
            results.append({'title': title})

    return results