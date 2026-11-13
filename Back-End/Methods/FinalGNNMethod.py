import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import SAGEConv

MODEL_FILE = "gnn_recommender.pt"
MAPPINGS_FILE = "gnn_mappings.pkl"
EMBEDDING_DIM = 16

# To only have one fixed output for the same input, we set the random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# 3-hop GNN Model for bipartite user-game graph

class SimpleBipartiteGNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Layer 1: Direct interactions (1-hop)
        self.conv1_u2g = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.conv1_g2u = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

        # Layer 2: Similar users / co-played games (2-hop)
        self.conv2_u2g = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.conv2_g2u = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

        # Layer 3: Broader taste clusters (3-hop)
        self.conv3_u2g = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        self.conv3_g2u = SAGEConv((hidden_dim, hidden_dim), hidden_dim)

    def forward(self, user_x, game_x, edge_index):
        rev_edge_index = edge_index.flip(0)

        # --- Hop 1 ---
        g_1 = self.conv1_u2g((user_x, game_x), edge_index).relu()
        u_1 = self.conv1_g2u((game_x, user_x), rev_edge_index).relu()

        # --- Hop 2 ---
        g_2 = self.conv2_u2g((u_1, g_1), edge_index).relu()
        u_2 = self.conv2_g2u((g_1, u_1), rev_edge_index).relu()

        # --- Hop 3 ---
        g_3 = self.conv3_u2g((u_2, g_2), edge_index).relu()
        u_3 = self.conv3_g2u((g_2, u_2), rev_edge_index).relu()

        # Average all 3 layers to balance personal history with broader graph signals
        u_final = (u_1 + u_2 + u_3) / 3
        g_final = (g_1 + g_2 + g_3) / 3

        return u_final, g_final

    #Train and Save the model (Runs only once)
def train_and_save_artifacts():
    print("No saved model found. Processing CSV files and training model...")

    df_subjective = pd.read_csv("Data_Subjective.csv")
    df_fixed = pd.read_csv("Data_Objective.csv")

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

    # Create edge index for the bipartite graph
    user_indices = list(df_subjective['user_id'].map(user_map).values)
    app_indices = list(df_subjective['app_id'].map(app_map).values)
    edge_index = torch.tensor([user_indices, app_indices], dtype=torch.long)

    # Initialize embeddings and model
    user_emb = torch.nn.Embedding(len(unique_users) + 1, EMBEDDING_DIM)
    game_emb = torch.nn.Embedding(len(unique_apps), EMBEDDING_DIM)
    model = SimpleBipartiteGNN(hidden_dim=EMBEDDING_DIM)

    #Train the model
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + 
        list(user_emb.parameters()) + 
        list(game_emb.parameters()), lr=0.01
    )

    model.train()
    epochs = 30
    batch_size = 65536  # Process 65k interactions at a time instead of millions at once
    num_edges = edge_index.size(1)
    print(f"Training GNN for {epochs} epochs (Total edges: {num_edges})...")

    for epoch in range(1, epochs + 1):
        # Shuffle edge indices each epoch
        perm = torch.randperm(num_edges)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_edges, batch_size):
            optimizer.zero_grad()

            # Forward pass over the graph
            u_final, g_final = model(user_emb.weight, game_emb.weight, edge_index)

            # Slice a mini-batch of positive edges
            batch_idx = perm[start : start + batch_size]
            pos_src = edge_index[0, batch_idx]
            pos_dst = edge_index[1, batch_idx]

            pos_scores = (u_final[pos_src] * g_final[pos_dst]).sum(dim=-1)

            # Negative samples only for this mini-batch
            neg_dst = torch.randint(0, len(unique_apps), (batch_idx.size(0),))
            neg_scores = (u_final[pos_src] * g_final[neg_dst]).sum(dim=-1)

            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f" Epoch {epoch:02d}/{epochs} | Avg Loss: {total_loss / num_batches:.4f}")

            
    # Save model and mappings
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'user_emb_state_dict': user_emb.state_dict(),
        'game_emb_state_dict': game_emb.state_dict(),
    }
    torch.save(checkpoint, MODEL_FILE)

    mappings = {
        'user_map': user_map,
        'app_map': app_map,
        'title_to_app_id': title_to_app_id,
        'app_id_to_title': app_id_to_title,
        'num_users': len(unique_users),
        'num_apps': len(unique_apps),
        'train_edge_index': edge_index,
    }
    with open(MAPPINGS_FILE, 'wb') as f:
        pickle.dump(mappings, f)
        
    print("Setup complete! Model weights and mappings saved to disk.\n")

#Recommendation function

def get_recommendations(input_game_names, top_k=10):
    # Only trains once if saved files don't exist yet
    if not os.path.exists(MODEL_FILE) or not os.path.exists(MAPPINGS_FILE):
        train_and_save_artifacts()

    # Load pre-saved mappings
    with open(MAPPINGS_FILE, 'rb') as f:
        mappings = pickle.load(f)

    app_map = mappings['app_map']
    title_to_app_id = mappings['title_to_app_id']
    app_id_to_title = mappings['app_id_to_title']
    num_users = mappings['num_users']
    num_apps = mappings['num_apps']

    # Load pre-saved weights
    model = SimpleBipartiteGNN(hidden_dim=EMBEDDING_DIM)
    user_emb = torch.nn.Embedding(num_users + 1, EMBEDDING_DIM)
    game_emb = torch.nn.Embedding(num_apps, EMBEDDING_DIM)

    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    user_emb.load_state_dict(checkpoint['user_emb_state_dict'])
    game_emb.load_state_dict(checkpoint['game_emb_state_dict'])
    model.eval()

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

    if not user_indices:
        print("No games match.")
        return

    
    new_edges = torch.tensor([user_indices, app_indices], dtype=torch.long)
    full_inference_edge_index = torch.cat([mappings['train_edge_index'], new_edges], dim=1)

    with torch.no_grad():
        u_x = user_emb.weight.clone()
        u_x[new_user_idx] = 0.0
        g_x = game_emb.weight
        target_user_embeddings, all_game_embeddings = model(u_x, g_x, full_inference_edge_index)
        target_user_vector = target_user_embeddings[new_user_idx]

    scores = torch.matmul(all_game_embeddings, target_user_vector)
    scores[played_game_indices] = -float('inf')

    k = min(top_k, num_apps)
    top_indices = torch.topk(scores, k=k).indices.tolist()
    inverse_app_map = {v: k for k, v in app_map.items()}

    print(f"\nRecommendations generated based on your input list:")
    for pos, idx in enumerate(top_indices, 1):
        original_app_id = inverse_app_map[idx]
        title = app_id_to_title.get(original_app_id, f"App ID {original_app_id}")
        print(f" {pos}. {title}")

#Example usage
if __name__ == "__main__":
    input_list = [
        "Dwarf Fortress",
        "Factorio",
        "RimWorld"
    ]
    get_recommendations(input_list, top_k=10)