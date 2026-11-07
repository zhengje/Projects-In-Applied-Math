import os
# Limit BLAS threadpool contention for maximum ALS training speed
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import pickle

MODEL_PATH = "implicit_mf_model.pkl"
def implicit_mf(force_retrain=False):

    #Check if a saved model exists and load it if available
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("Loading saved model and metadata from disk...")
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
        return (
            artifacts['model'], 
            None,
            artifacts['user_id_to_index_map'], 
            artifacts['app_index_map'], 
            artifacts['app_id_to_index_map'], 
            artifacts['game_title'], 
            artifacts['title_to_app_id_map']
        )

    print("Training new implicit matrix factorization model...")
    recomm_data = pd.read_csv('Back-End/Methods/Data_Subjective.csv')


    # Load game titles from app_id and build reverse mapping dictionary
    title_to_app_id_map = {}
    game_title = {}
    try:
        game_data = pd.read_csv('Data Cleaning and Explore/Data_FixedF.csv')
        game_title = dict(zip(game_data['app_id'], game_data['title']))
        
        # Build normalized lowercase title -> app_id lookup
        for app_id, title in game_title.items():
            if pd.notna(title):
                title_to_app_id_map[str(title).strip().lower()] = app_id
    except Exception as e:
        print(f"Warning: Could not load game titles. Error: {e}")

    recomm_data['recommendations'] = np.log1p(recomm_data['hours'].clip(lower=0)) + (recomm_data['is_recommended'].astype(int) * 1.0)
   
    #Filter out rows with recommendations <= 0
    recomm_data = recomm_data[recomm_data['recommendations'] > 0]

    #Create a sparse matrix for the implicit matrix factorization model
    user_ids = recomm_data['user_id'].astype("category")
    app_ids = recomm_data['app_id'].astype("category")

    user_index_map = dict(enumerate(user_ids.cat.categories))
    app_index_map = dict(enumerate(app_ids.cat.categories))

    user_id_to_index_map = {v: k for k, v in user_index_map.items()}
    app_id_to_index_map = {v: k for k, v in app_index_map.items()}

    recomm_data['user_index'] = user_ids.cat.codes
    recomm_data['app_index'] = app_ids.cat.codes

    user_app_matrix = csr_matrix((recomm_data['recommendations'], (recomm_data['user_index'], recomm_data['app_index'])), shape=(len(user_ids.cat.categories), len(app_ids.cat.categories)))

    #Initialize the implicit matrix factorization model
    model = implicit.als.AlternatingLeastSquares(factors=32, iterations=15, regularization=0.05, random_state=42,use_cg=True,use_native=True)

    #Train the model on the user-app interaction matrix
    model.fit(user_app_matrix)

    #Save trained model & maps to disk
    artifacts = {
        'model': model,
        'user_id_to_index_map': user_id_to_index_map,
        'app_index_map': app_index_map,
        'app_id_to_index_map': app_id_to_index_map,
        'game_title': game_title,
        'title_to_app_id_map': title_to_app_id_map
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Model saved to {MODEL_PATH}")

    return model, user_app_matrix, user_id_to_index_map, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map

def similar_games_multi(game_names, model, app_id_to_index_map, app_index_map, game_title, title_to_app_id_map, N=10):
    valid_app_ids = []
    resolved_titles = []

    # Map input string names to app_ids
    for name in game_names:
        search_name = name.strip().lower()
        app_id = title_to_app_id_map.get(search_name)

        # Fallback to partial match if exact name match fails
        if app_id is None or app_id not in app_id_to_index_map:
            matches = [t for t in title_to_app_id_map if search_name in t]
            matched_id = None
            for match in matches:
                candidate_id = title_to_app_id_map[match]
                if candidate_id in app_id_to_index_map:
                    matched_id = candidate_id
                    break

            if matched_id is not None:
                app_id = matched_id
                print(f"Using match: '{game_title[app_id]}' for search '{name}'")
            else:
                print(f"Skipping: Game '{name}' not found or has no user interaction data.")
                continue

        valid_app_ids.append(app_id)
        resolved_titles.append(game_title.get(app_id, str(app_id)))

    if not valid_app_ids:
        print("None of the provided game names could be matched to trained matrix items.")
        return
    id_indices = [app_id_to_index_map[id] for id in valid_app_ids]

    #Create CSR matrix 
    cols = id_indices
    rows = [0]* len(cols)
    data = [1.0] * len(cols)

    user_items = csr_matrix((data, (rows, cols)), shape=(1, model.item_factors.shape[0]))

    try:
        ids, scores = model.recommend(
            userid=0,
            user_items=user_items,
            N=N,
            filter_already_liked_items=True,
            recalculate_user=True
    )
    except TypeError:
        ids, scores = model.recommend(
        userid=0,
        user_items=user_items,
        N=N,
        filter_already_liked_items=True
    )

    if isinstance(ids, np.ndarray) and ids.ndim > 1:
        ids = ids[0]
        scores = scores[0]

    input_names = [game_title.get(id, f"App ID: {id}") for id in valid_app_ids]
    print(f"Recommended games [{', '.join(input_names)}]")

    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        rec_app_id = app_index_map[idx]
        rec_title = game_title.get(rec_app_id, f"App ID: {rec_app_id}")
        print(f"{rank:2d}.{rec_title[:45]:<45} (App ID: {rec_app_id}, Score: {score:.4f})")


if __name__ == "__main__":
    model, user_app_matrix, user_id_to_index_map, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = implicit_mf()

    # Pass a list of multiple App IDs (e.g., Dwarf Fortress + Factorio + RimWorld)
    input_games = ["Omori", "Resident Evil"]

    similar_games_multi(
        game_names=input_games,
        model=model,
        app_id_to_index_map=app_id_to_index_map,
        app_index_map=app_index_map,
        game_title=game_title,
        title_to_app_id_map=title_to_app_id_map,
        N=10
    )

#Output

# Recommended games [Dwarf Fortress, Factorio, RimWorld]
#  1.Space Engineers                               (App ID: 244850, Score: 0.1753)
#  2.Kerbal Space Program                          (App ID: 220200, Score: 0.1584)
#  3.Cities: Skylines                              (App ID: 255710, Score: 0.1364)
#  4.Oxygen Not Included                           (App ID: 457140, Score: 0.1300)
#  5.Path of Exile                                 (App ID: 238960, Score: 0.1150)
#  6.Satisfactory                                  (App ID: 526870, Score: 0.1091)
#  7.Kenshi                                        (App ID: 233860, Score: 0.1084)
#  8.Prison Architect                              (App ID: 233450, Score: 0.1040)
#  9.Project Zomboid                               (App ID: 108600, Score: 0.1011)
# 10.Dyson Sphere Program                          (App ID: 1366540, Score: 0.0996)                    

# Recommended games [OMORI, Resident Evil]
#  1.Among Us                                      (App ID: 945360, Score: 0.0195)
#  2.Hollow Knight                                 (App ID: 367520, Score: 0.0151)
#  3.Undertale                                     (App ID: 391540, Score: 0.0141)
#  4.NieR:Automata™                                (App ID: 524220, Score: 0.0140)
#  5.The Binding of Isaac: Rebirth                 (App ID: 250900, Score: 0.0140)
#  6.Celeste                                       (App ID: 504230, Score: 0.0137)
#  7.Inscryption                                   (App ID: 1092790, Score: 0.0124)
#  8.Resident Evil 2                               (App ID: 883710, Score: 0.0123)
#  9.DOOM Eternal                                  (App ID: 782330, Score: 0.0119)
# 10.METAL GEAR RISING: REVENGEANCE                (App ID: 235460, Score: 0.0115)