import os
# Limit BLAS threadpool contention for maximum ALS training speed
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit

def implicit_mf():
    recomm_data = pd.read_csv("Back-End/Methods/Data_Subjective.csv")

    #Load game titles from app_id
    try:
        game_data = pd.read_csv('Data Cleaning and Explore/Data_FixedF.csv')  # adjust path if inside another directory
        game_title = dict(zip(game_data['app_id'], game_data['title']))
    except Exception:
        game_title = {}

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
    return model, user_app_matrix, user_id_to_index_map, app_index_map, app_id_to_index_map, game_title

def similar_games_multi(app_ids, model, app_id_to_index_map, app_index_map, game_title, N=10):
    valid_app_ids = [id for id in app_ids if id in app_id_to_index_map]

    if not valid_app_ids:
        print("None of the App IDs provided were found in the dataset.")
        return

    id_indices = [app_id_to_index_map[id] for id in valid_app_ids]

    #Create CSR matrix 
    cols = id_indices
    rows = [0]* len(cols)
    data = [1.0] * len(cols)

    user_items = csr_matrix((data, (rows, cols)), shape=(1, model.item_factors.shape[0]))

    ids, scores = model.recommend(
        userid =0,
        user_items = user_items,
        N=N,
        filter_already_liked_items = True,
        recalculate_user = True
    )

    input_names = [game_title.get(id, f"App ID: {id}") for id in valid_app_ids]
    print(f"Recommended games [{', '.join(input_names)}]")

    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        rec_app_id = app_index_map[idx]
        rec_title = game_title.get(rec_app_id, f"App ID: {rec_app_id}")
        print(f"{rank:2d}.{rec_title[:45]:<45} (App ID: {rec_app_id}, Score: {score:.4f})")


if __name__ == "__main__":
    model, user_app_matrix, user_id_to_index_map, app_index_map, app_id_to_index_map, game_title = implicit_mf()

    # Pass a list of multiple App IDs (e.g., Dwarf Fortress + Factorio + RimWorld)
    input_app_ids = [975370, 427520, 294100]

    similar_games_multi(
        app_ids=input_app_ids,
        model=model,
        app_id_to_index_map=app_id_to_index_map,
        app_index_map=app_index_map,
        game_title=game_title,
        N=10
    )

#Output
# Recommended games [Dwarf Fortress, Factorio, RimWorld]
#  1.Space Engineers                               (App ID: 244850, Score: 0.1753
#  2.Kerbal Space Program                          (App ID: 220200, Score: 0.1584
#  3.Cities: Skylines                              (App ID: 255710, Score: 0.1364
#  4.Oxygen Not Included                           (App ID: 457140, Score: 0.1300
#  5.Path of Exile                                 (App ID: 238960, Score: 0.1150
#  6.Satisfactory                                  (App ID: 526870, Score: 0.1091
#  7.Kenshi                                        (App ID: 233860, Score: 0.1084
#  8.Prison Architect                              (App ID: 233450, Score: 0.1040
#  9.Project Zomboid                               (App ID: 108600, Score: 0.1011
# 10.Dyson Sphere Program                          (App ID: 1366540, Score: 0.0996