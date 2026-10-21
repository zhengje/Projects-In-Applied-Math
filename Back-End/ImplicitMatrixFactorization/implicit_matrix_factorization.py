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
    recomm_data = pd.read_csv('Data Cleaning and Explore/recommendationsF.csv')

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


def recommend_games(user_id, model, user_app_matrix, user_id_to_index_map, app_index_map, game_title, num_recommendations=10):
    if user_id not in user_id_to_index_map:
        print(f"User ID {user_id} not found in the dataset.")
        return
    
    user_index = user_id_to_index_map[user_id]
    ids, scores = model.recommend(user_index, user_app_matrix[user_index], N=num_recommendations,filter_already_liked_items=True)

    print(f"Recommendations for user {user_id}:")
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        app_id = app_index_map[idx]
        title = game_title.get(app_id, f"App ID: {app_id}")
        print(f"{rank:2d}. {title[:45]:<45} (App ID: {app_id}), Score: {score:.4f}")

def similar_games(app_id, model, app_id_to_index_map, app_index_map, game_title, N=5):
    #Find similar games based on the trained model
    if app_id not in app_id_to_index_map:
        print(f"App ID {app_id} not found in the dataset.")
        return

    item_index = app_id_to_index_map[app_id]
    ids, scores = model.similar_items(item_index, N=N+1)

    title = game_title.get(app_id, f"App ID: {app_id}")
    print(f"Games similar to '{title}' (App ID: {app_id}):")
    for rank, (idx, score) in enumerate(zip(ids[1:], scores[1:]), start=1):  # Skip the first item as it is the same game
        similar_app_id = app_index_map[idx]
        similar_title = game_title.get(similar_app_id, f"App ID: {similar_app_id}")
        print(f"{rank:2d}. {similar_title[:45]:<45} (App ID: {similar_app_id}), Score: {score:.4f}")

if __name__ == "__main__":
    model, user_app_matrix, user_id_to_index_map, app_index_map, app_id_to_index_map, game_title = implicit_mf()

    # Example usage
    #     recommend_games(
    #     user_id=51580, 
    #     model=model, 
    #     user_app_matrix=user_app_matrix, 
    #     user_id_to_index_map=user_id_to_index_map, 
    #     app_index_map=app_index_map, 
    #     game_title=game_title, 
    #     num_recommendations=10)
    similar_games(
        app_id=975370, 
        model=model, 
        app_id_to_index_map= app_id_to_index_map,
        app_index_map=app_index_map, 
        game_title=game_title, 
        N=10)
    
#Output:
'''
Recommendations for user 51580:
 1. DOOM Eternal                                  (App ID: 782330), Score: 0.2669
 2. Factorio                                      (App ID: 427520), Score: 0.2080
 3. Deep Rock Galactic                            (App ID: 548430), Score: 0.2044
 4. Halo: The Master Chief Collection             (App ID: 976730), Score: 0.1668
 5. Killing Floor 2                               (App ID: 232090), Score: 0.1329
 6. Hades                                         (App ID: 1145360), Score: 0.1257
 7. Call of Duty®: Black Ops III                  (App ID: 311210), Score: 0.1209
 8. Elite Dangerous                               (App ID: 359320), Score: 0.1137
 9. METAL GEAR SOLID V: THE PHANTOM PAIN          (App ID: 287700), Score: 0.1119
10. Halo Infinite                                 (App ID: 1240440), Score: 0.1117
Games similar to 'Dwarf Fortress' (App ID: 975370):
 1. Avorion                                       (App ID: 445220), Score: 0.7748
 2. Going Medieval                                (App ID: 1029780), Score: 0.7656
 3. Hardspace: Shipbreaker                        (App ID: 1161580), Score: 0.7646
 4. Against the Storm                             (App ID: 1336490), Score: 0.7644
 5. Stellaris                                     (App ID: 281990), Score: 0.7386
 6. Timberborn                                    (App ID: 1062090), Score: 0.7333
 7. Battle Brothers                               (App ID: 365360), Score: 0.7313
 8. X4: Foundations                               (App ID: 392160), Score: 0.7241
 9. BATTLETECH                                    (App ID: 637090), Score: 0.6930
10. RimWorld                                      (App ID: 294100), Score: 0.6789                              (App ID: 1022980), Score: 0.9187
'''
