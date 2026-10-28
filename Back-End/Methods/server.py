from flask import Flask, jsonify, request
import pandas as pd
import FinalKNNMethod as KNN
import FinalSimilarityMatrixMethod as SM
# import implicit_matrix_factorization as IMF
app = Flask(__name__)


# Retrieve Game Information from Database (Search Bar)
def get_names():
    titles = []
    data = KNN.load_data()
    df = pd.DataFrame(data)

    for i in range(len(df)):
        temp = {}
        temp['Titles'] = df['title'][i]

        titles.append(temp)

    return titles

# Pass Names to Front End (search Bar)
@app.route("/game_names")
def FE_game_names():
    return get_names()

# Pass User Inputs into Backend (KNN)
@app.route("/game_KNN",methods=["POST"],strict_slashes=False)
def get_reccomendations_knn():
    json_names = request.get_json()
    names = json_names.get('body',[])
    vals = KNN.return_recs_KNN(names)

    return jsonify({ 
        "message": "Success",
        "name_recieved": names,
        "returned_data": vals
    })

@app.route("/game_MatSim", methods=["POST"], strict_slashes=False)
def get_reccomendations_msim():
    json_names = request.get_json()
    names = json_names.get('body',[])
    vals = SM.return_recs_SM(names)

    return jsonify({ 
        "message": "Success",
        "name_recieved": names,
        "returned_data": vals
    })


if __name__ == "__main__":
    app.run(debug=True)


# Pass User Inputs into Backend (Matrix Similarity)
# if __name__ == "__main__":
#     app.run(debug=True)


# Pass User Inputs into Backend (Implicit Matrix)
# @app.route("/game_ImpM",methods=["POST"],strict_slashes=False)
# def get_reccomendations():
#     json_names = request.get_json()
#     names = json_names.get('body',[])
#     vals = IM.retrieve_rec_data

#     return jsonify({ 
#         "message": "Success",
#         "name_recieved": names,
#         "returned_data": vals
#     })

# if __name__ == "__main__":
#     app.run(debug=True)