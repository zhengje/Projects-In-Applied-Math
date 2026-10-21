from flask import Flask, jsonify, request
import pandas as pd
import KNNTest as k
app = Flask(__name__)


# Retrieve Game Information from Database (Search Bar)
def get_names():
    titles = []
    data = k.load_data()
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


# Pass User Inputs into Backend
@app.route("/game_recommendations",methods=["POST"],strict_slashes=False)
def get_reccomendations():
    json_names = request.get_json()
    names = json_names.get('body',[])
    vals = k.retrieve_rec_data

    return jsonify({ 
        "message": "Success",
        "name_recieved": names,
        "returned_data": vals
    })

if __name__ == "__main__":
    app.run(debug=True)