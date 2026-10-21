from flask import Flask, jsonify, request
import pandas as pd
app = Flask(__name__)

def read_csv():
    
        arr = []
        data = pd.DataFrame(df)

        for i in range(len(data)):
            
            temp = {}
            temp['Name']  = data['Name'][i]
            temp['Primary_Genre'] = data['Primary_Genre'][i]

            if type(data['All_Tags'][i]) != float:
                temp['All_Tags'] = data['All_Tags'][i].split(";")

            else:
                temp['All_Tags'] = ''
            
            arr.append(temp)

    return arr

# Recomendation bar for popular steam games 2024-2026
@app.route("/games_steam_csv",methods=["POST"],strict_slashes=False)
def get_games_steam_csv():
    tags = request.get_json()
    names = tags.get('title', "")
    selected_tags = tags.get('tags', [])
    vals = sc.return_recs(names)

    return jsonify({ 
        "message": "Success recieved Tags",
        "name_recieved": names,
        "tags_recieved": selected_tags,
        "returned_data": vals
    })


# Potential game tags for user selection
@app.route("/game_tags")
def get_game_tags():
    print(read_csv)
    return read_csv()

if __name__ == "__main__":
    app.run(debug=True)
