from flask import Flask, jsonify, request
import pandas as pd

def read_csv():
    df = pd.read_csv("steam_games_2026.csv", usecols=['Name', 'Primary_Genre', 'All_Tags'])

    arr = []
    data = pd.DataFrame(df)

    for i in range(len(data))
        
        temp = {}
        temp['Name']  = data['Name'][i]
        temp['Primary_Genre'] = data['Primary_Genre'][i]

        if type(data['All_Tags'][i]) != float:
            temp['All_Tags'] = data['All_Tags'][i].split(";")

        else:
            temp['All_Tags'] = ''
        
        arr.append(temp)

    return arr

@app.route("/games")

def csv_front():
    return read_csv()