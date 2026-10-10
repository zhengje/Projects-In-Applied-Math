import pandas as pd
import ast



#Define standard Steam Genres:
steam_genres = {
    "Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", 
    "Strategy", "Sports", "Racing", "Massively Multiplayer", "Early Access", 
    "Free to Play", "Puzzle", "Anime", "JRPG", "VR"
}

#Define standard Steam Categories:
steam_categories = {
    "Singleplayer", "Multiplayer", "Co-op", "Online Co-op", "Local Co-op", 
    "Local Multiplayer", "Cross-Platform Multiplayer", "Shared/Split Screen", 
    "Full controller support", "Partial Controller Support", "Controller", 
    "Steam Achievements", "Steam Cloud", "PvP", "Online PvP", "LAN PvP", 
    "PvE", "MMO", "In-App Purchases", "Split Screen"
}

def parse_tag_list(val):
    if pd.isna(val) or not val:
        return []
    if isinstance(val, list):
        return val
    try:
        # Converts "['Action', 'Adventure']" string to actual list ['Action', 'Adventure']
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Fallback if comma-separated plain text
        return [item.strip() for item in str(val).split(",") if item.strip()]


#Load game dataset
df_games = pd.read_csv("combined_games_data.csv")

#To store extracted columns:
genres_list = []
categories_list = []
filtered_tags_list = []


for tags_raw in df_games["tags"]:
    parsed = parse_tag_list(tags_raw)
    
    extracted_genres = []
    extracted_categories = []
    extracted_tags = []
    
    for item in parsed:
        if item in steam_genres:
            extracted_genres.append(item)
        elif item in steam_categories:
            extracted_categories.append(item)
        else:
            extracted_tags.append(item)


    #Join the extracted lists into strings
    genres_list.append(", ".join(extracted_genres))
    categories_list.append(", ".join(extracted_categories))
    filtered_tags_list.append(", ".join(extracted_tags))

#Assign the new columns 
df_games["genres"] = genres_list
df_games["categories"] = categories_list
df_games["filtered_tags"] = filtered_tags_list

#Save new columns to games.csv
df_games.to_csv("combined_games_data.csv", index=False)
