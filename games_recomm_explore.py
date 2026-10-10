import pandas as pd

# Load Datasets
game_data = pd.read_csv('combined_games_data.csv')
recomm_data = pd.read_csv('recommendations.csv')

# Get sets of unique IDs
game_id = set(game_data['app_id'])
recomm_id = set(recomm_data['app_id'])

# Compute intersection and unique counts directly from sets
intersection = len(game_id.intersection(recomm_id))
games_unique = len(game_id)
recomm_unique = len(recomm_id)

# Print summary
print(f"Number of unique game IDs in game_data: {games_unique:,}") #50,872
print(f"Number of unique game IDs in recommendations: {recomm_unique:,}") #37,610
print(f"Number of intersecting game IDs: {intersection:,}") #37,610