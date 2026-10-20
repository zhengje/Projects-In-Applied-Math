import pandas as pd
import ast
import numpy as np

# Load Datasets
game_data = pd.read_csv('data_fixed.csv')
recomm_data = pd.read_csv('recommendations.csv')

# Get sets of unique IDs
game_id = set(game_data['app_id'])
recomm_id = set(recomm_data['app_id'])


#Check how many rows have no values in genre, categories and filtered tags
columns_to_check = ['genres', 'categories', 'filtered_tags']
for column in columns_to_check:
    missing_count = (game_data[column].isna() |
    game_data[column].astype(str).str.strip().isin(['','[]'])
    )
    missing_sum = missing_count.sum()
    print(f"Number of rows with no values in '{column}': {missing_sum:,}")

#Missing values in genres: 0
#Missing values in categories: 10,484
#Missing values in filtered_tags: 0



#Remove rows with no values in genres and filtered tags
game_data['filtered_tags'] = game_data['filtered_tags'].replace([''],np.nan)
filtered_game_data = game_data.dropna(subset=['genres', 'filtered_tags'])
print("Remaining missing values in 'genres':", game_data['genres'].isna().sum())#0
print("Remaining missing values in 'filtered_tags':", game_data['filtered_tags'].isna().sum())#0
print("Total remaining rows in dataset:", len(game_data)) #34477


# Compute intersection and unique counts directly from sets
intersection = len(game_id.intersection(recomm_id))
games_unique = len(game_id)
recomm_unique = len(recomm_id)

# Print summary
print(f"Number of unique game IDs in game_data: {games_unique:,}")  #34,477
print(f"Number of unique game IDs in recommendations: {recomm_unique:,}") #37,610
print(f"Number of intersecting game IDs: {intersection:,}") #34,477

# Filter recomm_data to only include rows with app_id present in recomm_data
filtered_game_data = game_data[game_data['app_id'].isin(recomm_id)]
filtered_recomm_data = recomm_data[recomm_data['app_id'].isin(game_id)]

filter_recomm_unique = len(set(filtered_recomm_data['app_id']))
print(f"Number of unique game IDs in filtered recommendations: {filter_recomm_unique:,}") #34,477
print(f"Original recommendations rows: {len(recomm_data):,}")  #41,154,794
print(f"Filtered recommendations rows: {len(filtered_recomm_data):,}") #40,995,212

# Save the shortened CSV back to combined_games_data.csv
filtered_game_data.to_csv('data_fixed.csv', index=False)
filtered_recomm_data.to_csv('recommendations.csv', index=False)

#Extract unique tags
unique_tags = set()

for cell in game_data['tags'].dropna():
    try:
        tag_list = ast.literal_eval(cell)
        if isinstance(tag_list, list):
             # Clean and standardize string structure
            cleaned_tag = [str(t).strip() for t in tag_list if t]
            unique_tags.update(cleaned_tag)
                   
                    
    except (ValueError, SyntaxError):
        continue

all_unique_tags = sorted(list(unique_tags))
print(f"Total standardized unique tags: {len(all_unique_tags)}")
print("All tags:",all_unique_tags)


