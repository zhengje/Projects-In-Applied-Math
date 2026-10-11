import pandas as pd
import ast

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

# Filter game_data to only include rows with app_id present in recomm_data
filtered_game_data = game_data[game_data['app_id'].isin(recomm_id)]

print(f"Original combined_games_data rows: {len(game_data):,}") #50,872
print(f"Filtered combined_games_data rows: {len(filtered_game_data):,}")#37,610

# Save the shortened CSV back to combined_games_data.csv
filtered_game_data.to_csv('combined_games_data.csv', index=False)

#Check how many rows have no values in genre, categories and filtered tags
columns_to_check = ['genres', 'categories', 'filtered_tags']
for column in columns_to_check:
    missing_count = (filtered_game_data[column].isna() |
    filtered_game_data[column].astype(str).str.strip().isin(['','[]'])
    )
    missing_sum = missing_count.sum()
    print(f"Number of rows with no values in '{column}': {missing_sum:,}")
#missing values in genres: 909
#missing values in categories: 13,697
#missing values in filtered tags: 3,421

#Extract unique tags

'''unique_tags = set()

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

'''