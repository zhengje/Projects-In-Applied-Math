import ast
import re
import pandas as pd

def normalize_title(text):
    if not isinstance(text, str):
        return ""
    # Remove trademark/registered symbols, lowercase, and collapse whitespace
    text = re.sub(r'[™®©]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# 1. Load Data_Objective and build a lookup map: normalized_title -> (app_id, official_title)
games_data = pd.read_csv("Data_Objective.csv")

title_to_game = {}
for _, row in games_data.dropna(subset=['title']).iterrows():
    norm = normalize_title(row['title'])
    if norm and norm not in title_to_game:
        title_to_game[norm] = (int(row['app_id']), row['title'])

rows = []

# 2. Parse the JSON and match on normalized item_name
with open(r"C:\Users\shann\Downloads\australian_users_items.json\australian_users_items.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            user_data = ast.literal_eval(line)
            user_id = user_data.get('user_id')
            steam_id = user_data.get('steam_id')
            user_url = user_data.get('user_url')
            items = user_data.get('items', [])

            matched_items = []
            for item in items:
                norm_name = normalize_title(item.get('item_name'))
                if norm_name in title_to_game:
                    matched_items.append((item, title_to_game[norm_name]))

            for item, (canonical_app_id, canonical_title) in matched_items:
                rows.append({
                    'user_id': user_id,
                    'items_count': len(matched_items),
                    'steam_id': steam_id,
                    'user_url': user_url,
                    'app_id': canonical_app_id,          # Standardized ID from Data_Objective
                    'original_item_id': item.get('item_id'),
                    'item_name': canonical_title,        # Standardized title from Data_Objective
                    'playtime_forever': item.get('playtime_forever'),
                    'playtime_2weeks': item.get('playtime_2weeks')
                })

# 3. Export to CSV
df = pd.DataFrame(rows)
df.to_csv("australian_users_items_filtered.csv", index=False)

print(f"Saved {len(df)} rows across {df['app_id'].nunique()} unique games.")