import pandas as pd
import ast



#Define standard Steam Genres:
steam_genres = {
    "Action", "Adventure", "Anime", "CRPG", "Casual", "Early Access", 
    "Free to Play", "Indie", "JRPG", "Massively Multiplayer", "Puzzle", 
    "RPG", "Racing", "Simulation", "Sports", "Strategy", "VR"
}

steam_categories = {
    "Asymmetric VR", "Asynchronous Multiplayer", "Benchmark", "Co-op", 
    "Co-op Campaign", "Competitive", "Controller", "Cross-Platform Multiplayer", 
    "Full controller support", "In-App Purchases", "LAN PvP", "Level Editor", 
    "Local Co-Op", "Local Co-op", "Local Multiplayer", "MMO", "MMORPG", 
    "MOBA", "Mod", "Moddable", "Mouse only", "Multiplayer", "Online Co-Op", 
    "Online Co-op", "Online PvP", "Partial Controller Support", "PvE", "PvP", 
    "Shared/Split Screen", "Singleplayer", "Social Deduction", "Software", 
    "Software Training", "Split Screen", "Steam Achievements", "Steam Cloud", 
    "Steam Machine", "Team-Based", "Touch-Friendly", "TrackIR", "Utilities", 
    "VR", "Voice Control"
}

steam_descriptors = {
    "1980s", "1990's", "2.5D", "2D", "2D Fighter", "2D Platformer", 
    "360 Video", "3D", "3D Fighter", "3D Platformer", "3D Vision", 
    "4 Player Local", "4X", "6DOF", "8-bit Music", "ATV", "Abstract", 
    "Action RPG", "Action RTS", "Action Roguelike", "Action-Adventure", 
    "Addictive", "Agriculture", "Aliens", "Alternate History", "Ambient", 
    "America", "Animation & Modeling", "Arcade", "Archery", "Arena Shooter", 
    "Artificial Intelligence", "Assassin", "Atmospheric", "Audio Production", 
    "Auto Battler", "Automation", "Automobile Sim", "BMX", "Base Building", 
    "Baseball", "Based On A Novel", "Basketball", "Battle Royale", "Beat 'em up", 
    "Beautiful", "Bikes", "Blood", "Board Game", "Boss Rush", "Bowling", 
    "Boxing", "Building", "Bullet Hell", "Bullet Time", "Capitalism", 
    "Card Battler", "Card Game", "Cartoon", "Cartoony", "Cats", 
    "Character Action Game", "Character Customization", "Chess", 
    "Choices Matter", "Choose Your Own Adventure", "Cinematic", "City Builder", 
    "Class-Based", "Classic", "Clicker", "Coding", "Cold War", "Collectathon", 
    "Colony Sim", "Colorful", "Combat", "Combat Racing", "Comedy", "Comic Book", 
    "Conspiracy", "Conversation", "Cooking", "Cozy", "Crafting", 
    "Creature Collector", "Cricket", "Crime", "Crowdfunded", "Cult Classic", 
    "Cute", "Cyberpunk", "Cycling", "Dark", "Dark Comedy", "Dark Fantasy", 
    "Dark Humor", "Dating Sim", "Deckbuilding", "Demons", "Design & Illustration", 
    "Destruction", "Detective", "Difficult", "Dinosaurs", "Diplomacy", 
    "Documentary", "Dog", "Dragons", "Drama", "Driving", "Dungeon Crawler", 
    "Dungeons & Dragons", "Dynamic Narration", "Dystopian", "Economy", 
    "Education", "Electronic", "Electronic Music", "Emotional", "Epic", 
    "Episodic", "Escape Room", "Experience", "Experimental", "Exploration", 
    "FMV", "FPS", "Faith", "Family Friendly", "Fantasy", "Farming", 
    "Farming Sim", "Fast-Paced", "Feature Film", "Female Protagonist", 
    "Fighting", "First-Person", "Fishing", "Flight", "Football (American)", 
    "Football (Soccer)", "Foreign", "Fox", "Funny", "Futuristic", "Gambling", 
    "Game Development", "GameMaker", "Games Workshop", "Gaming", "God Game", 
    "Golf", "Gore", "Gothic", "Grand Strategy", "Great Soundtrack", 
    "Grid-Based Movement", "Gun Customization", "Hack and Slash", "Hacking", 
    "Hand-drawn", "Hardware", "Heist", "Hentai", "Hero Shooter", "Hex Grid", 
    "Hidden Object", "Historical", "Hobby Sim", "Hockey", "Horror", "Horses", 
    "Hunting", "Idler", "Illuminati", "Immersive", "Immersive Sim", 
    "Instrumental Music", "Intentionally Awkward Controls", "Interactive Fiction", 
    "Inventory Management", "Investigation", "Isometric", "Jet", "Job Simulator", 
    "Jump Scare", "Kickstarter", "LEGO", "LGBTQ+", "Lemmings", "Life Sim", 
    "Linear", "Logic", "Loot", "Looter Shooter", "Lore-Rich", "Lovecraftian", 
    "Magic", "Mahjong", "Management", "Mars", "Martial Arts", 
    "Massively Multiplayer", "Match 3", "Mature", "Mechs", "Medical Sim", 
    "Medieval", "Memes", "Metroidvania", "Military", "Mini Golf", "Minigames", 
    "Minimalist", "Mining", "Modern", "Motocross", "Motorbike", "Movie", 
    "Multiple Endings", "Music", "Music-Based Procedural Generation", "Musou", 
    "Mystery", "Mystery Dungeon", "Mythology", "NSFW", "Narration", "Narrative", 
    "Nature", "Naval", "Naval Combat", "Ninja", "Noir", "Nonlinear", "Nostalgia", 
    "Nudity", "Offroad", "Old School", "On-Rails Shooter", "Open World", 
    "Open World Survival Craft", "Otome", "Outbreak Sim", "Parkour", "Parody", 
    "Party", "Party Game", "Party-Based RPG", "Perma Death", "Philosophical", 
    "Photo Editing", "Physics", "Pinball", "Pirates", "Pixel Graphics", 
    "Platformer", "Point & Click", "Political", "Political Sim", "Politics", 
    "Pool", "Post-apocalyptic", "Precision Platformer", "Procedural Generation", 
    "Programming", "Psychedelic", "Psychological", "Psychological Horror", 
    "Puzzle Platformer", "Quick-Time Events", "RPGMaker", "RTS", 
    "Real Time Tactics", "Real-Time", "Real-Time with Pause", "Realistic", 
    "Reboot", "Relaxing", "Remake", "Replay Value", "Resource Management", 
    "Retro", "Rhythm", "Robots", "Rock Music", "Roguelike", 
    "Roguelike Deckbuilder", "Roguelite", "Roguevania", "Romance", "Rome", 
    "Rugby", "Runner", "Sailing", "Sandbox", "Satire", "Sci-fi", "Science", 
    "Score Attack", "Sequel", "Sexual Content", "Shoot 'Em Up", "Shooter", 
    "Shop Keeper", "Short", "Side Scroller", "Silent Protagonist", 
    "Skateboarding", "Skating", "Skiing", "Sniper", "Snooker", "Snow", 
    "Snowboarding", "Sokoban", "Solitaire", "Souls-like", "Soundtrack", 
    "Space", "Space Sim", "Spaceships", "Spectacle fighter", "Spelling", 
    "Sports", "Stealth", "Steampunk", "Story Rich", "Strategy RPG", 
    "Stylized", "Submarine", "Superhero", "Supernatural", "Surreal", 
    "Survival", "Survival Horror", "Swordplay", "Tabletop", "Tactical", 
    "Tactical RPG", "Tanks", "Tennis", "Text-Based", "Third Person", 
    "Third-Person Shooter", "Thriller", "Tile-Matching", "Time Attack", 
    "Time Management", "Time Manipulation", "Time Travel", "Top-Down", 
    "Top-Down Shooter", "Tower Defense", "Trading", "Trading Card Game", 
    "Traditional Roguelike", "Trains", "Transhumanism", "Transportation", 
    "Trivia", "Turn-Based", "Turn-Based Combat", "Turn-Based Strategy", 
    "Turn-Based Tactics", "Tutorial", "Twin Stick Shooter", "Typing", 
    "Underground", "Underwater", "Unforgiving", "Vampire", "Vehicular Combat", 
    "Video Production", "Vikings", "Villain Protagonist", "Violent", 
    "Visual Novel", "Volleyball", "Voxel", "Walking Simulator", "War", 
    "Wargame", "Warhammer 40K", "Web Publishing", "Well-Written", "Werewolves", 
    "Western", "Wholesome", "Word Game", "World War I", "World War II", 
    "Wrestling", "Zombies", "eSports"
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
df_games = pd.read_csv("data_fixed.csv")

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
df_games.to_csv("data_fixed.csv", index=False)

#Check how many rows have no values in genre, categories and filtered tags
columns_to_check = ['genres', 'categories', 'filtered_tags']
for column in columns_to_check:
    missing_count = (df_games[column].isna() |
    df_games[column].astype(str).str.strip().isin(['','[]'])
    )
    missing_sum = missing_count.sum()
    print(f"Number of rows with no values in '{column}': {missing_sum:,}")
#missing values in genres: 0
#missing values in categories: 10,484
#missing values in filtered tags: 0