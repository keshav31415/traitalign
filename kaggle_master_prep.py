import pandas as pd
import numpy as np
import re
import os
import gc
import json
import zipfile
from scipy.stats import entropy
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else 1995

def calculate_shannon_entropy(series):
    counts = series.value_counts(normalize=True)
    return entropy(counts) if len(counts) > 0 else 0.0

def zip_files(zip_name, file_paths):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
    print(f"--> Created {zip_name} successfully!")

# -------------------------------------------------------------------
# MOVIELENS 1M PREP
# -------------------------------------------------------------------
def run_movielens_prep(users_path, movies_path, ratings_path):
    print("\n==============================================")
    print("🎬 STARTING MOVIELENS 1M PREPROCESSING 🎬")
    print("==============================================")
    
    # Load data (handling missing headers and '::' separator)
    df_u = pd.read_csv(users_path, sep='::', engine='python', header=None, names=['UserID', 'Gender', 'Age', 'OccupationID', 'Zip'])
    df_m = pd.read_csv(movies_path, sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    df_r = pd.read_csv(ratings_path, sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    df_m['Year'] = df_m['Title'].apply(extract_year)
    movie_pop = df_r.groupby('MovieID').size().reset_index(name='Popularity')
    df_m = df_m.merge(movie_pop, on='MovieID', how='left').fillna({'Popularity': 0})
    
    all_genres = sorted(list(set('|'.join(df_m['Genres'].dropna()).split('|'))))
    history = df_r.merge(df_m, on='MovieID')
    
    print("Calculating Behavioral Proxies...")
    proxy_records = []
    for user_id, group in history.groupby('UserID'):
        user_genres = '|'.join(group['Genres']).split('|')
        genre_entropy = calculate_shannon_entropy(pd.Series(user_genres))
        adren_count = sum(1 for g in user_genres if g in {'Action', 'Thriller', 'Horror', 'Sci-Fi'})
        
        genre_counts = pd.Series(user_genres).value_counts()
        genre_dist = [genre_counts.get(g, 0) / len(user_genres) for g in all_genres]
        
        proxy_records.append({
            'UserID': user_id,
            'raw_nostalgia': group['Year'].mean(),
            'raw_mainstream': group['Popularity'].mean(),
            'raw_diversity': genre_entropy,
            'adrenaline_pref': adren_count / len(user_genres) if len(user_genres) > 0 else 0,
            'raw_rating': group['Rating'].mean(),
            'genre_dist': genre_dist
        })
        
    proxies_df = pd.DataFrame(proxy_records)
    for col in ['raw_nostalgia', 'raw_mainstream', 'raw_diversity', 'raw_rating']:
        col_min, col_max = proxies_df[col].min(), proxies_df[col].max()
        proxies_df[col.replace('raw_', '')] = (proxies_df[col] - col_min) / (col_max - col_min + 1e-8)
        
    final_proxies = proxies_df[['UserID', 'nostalgia', 'mainstream', 'diversity', 'adrenaline_pref', 'rating', 'genre_dist']]
    
    df_u['Age_Norm'] = df_u['Age'] / 100.0
    df_u['Gender_Idx'] = df_u['Gender'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1}).fillna(0).astype(int)
    user_metadata = df_u[['UserID', 'Gender_Idx', 'Age_Norm', 'OccupationID']]
    
    print("Running MiniLM for Occupations...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    occ_mapping = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student", 5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}
    occ_embeddings = model.encode([occ_mapping[i] for i in range(21)], convert_to_tensor=True).cpu().numpy()
    
    # Save files
    final_proxies.to_csv("ml1m_proxies.csv", index=False)
    user_metadata.to_csv("ml1m_user_metadata.csv", index=False)
    df_r.to_csv("ml1m_interactions.csv", index=False) # Keep interactions for GNN
    np.save("ml1m_occupation_embeddings.npy", occ_embeddings)
    
    zip_files("movielens_traitalign_data.zip", [
        "ml1m_proxies.csv", "ml1m_user_metadata.csv", 
        "ml1m_interactions.csv", "ml1m_occupation_embeddings.npy"
    ])
    print("✅ MovieLens Prep Complete!")

# -------------------------------------------------------------------
# LASTFM 360K PREP
# -------------------------------------------------------------------
def run_lastfm_prep(users_path, plays_path, num_users=9000):
    print("\n==============================================")
    print("🎵 STARTING LASTFM 360K PREPROCESSING 🎵")
    print("==============================================")
    
    # Try reading TSV or CSV safely
    try:
        df_u = pd.read_csv(users_path, sep='\t', header=None, names=['UserID', 'Gender', 'Age', 'Country', 'Signup'])
        if df_u['Country'].isnull().all(): # Fallback to comma if tab failed
            df_u = pd.read_csv(users_path, sep=',', header=None, names=['UserID', 'Gender', 'Age', 'Country', 'Signup'])
    except:
        df_u = pd.read_csv(users_path)
        df_u.columns = ['UserID', 'Gender', 'Age', 'Country', 'Signup']

    df_u = df_u.dropna(subset=['Gender', 'Age', 'Country'])
    df_u['Age'] = pd.to_numeric(df_u['Age'], errors='coerce')
    df_u = df_u[(df_u['Age'] >= 10) & (df_u['Age'] <= 100)]
    
    sampled_users = df_u.sample(n=min(num_users, len(df_u)), random_state=42)
    valid_user_ids = set(sampled_users['UserID'])
    
    print(f"Extracting interactions for {len(valid_user_ids)} users from 3.93GB file...")
    filtered_chunks = []
    try:
        # Assuming TSV first
        for chunk in pd.read_csv(plays_path, sep='\t', header=None, names=['UserID', 'ArtistMBID', 'ArtistName', 'Plays'], chunksize=1000000):
            filtered_chunks.append(chunk[chunk['UserID'].isin(valid_user_ids)])
    except:
        # Fallback
        for chunk in pd.read_csv(plays_path, sep=',', header=None, names=['UserID', 'ArtistMBID', 'ArtistName', 'Plays'], chunksize=1000000):
            filtered_chunks.append(chunk[chunk['UserID'].isin(valid_user_ids)])
            
    df_plays = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks; gc.collect()
    
    print("Calculating Behavioral Proxies...")
    artist_pop = df_plays.groupby('ArtistMBID')['Plays'].sum().reset_index(name='GlobalPlays')
    df_plays = df_plays.merge(artist_pop, on='ArtistMBID')
    top_10 = artist_pop['GlobalPlays'].quantile(0.90)
    
    proxy_records = []
    for user_id, group in df_plays.groupby('UserID'):
        total_plays = group['Plays'].sum()
        proxy_records.append({
            'UserID': user_id,
            'raw_engagement': total_plays,
            'discovery_rate': len(group) / total_plays if total_plays > 0 else 0,
            'raw_mainstream': np.average(group['GlobalPlays'], weights=group['Plays']) if total_plays > 0 else 0,
            'raw_diversity': entropy(group['Plays'] / total_plays) if total_plays > 0 else 0,
            'long_tail_pref': group[group['GlobalPlays'] < top_10]['Plays'].sum() / total_plays if total_plays > 0 else 0
        })
        
    proxies_df = pd.DataFrame(proxy_records)
    for col in ['raw_engagement', 'raw_mainstream', 'raw_diversity']:
        col_min, col_max = proxies_df[col].min(), proxies_df[col].max()
        proxies_df[col.replace('raw_', '')] = (proxies_df[col] - col_min) / (col_max - col_min + 1e-8)
        
    final_proxies = proxies_df[['UserID', 'engagement', 'discovery_rate', 'mainstream', 'diversity', 'long_tail_pref']]
    
    sampled_users['Age_Norm'] = sampled_users['Age'] / 100.0
    sampled_users['Gender_Idx'] = sampled_users['Gender'].map({'m': 0, 'f': 1, 'M': 0, 'F': 1}).fillna(0).astype(int)
    user_metadata = sampled_users[['UserID', 'Gender_Idx', 'Age_Norm', 'Country']]
    
    print("Running MiniLM for Countries...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    unique_countries = sorted(sampled_users['Country'].unique().tolist())
    country_embeddings = model.encode(unique_countries, convert_to_tensor=True).cpu().numpy()
    country_to_idx = {country: idx for idx, country in enumerate(unique_countries)}
    
    # Map string country to index for GNN
    user_metadata['CountryIdx'] = user_metadata['Country'].map(country_to_idx)
    user_metadata = user_metadata.drop(columns=['Country'])
    
    # Save files
    final_proxies.to_csv("lastfm_proxies.csv", index=False)
    user_metadata.to_csv("lastfm_user_metadata.csv", index=False)
    df_plays[['UserID', 'ArtistMBID', 'Plays']].to_csv("lastfm_interactions.csv", index=False)
    np.save("lastfm_country_embeddings.npy", country_embeddings)
    with open("lastfm_country_to_idx.json", "w") as f:
        json.dump(country_to_idx, f)
        
    zip_files("lastfm_traitalign_data.zip", [
        "lastfm_proxies.csv", "lastfm_user_metadata.csv", "lastfm_interactions.csv", 
        "lastfm_country_embeddings.npy", "lastfm_country_to_idx.json"
    ])
    print("✅ LastFM Prep Complete!")


if __name__ == "__main__":
    print("🚀 Starting Master TraitAlign Preprocessing Pipeline...")
    
    # ==========================================
    # MOVIELENS RUN
    # ==========================================
    # Change these paths to your Kaggle MovieLens 1M paths!
    ml_users = '/kaggle/input/datasets/odedgolden/movielens-1m-dataset/users.dat'
    ml_movies = '/kaggle/input/datasets/odedgolden/movielens-1m-dataset/movies.dat'
    ml_ratings = '/kaggle/input/datasets/odedgolden/movielens-1m-dataset/ratings.dat'
    
    if os.path.exists(ml_users):
        run_movielens_prep(ml_users, ml_movies, ml_ratings)
    else:
        print(f"Skipping MovieLens... could not find {ml_users}")
        
    # ==========================================
    # LASTFM RUN
    # ==========================================
    # Change these paths to your Kaggle LastFM paths!
    lfm_users = '/kaggle/input/datasets/neferfufi/lastfm/usersha1-profile.csv'
    lfm_plays = '/kaggle/input/datasets/neferfufi/lastfm/usersha1-artmbid-artname-plays.csv' # Adjust if it's .tsv
    
    if os.path.exists(lfm_users):
        run_lastfm_prep(lfm_users, lfm_plays, num_users=9000)
    else:
        print(f"Skipping LastFM... could not find {lfm_users}")
        
    print("\n🎉 All Processing Done! Look for the 2 ZIP files in your Kaggle output directory!")
