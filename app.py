import streamlit as st
import pandas as pd
import joblib
import zipfile
import os
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv
import os
import base64, requests, urllib.parse

load_dotenv()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    return df


@st.cache_resource
def load_components(zip_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('model_tmp')
    model = joblib.load(os.path.join('model_tmp', 'nn_model.joblib'))
    text_pipeline = joblib.load(os.path.join('model_tmp', 'text_pipeline.joblib')) if os.path.exists(os.path.join('model_tmp', 'text_pipeline.joblib')) else None
    numerical_pipeline = joblib.load(os.path.join('model_tmp', 'numerical_pipeline.joblib')) if os.path.exists(os.path.join('model_tmp', 'numerical_pipeline.joblib')) else None
    text_features = joblib.load(os.path.join('model_tmp', 'text_features.joblib'))
    numerical_features = joblib.load(os.path.join('model_tmp', 'numerical_features.joblib'))
    return model, text_pipeline, numerical_pipeline, text_features, numerical_features

def recommend(song: str, artist: str, df: pd.DataFrame, model, text_pipeline, numerical_pipeline, text_features, numerical_features, feature_weights, top_n: int = 5):
    # Find the index of the query song
    query_series = df.loc[(df['track_name'].str.lower() == song.lower()) & (df['artist_name'].str.lower() == artist.lower())]
    if query_series.empty:
        return []
    idx = query_series.index[0]
    query_data = df.iloc[[idx]].copy()
    # Transform query song features separately
    if text_pipeline and text_features:
        combined_query_text = query_data[text_features].fillna('').agg(' '.join, axis=1)
        query_text_matrix = text_pipeline.transform(combined_query_text)
    else:
        query_text_matrix = None
    query_numerical_matrix = numerical_pipeline.transform(query_data[numerical_features]) if numerical_pipeline else None
    # Combine and weight query features
    import numpy as np
    from scipy.sparse import hstack
    if query_text_matrix is not None and query_numerical_matrix is not None:
        if hasattr(query_numerical_matrix, 'toarray'):
            query_numerical_matrix = query_numerical_matrix.toarray()
        weighted_query_text_matrix = query_text_matrix * sum(feature_weights[col] for col in text_features) / len(text_features) if text_features else query_text_matrix
        weighted_query_numerical_matrix = query_numerical_matrix * sum(feature_weights[col] for col in numerical_features) / len(numerical_features) if numerical_features else query_numerical_matrix
        query_vec = hstack([weighted_query_text_matrix, weighted_query_numerical_matrix])
    elif query_text_matrix is not None:
        query_vec = query_text_matrix * sum(feature_weights[col] for col in text_features) / len(text_features) if text_features else query_text_matrix
    elif query_numerical_matrix is not None:
        query_vec = query_numerical_matrix * sum(feature_weights[col] for col in numerical_features) / len(numerical_features) if numerical_features else query_numerical_matrix
    else:
        return []
    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n + 1)
    recs = []
    for dist, i in zip(distances[0], indices[0]):
        if i != idx:
            recs.append({
                'Track': df.at[i, 'track_name'],
                'Artist': df.at[i, 'artist_name'],
                'Score': f"{1 - dist:.2f}"
            })
        if len(recs) == top_n:
            break
    return recs

def spotify_link(artist, track):
    
    client_id = st.secrets["spotify"]["client_id"]
    client_secret = st.secrets["spotify"]["client_secret"]


    auth_header   = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    token_res = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={ "Authorization": f"Basic {auth_header}" },
        data={ "grant_type": "client_credentials" }
    )
    token = token_res.json().get("access_token")
    if not token:
        return None

    # 2) Perform a track search
    q   = urllib.parse.quote(f"track:{track} artist:{artist}")
    url = f"https://api.spotify.com/v1/search?q={q}&type=track&limit=1"
    res = requests.get(url, headers={ "Authorization": f"Bearer {token}" })
    items = res.json().get("tracks", {}).get("items", [])

    # 3) Extract and return the track ID
    if not items:
        return None
    return items[0]["id"]

    

def main():
    st.set_page_config(page_title='🎵 Music Recommender', layout='wide')
    st.title('🎵 Music Recommender System')

    st.sidebar.header('Upload & Settings')
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    data_path = uploaded if uploaded else 'tcc_ceds_music.csv'

    with st.spinner('Loading data…'):
        df = load_data(data_path)
    with st.spinner('Loading trained model components…'):
        model, text_pipeline, numerical_pipeline, text_features, numerical_features = load_components('music_recommender_model.zip')

    # Feature weights (should match training)
    feature_weights = {
        'artist_name': 3.0,
        'track_name': 1.0,
        'genre': 3.5,
        'lyrics': 2.2,
        'world/life': 3.0,
        'violence': 3.0,
        'dating': 3.0,
        'release_date': 2.5,
        'len': 1.0,
        'danceability': 3.0,
        'loudness': 3.0,
        'acousticness': 3.0,
        'instrumentalness': 3.0,
        'valence': 3.0,
        'energy': 3.0
    }



    st.markdown('## Get Recommendations')
    track_list = df['track_name'].unique()
    artist_list = df['artist_name'].unique()
    selected_track = st.selectbox('Select a track', track_list, index=0)
    selected_artist = st.selectbox('Select an artist', artist_list, index=0)
    num_rec = st.slider('Number of recommendations', 1, 10, 5)
    if st.button('Recommend'):
        recs = recommend(selected_track, selected_artist, df, model, text_pipeline, numerical_pipeline, text_features, numerical_features, feature_weights, top_n=num_rec)
        if recs:
            for r in recs:
        # Display title + score
                st.subheader(f"{r['Track']} — {r['Artist']}  |  Score: {r['Score']}")
                artist = r['Artist']
                song  = r['Track']
                a = f"https://open.spotify.com/embed/track/{spotify_link(artist, song)}"
                # Build and render the Spotify iframe
                iframe = f"""
                <iframe style="border-radius:12px"
                        src="{a}"
                        width="300" height="80" frameborder="0"
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                        loading="lazy">
                </iframe>
                """
                st_html(iframe, height=100)

        st.markdown('## Dataset Overview')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Songs', df.shape[0])
    col2.metric('Unique Artists', df['artist_name'].nunique())
    col3.metric('Genres', df['genre'].nunique())

    st.markdown('### Top 10 Genres')
    genre_counts = df['genre'].value_counts().nlargest(10)
    st.bar_chart(genre_counts)
if __name__ == '__main__':
    main()