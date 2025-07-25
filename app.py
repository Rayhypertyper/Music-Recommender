import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    df['combined_features'] = df['genre'] + ' ' + df['artist_name'] + ' ' + df['track_name']
    return df

@st.cache_data
def compute_similarity_matrix(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(track: str, df: pd.DataFrame, sim_matrix, top_n: int = 5):
    if track not in df['track_name'].values:
        return []
    idx = df.index[df['track_name'] == track][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted([s for s in scores if s[1] < 1.0], key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in scores[:top_n]:
        recommendations.append({
            'Track': df.at[i, 'track_name'],
            'Artist': df.at[i, 'artist_name'],
            'Score': f"{score:.2f}"
        })
    return recommendations

def main():
    st.set_page_config(page_title='🎵 Music Recommender', layout='wide')
    st.title('🎵 Music Recommender System')

    st.sidebar.header('Upload & Settings')
    uploaded = st.sidebar.file_uploader('Upload CSV', type=['csv'])
    data_path = uploaded if uploaded else 'tcc_ceds_music.csv'

    with st.spinner('Loading data…'):
        df = load_data(data_path)
    with st.spinner('Computing similarity matrix…'):
        sim_matrix = compute_similarity_matrix(df)

    st.markdown('## Dataset Overview')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Songs', df.shape[0])
    col2.metric('Unique Artists', df['artist_name'].nunique())
    col3.metric('Genres', df['genre'].nunique())

    st.markdown('### Top 10 Genres')
    genre_counts = df['genre'].value_counts().nlargest(10)
    st.bar_chart(genre_counts)

    st.markdown('## Get Recommendations')
    track_list = sorted(df['track_name'].unique())
    selected_track = st.selectbox('Select a track', track_list)
    num_rec = st.slider('Number of recommendations', 1, 10, 5)
    if st.button('Recommend'):
        recs = recommend(selected_track, df, sim_matrix, top_n=num_rec)
        if recs:
            st.markdown(f"### Recommendations for **{selected_track}**:")
            st.table(recs)
        else:
            st.warning('Track not found or no recommendations available.')

if __name__ == '__main__':
    main()
