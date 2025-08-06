# Music Recommender System 🎵

This project is a **Music Recommender System** built using Python and Streamlit. It analyzes song data and provides recommendations based on similarity metrics such as TF-IDF and cosine similarity.

## Features

- **Song Recommendations**: Get song recommendations based on a selected track and artist.
- **Spotify Integration**: Embedded Spotify player for recommended songs.
- **Data Visualization**: Visualize top genres and dataset statistics.
- **Customizable Inputs**: Upload your own dataset for recommendations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/music-recommender.git
   cd music-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory:
     ```properties
     client_id = "your_spotify_client_id"
     client_secret = "your_spotify_client_secret"
     ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

3. Upload a CSV file or use the default dataset (`tcc_ceds_music.csv`).

4. Select a track, artist, and the number of recommendations to get personalized song suggestions.

## Dataset

The default dataset (`tcc_ceds_music.csv`) contains the following columns:
- `artist_name`
- `track_name`
- `release_date`
- `genre`
- `lyrics`
- Various numerical features like `danceability`, `energy`, etc.

## Model Components

The recommender system uses:
- **TF-IDF**: To calculate term importance in song lyrics.
- **Cosine Similarity**: To measure similarity between songs.
- **Pre-trained Pipelines**: For text and numerical feature processing.

## Spotify Integration

The app uses Spotify's API to fetch track IDs for embedding songs. Ensure you have valid Spotify API credentials in `.env` or `.streamlit/secrets.toml`.

## File Structure

```
.
├── app.py                # Streamlit application
├── main.py               # Core recommendation logic
├── tcc_ceds_music.csv    # Default dataset
├── model_tmp/            # Pre-trained model components
├── .streamlit/           # Streamlit configuration
│   └── .streamlit/secrets.toml      # Spotify API credentials
├── requirements.txt      # Python dependencies
└── readme.md             # Project documentation
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Streamlit**: For building interactive web applications.
- **Spotify API**: For song embedding and metadata.
- **Scikit-learn**: For machine learning utilities.

Enjoy discovering new music! 🎶