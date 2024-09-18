import streamlit as st
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Function to find nearest neighbors by ID (without re-fitting the model)
def get_nearest_neighbors(df, nn_model, scaler, query_index, n_neighbors=3):

    # Get the scaled data point for the query_ndex (no need to re-scale everything)
    query_point = scaler.transform(df.loc[[query_index]])
    
    # Get the neighbors (including itself)
    distances, indices = nn_model.kneighbors(query_point, n_neighbors=n_neighbors)
    
    # Extract the neighbor IDs and distances
    neighbor_ids = df.index[indices.flatten()].tolist()
    neighbor_distances = distances.flatten().tolist()
    
    return list(zip(neighbor_ids, neighbor_distances))


st.write("# Spotify Recommendation System")

artist_name = st.text_input('Enter Artist name')
song_name = st.text_input('Enter Song name')


if st.button('Predict'):

    songs_df = pd.read_csv('related_artists_discography_first_level_p1.csv', index_col=0)
    # Example usage
    # song_name = 'anything'
    # artist_name = 'Adrianne Lenker'
    query_index = song_name + ' - ' + artist_name


    songs_df_prep = songs_df.copy()
    songs_df_prep.index = songs_df_prep.index + ' - ' + songs_df_prep.artist
    # rel_artists = songs_df_prep.related_artists.loc[query_index]
    # print(rel_artists)
    # songs_df_prep = songs_df_prep[songs_df_prep.artist.isin(rel_artists + [artist_name])]


    predictors = ['acousticness',
                'danceability',
                'energy',
                'loudness',
                'speechiness'
                ]

    songs_df_prep = songs_df_prep[predictors].copy()
    # songs_df_prep

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(songs_df_prep)

    nn = NearestNeighbors(n_neighbors=3, 
                        metric='euclidean',)
    nn.fit(X_scaled)

    # query_index = 'Walk In The Park - Beach House'
    neighbors = get_nearest_neighbors(songs_df_prep, nn, scaler, query_index, n_neighbors=10)

    # print(f"The nearest neighbors for {query_index} are:")
    # for neighbor_id, distance in neighbors:
    #     print(f"Song: {neighbor_id}, Distance: {round(distance,2)}")



    st.write(f'<p class="big-font">The nearest neighbors for {query_index} are:</p>', unsafe_allow_html=True)
    for neighbor_id, distance in neighbors:
        st.write(f"Song: {neighbor_id}, Distance: {round(distance,2)}")






