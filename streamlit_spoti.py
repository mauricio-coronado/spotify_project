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


st.title("Spotify Recommendation System")

st.markdown("""
 * Use the menu at left to select the artist and song you'd like recommendations for.
 * Results and additional plots will appear below.
""")

songs_df = pd.read_csv('1001_artists_discography.csv', index_col=0)
songs_df.index = songs_df.index.astype(str)

st.sidebar.markdown("## Select Artist and Song")

# list of options for autocomplete-like functionality
artist_options = sorted(songs_df.artist.unique())   
default_option = ''  # blank default option
# prepend the default blank option to the options list
artist_options = [default_option] + artist_options

# display a selectbox where users can choose from predefined options
artist_name = st.sidebar.selectbox('Artist:', artist_options)

if artist_name == default_option:
    song_options = sorted(songs_df.index.unique())
else:
    song_options = sorted(songs_df[songs_df.artist == artist_name].index.unique())

song_options = [default_option] + song_options
song_name = st.sidebar.selectbox('Song:', song_options)


if st.sidebar.button('Predict'):

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
    neighbors = get_nearest_neighbors(songs_df_prep, nn, scaler, query_index, n_neighbors=20)

    # print(f"The nearest neighbors for {query_index} are:")
    # for neighbor_id, distance in neighbors:
    #     print(f"Song: {neighbor_id}, Distance: {round(distance,2)}")



    # st.write(f'<p class="big-font">The nearest neighbors for {query_index} are:</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3,1])
    col1.markdown('#### Song')
    col2.markdown('#### Distance')
    # col1.write('Song')
    # col2.write('Distance')

    # # Three columns with different widths
    # col1, col2, col3 = st.columns([3,1,1])
    # # col1 is wider

    # # Using 'with' notation:
    # >>> with col1:
    # >>>     st.write('This is column 1')
    
    for neighbor_id, distance in neighbors:
        col1.write(neighbor_id)
        col2.write(round(distance,2))
        # st.write(f"Song: {neighbor_id}, Distance: {round(distance,2)}")






