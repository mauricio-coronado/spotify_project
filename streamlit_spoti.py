import streamlit as st
# import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import ast
import numpy as np

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


def map_popularity_tier(popularity):
    
    # use np.where to assign categories based on popularity value
    category = np.where(popularity <= 25, 'Hidden Gem',
               np.where(popularity <= 50, 'Low-Profile Banger',
               np.where(popularity <= 75, 'Known and Loved Classic', 'Chart Topper Hit')))
    
    return category

def color_scale(val):

    '''
    val must be between 0 and 1
    '''

    # if the value is between 0.5 and 1 create a color scale 
    # that goes from yellow to green and assign its color
    if val > 0.5:
        val = (val - 0.5)*2 # rescalling the interval
        color = f'rgb({int(255*(1-val))}, {int(255)}, 0)'
    # if the value is between 0 and 0.5 create a color scale 
    # that goes from red to yellow and assign its color
    else:
        val = val*2 # rescalling the interval
        color = f'rgb({int(255)}, {int(255*val)}, 0)'
    
    return f'color: {color}'

st.title("Track Explorer")

st.markdown("""
 * Use the menu at left to select the artist and song you'd like recommendations for.
 * Results and additional plots will appear below.
""")

songs_df = pd.read_csv('1001_artists_and_related_discography.csv', index_col=0)

related_artists_df = pd.read_csv('related_artists_data.csv', index_col=0)
# songs_df = songs_df.merge(related_artists_df, on='artist')
related_artists_df['related_artists'] = related_artists_df['related_artists'].apply(ast.literal_eval)
# songs_df = songs_df.set_index('name')
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


    songs_df.index = songs_df.index + ' - ' + songs_df.artist
    # remove duplicates and keep first
    songs_df = songs_df[~songs_df.index.duplicated(keep='first')]

    rel_artists = related_artists_df.loc[artist_name, 'related_artists']
    # print(rel_artists)
    
    songs_df_prep = songs_df.copy()
    songs_df_prep = songs_df_prep[songs_df_prep.artist.isin(rel_artists + [artist_name])]


    predictors = [
                'acousticness',
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



    # st.write(f'<p class="big-font">The nearest neighbors for {query_index} are:</p>', unsafe_allow_html=True)
    
    st.write(f'### {song_name} by {artist_name}:')



    categories = songs_df[predictors].columns

    songs_df_prep = songs_df.copy()
    # songs_df_prep.index = songs_df_prep.index + ' - ' + songs_df_prep.artist
    scaler = MinMaxScaler()
    # Normalize the column
    songs_df_prep['loudness'] = scaler.fit_transform(songs_df_prep[['loudness']])

    fig = make_subplots(rows=1, 
                        cols=2, 
                        specs=[[{"type": "polar"}, {"type": "bar"}]], 
                        column_widths=[2/3, 1/3], 
                        # subplot_titles=("Sine Wave", "Cosine Wave")
                        )

    fig.add_trace(go.Scatterpolar(
        r=songs_df_prep[predictors].loc[query_index,:],
        theta=categories,
        fill='toself',
        name=query_index
    ),
    row=1,
    col=1)

    fig.add_trace(go.Bar(
        x=['Popularity'], 
        y=[songs_df_prep['popularity'].loc[query_index]],
        width=[0.5],
        name='Popularity'
        # range_y=[0,100]
        ),
    row=1,
    col=2)

    # fig.add_trace(go.Scatterpolar(
    #     r=songs_df_prep[predictors].loc[song_two,:],
    #     theta=categories,
    #     fill='toself',
    #     name=song_two
    # ))

    fig.update_yaxes(range=[0, 100], row=1, col=2)

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        ),
    ),
    showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, )
    
    
    # col1, col2, col3 = st.columns([3,2,1])
    # col1.markdown('#### Song')
    # col2.markdown('#### Artist')
    # col3.markdown('#### Similarity')
    
    # for neighbor_id, distance in neighbors:
    #     song, artist = neighbor_id.rsplit(' - ', 1)
    #     col1.write(song)
    #     col2.write(artist)
    #     col3.write(round(1/(1+distance),2))
    #     # st.write(f"Song: {neighbor_id}, Distance: {round(distance,2)}")


    # Create empty lists to hold data for the DataFrame
    songs = []
    artists = []
    similarities = []
    popularities = []

    # Extract data from neighbors
    for neighbor_id, distance in neighbors:
        song, artist = neighbor_id.rsplit(' - ', 1)
        popularity = songs_df.loc[neighbor_id, 'popularity']
        songs.append(song)
        artists.append(artist)
        similarities.append(round(1 / (1 + distance), 2))
        popularities.append(map_popularity_tier(popularity))

    # Create a DataFrame from the collected data
    recommendations = {
        'Song': songs,
        'Artist': artists,
        'Similarity': similarities,
        'Popularity Tier': popularities
        }
    


    recommendations_df = pd.DataFrame(recommendations)
    styled_df = recommendations_df.style.format({'Similarity': '{:.2f}'}).applymap(color_scale, subset=['Similarity'])

    # Display the DataFrame as a table
    st.dataframe(styled_df,
# .style.background_gradient(
#                                                         subset=['Similarity'], 
#                                                         cmap='RdYlGn',  # Red to Yellow to Green color map
#                                                         vmin=0,          # Minimum value for color scale
#                                                         vmax=1          # Maximum value for color scale
#                                                     ), 
                 hide_index=True, 
                 column_config={
                     'Song': st.column_config.Column(width='large'),
                     'Artist': st.column_config.Column(width='medium'),
                     'Similarity': st.column_config.Column(width='small'),
                     'Popularity Tier': st.column_config.Column(width='medium')
                                })



