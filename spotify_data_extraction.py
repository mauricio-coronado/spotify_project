import spotipy
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import time
import requests

load_dotenv()

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')
scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope,
                                               client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri
                                               ))


def get_playlist_tracks(username,playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


def get_track_features(id):
    
    metadata = sp.track(id)
    features = sp.audio_features(id)

    track_features = {}
    # metadata
    track_features['name'] = metadata['name']
    track_features['album'] = metadata['album']['name']
    track_features['artist'] = metadata['album']['artists'][0]['name']
    track_features['release_date'] = metadata['album']['release_date']
    track_features['length'] = metadata['duration_ms']
    track_features['popularity'] = metadata['popularity']

    # audio features
    track_features['acousticness'] = features[0]['acousticness']
    track_features['danceability'] = features[0]['danceability']
    track_features['energy'] = features[0]['energy']
    track_features['instrumentalness'] = features[0]['instrumentalness']
    track_features['liveness'] = features[0]['liveness']
    track_features['loudness'] = features[0]['loudness']
    track_features['speechiness'] = features[0]['speechiness']
    track_features['tempo'] = features[0]['tempo']
    track_features['time_signature'] = features[0]['time_signature']

    # track = [name, album, artist, release_date, length, popularity, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
    return track_features


def split_list_into_sublists(lst, max_sublist_size=100):
    return [lst[i:i + max_sublist_size] for i in range(0, len(lst), max_sublist_size)]


def get_batch_track_features(ids, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # Retrieve metadata and audio features for the batch of IDs
            metadata = sp.tracks(ids)
            features = sp.audio_features(ids)
            
            if features is None or any(f is None for f in features):
                raise ValueError(f"Could not retrieve audio features for some tracks in batch {ids}")
            
            # Compile features for each track in the batch
            batch_features = []
            for i, track_id in enumerate(ids):
                track_metadata = metadata['tracks'][i]
                track_features = {
                    'name': track_metadata['name'],
                    'album': track_metadata['album']['name'],
                    'artist': track_metadata['album']['artists'][0]['name'],
                    'release_date': track_metadata['album']['release_date'],
                    'length': track_metadata['duration_ms'],
                    'popularity': track_metadata['popularity'],
                    'acousticness': features[i]['acousticness'],
                    'danceability': features[i]['danceability'],
                    'energy': features[i]['energy'],
                    'instrumentalness': features[i]['instrumentalness'],
                    'liveness': features[i]['liveness'],
                    'loudness': features[i]['loudness'],
                    'speechiness': features[i]['speechiness'],
                    'tempo': features[i]['tempo'],
                    'time_signature': features[i]['time_signature']
                }
                batch_features.append(track_features)
            
            return batch_features
        
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                print(e.headers['retry-after'])
                retry_after = int(e.headers.get("Retry-After", 10))  # Default to 10 second if not provided
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                retries += 1
            else:
                print(e.http_status)
                print(f"An errror occurred: {e}")
                return None
        
        except Exception as e:
            print(f"An error occurreeed: {e}")
            return None
    
    print("Max retries reached. Exiting...")
    return None


def process_tracks(track_ids, batch_size=100):
    all_features = []
    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i:i+batch_size]
        batch_features = get_batch_track_features(batch_ids)
        if batch_features:
            all_features.extend(batch_features)        
        if i > 0 and i % (batch_size * 10) == 0:
            print(f"Processed {i} tracks...")
        time.sleep(1)
        
    return all_features


def get_artist_id(artist_name):
    
    # sometimes the first result is not the correct one, especially
    # for artists with short names, so we'll request more than
    # just a single result, as specified in the limit variable
    limit = 10
    results = sp.search(q=artist_name, type='artist', limit=limit)
    if results['artists']['items']:
         # Check each artist result to ensure we get the correct one
        for artist in results['artists']['items']:
            # print(artist['name'])
            if artist_name == artist['name']:
                return artist['id']
    
    # If no correct artist is found, returns None
    print(f"Artist '{artist_name}' not found in the top {limit} results.")
    return None


def get_artist_albums(artist_name, max_retries=5):
    
    artist_id = get_artist_id(artist_name)
    retries = 0
    
    if artist_id:
        while retries < max_retries:
            try:
                albums = sp.artist_albums(artist_id, 
                                        album_type='album',
                                        limit=50,
                                        )
                album_names = {}
                while albums:
                    for album in albums['items']:
                        album_name = album['name']
                        # exclude live albums and deluxe versions
                        excluded_keywords = {'live', 'deluxe', 'anniversary', 'expanded'}
                        # excluded_keywords = {}
                        if not any(keyword in album_name.lower() for keyword in excluded_keywords):
                            album_names[album_name] = album['id']
                    if albums['next']:
                        albums = sp.next(albums)
                    else:
                        break

                return album_names
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 10))  # Default to 10 second if not provided
                    print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    retries += 1
                else:
                    print(f"An error occurred: {e}")
                    return None
            
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
        
        print("Max retries reached. Exiting...")
        return None            
    else:
        return None
    

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# Function to get artist genres
def get_artist_genre(artist_name):
    # Search for the artist by name
    limit=5
    results = sp.search(q=artist_name, type='artist', limit=limit)
    
    if results['artists']['items']:
        # we make sure the artist is the one we're looking for
        for i in range(limit):
            
            artist = results['artists']['items'][i]  # Get the first search result
            # print(artist['name'])
            if artist_name == artist['name']:
                artist_genre = artist['genres'][0] if artist['genres'] else 'unknown'  # Get genres
                return artist_genre
    else:
        return None

# Function to get artist genres
def get_artist_genres(artist_name):
    
    limit = 5
    # Search for the artist by name
    results = sp.search(q=artist_name, type='artist', limit=limit)
    if results['artists']['items']:
        # sometimes the search gives incorrect results, we make sure the
        # artist is the one we're looking for
        for i in range(limit):
            artist = results['artists']['items'][i]  # Get the first search result
            if artist_name == artist['name']:
                artist_genres = artist['genres']  # Get genres
        # artist_genre = artist['genres'][0]  # Get genres
                return artist_genres
        return None
    else:
        return None
