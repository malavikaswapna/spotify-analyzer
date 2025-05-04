"""
# Spotify Listening Pattern Analyzer
# A comprehensive tool to analyze Spotify listening history, visualize patterns, and recommend new music

## Table of Contents
# 1. Setting Up Spotify API Access
# 2. Data Collection
# 3. Data Analysis
# 4. Data Visualization
# 5. Music Recommendation Engine
# 6. Web Application Integration
"""

import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import networkx as nx
from collections import Counter
import time
import requests
import random
from flask import Flask, request, redirect, jsonify, render_template

########################
# 1. SPOTIFY API SETUP #
########################

def setup_spotify_client():
    """Configure and return Spotify API client with proper authorization"""
    
    # Spotify API credentials - you'll need to register your app in Spotify Developer Dashboard
    SPOTIPY_CLIENT_ID = '17c44f76ab0449459435df090c2d769c'
    SPOTIPY_CLIENT_SECRET = 'c6bdca7f86e34ac5b5a393c3aebf5314'
    SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:8888/callback'
    
    # Set scopes - these determine what data we can access
    scope = "user-library-read user-read-recently-played user-top-read playlist-read-private user-read-currently-playing"
    
    # Create SpotifyOAuth object for authentication
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=scope,
        cache_path=".spotipyoauthcache",
        show_dialog=True
    )
    
    # Create and return authenticated Spotipy client
    return spotipy.Spotify(auth_manager=sp_oauth)

###################
# 2. DATA COLLECTION #
###################

class SpotifyDataCollector:
    """Class to collect and organize Spotify listening data"""
    
    def __init__(self, spotify_client):
        self.sp = spotify_client
        self.user_data = {}
        
    def collect_user_profile(self):
        """Get basic information about the current user"""
        try:
            self.user_data['profile'] = self.sp.current_user()
            print(f"Collected profile data for: {self.user_data['profile']['display_name']}")
            return self.user_data['profile']
        except Exception as e:
            print(f"Error collecting user profile: {e}")
            return None
    
    def collect_recently_played(self, limit=50):
        """Get user's recently played tracks"""
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            recently_played = []
            
            for item in results['items']:
                track = item['track']
                played_at = item['played_at']
                
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'played_at': played_at,
                    'played_at_hour': datetime.strptime(played_at, '%Y-%m-%dT%H:%M:%S.%fZ').hour,
                    'played_at_day': datetime.strptime(played_at, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%A'),
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                }
                recently_played.append(track_data)
            
            self.user_data['recently_played'] = recently_played
            print(f"Collected {len(recently_played)} recently played tracks")
            return recently_played
        except Exception as e:
            print(f"Error collecting recently played tracks: {e}")
            return []
    
    def collect_top_tracks(self, time_range='medium_term', limit=50):
        """Get user's top tracks for a given time range
        time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (years)
        """
        try:
            results = self.sp.current_user_top_tracks(time_range=time_range, limit=limit)
            top_tracks = []
            
            for i, track in enumerate(results['items']):
                track_data = {
                    'rank': i + 1,
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'time_range': time_range
                }
                top_tracks.append(track_data)
            
            self.user_data[f'top_tracks_{time_range}'] = top_tracks
            print(f"Collected {len(top_tracks)} top tracks for {time_range}")
            return top_tracks
        except Exception as e:
            print(f"Error collecting top tracks: {e}")
            return []
    
    def collect_top_artists(self, time_range='medium_term', limit=50):
        """Get user's top artists for a given time range"""
        try:
            results = self.sp.current_user_top_artists(time_range=time_range, limit=limit)
            top_artists = []
            
            for i, artist in enumerate(results['items']):
                artist_data = {
                    'rank': i + 1,
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres'],
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total'],
                    'time_range': time_range
                }
                top_artists.append(artist_data)
            
            self.user_data[f'top_artists_{time_range}'] = top_artists
            print(f"Collected {len(top_artists)} top artists for {time_range}")
            return top_artists
        except Exception as e:
            print(f"Error collecting top artists: {e}")
            return []
    
    def collect_saved_tracks(self, limit=50):
        """Get user's saved tracks"""
        try:
            results = self.sp.current_user_saved_tracks(limit=limit)
            saved_tracks = []
            
            for item in results['items']:
                track = item['track']
                added_at = item['added_at']
                
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'added_at': added_at,
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                }
                saved_tracks.append(track_data)
            
            self.user_data['saved_tracks'] = saved_tracks
            print(f"Collected {len(saved_tracks)} saved tracks")
            return saved_tracks
        except Exception as e:
            print(f"Error collecting saved tracks: {e}")
            return []
    
    def collect_audio_features(self, track_ids):
        """Get audio features for a list of tracks"""
        try:
            # Spotify API can only handle 100 tracks at a time
            audio_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                audio_features.extend(batch_features)
                time.sleep(0.5)  # avoid hitting API rate limits
            
            self.user_data['audio_features'] = audio_features
            print(f"Collected audio features for {len(audio_features)} tracks")
            return audio_features
        except Exception as e:
            print(f"Error collecting audio features: {e}")
            return []
    
    def collect_all_data(self):
        """Collect all available user data"""
        self.collect_user_profile()
        self.collect_recently_played()
        
        for time_range in ['short_term', 'medium_term', 'long_term']:
            self.collect_top_tracks(time_range)
            self.collect_top_artists(time_range)
        
        self.collect_saved_tracks()
        
        # Collect audio features for top tracks
        all_track_ids = []
        for time_range in ['short_term', 'medium_term', 'long_term']:
            track_ids = [track['id'] for track in self.user_data.get(f'top_tracks_{time_range}', [])]
            all_track_ids.extend(track_ids)
        
        # Add recently played and saved tracks
        recent_ids = [track['id'] for track in self.user_data.get('recently_played', [])]
        saved_ids = [track['id'] for track in self.user_data.get('saved_tracks', [])]
        all_track_ids.extend(recent_ids)
        all_track_ids.extend(saved_ids)
        
        # Remove duplicates
        unique_track_ids = list(set(all_track_ids))
        self.collect_audio_features(unique_track_ids)
        
        return self.user_data
    
    def save_data_to_file(self, filename='spotify_data.json'):
        """Save collected data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.user_data, f)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data to file: {e}")
            return False
    
    def load_data_from_file(self, filename='spotify_data.json'):
        """Load data from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.user_data = json.load(f)
            print(f"Data loaded from {filename}")
            return self.user_data
        except Exception as e:
            print(f"Error loading data from file: {e}")
            return None

###################
# 3. DATA ANALYSIS #
###################

class SpotifyDataAnalyzer:
    """Class to analyze Spotify listening data"""
    
    def __init__(self, user_data):
        self.user_data = user_data
        self.analysis_results = {}
    
    def create_track_dataframe(self):
        """Combine all track data into a single dataframe with audio features"""
        # Combine all track sources
        all_tracks = []
        
        # Add top tracks from different time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            top_tracks = self.user_data.get(f'top_tracks_{time_range}', [])
            for track in top_tracks:
                track['source'] = f'top_{time_range}'
                all_tracks.append(track)
        
        # Add recently played
        for track in self.user_data.get('recently_played', []):
            track['source'] = 'recently_played'
            all_tracks.append(track)
        
        # Add saved tracks
        for track in self.user_data.get('saved_tracks', []):
            track['source'] = 'saved'
            all_tracks.append(track)
        
        # Create dataframe
        tracks_df = pd.DataFrame(all_tracks)
        
        # Create audio features dataframe
        audio_features = self.user_data.get('audio_features', [])
        if audio_features:
            features_df = pd.DataFrame(audio_features)
            # Remove duplicates - keep just the first instance of each track
            tracks_df = tracks_df.drop_duplicates(subset='id')
            
            # Merge tracks with audio features
            tracks_df = pd.merge(tracks_df, features_df, left_on='id', right_on='id', how='left')
        
        self.analysis_results['tracks_df'] = tracks_df
        return tracks_df
    
    def create_artist_dataframe(self):
        """Combine all artist data into a single dataframe"""
        all_artists = []
        
        # Add top artists from different time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            top_artists = self.user_data.get(f'top_artists_{time_range}', [])
            for artist in top_artists:
                artist['source'] = f'top_{time_range}'
                all_artists.append(artist)
        
        # Create dataframe
        artists_df = pd.DataFrame(all_artists)
        # Remove duplicates - keep higher ranked instance
        artists_df = artists_df.sort_values('rank').drop_duplicates(subset='id')
        
        self.analysis_results['artists_df'] = artists_df
        return artists_df
    
    def analyze_listening_patterns(self):
        """Analyze when user listens to music"""
        recently_played = self.user_data.get('recently_played', [])
        if not recently_played:
            return None
        
        df = pd.DataFrame(recently_played)
        
        # Convert played_at to datetime
        df['played_at'] = pd.to_datetime(df['played_at'])
        
        # Extract hour and day of week
        df['hour'] = df['played_at'].dt.hour
        df['day_of_week'] = df['played_at'].dt.dayofweek
        df['day_name'] = df['played_at'].dt.day_name()
        
        # Count plays by hour
        hour_counts = df['hour'].value_counts().sort_index()
        
        # Count plays by day of week (0=Monday, 6=Sunday)
        day_counts = df['day_of_week'].value_counts().sort_index()
        day_name_counts = df['day_name'].value_counts()
        
        # Group by hour and day for heatmap
        hour_day_counts = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        results = {
            'hour_counts': hour_counts.to_dict(),
            'day_counts': day_counts.to_dict(),
            'day_name_counts': day_name_counts.to_dict(),
            'hour_day_counts': hour_day_counts.values.tolist(),
            'hour_day_labels': {
                'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'hours': list(range(24))
            }
        }
        
        self.analysis_results['listening_patterns'] = results
        return results

    def analyze_audio_features(self):
        """Analyze audio features of tracks"""
        if 'tracks_df' not in self.analysis_results:
            self.create_track_dataframe()

        tracks_df = self.analysis_results['tracks_df']
        
        # Select relevant columns
        features = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        # Check if any of the feature columns exist in the dataframe
        available_features = [f for f in features if f in tracks_df.columns]
        if not available_features:
            # Return empty results if no audio features are available
            results = {
                'feature_means': {},
                'feature_distributions': {},
                'mood_counts': {},
                'pca_data': [],
                'cluster_centers': [],
                'clusters': [],
                'track_info': []
            }
            self.analysis_results['audio_features'] = results
            return results
        
        # Use only available features
        feature_df = tracks_df[available_features].copy()

        feature_df = feature_df.fillna(feature_df.mean())
        
        # Calculate averages
        feature_means = feature_df.mean().to_dict()
        
        # Calculate distributions
        feature_distributions = {}
        for feature in available_features:
            feature_distributions[feature] = tracks_df[feature].describe().to_dict()
        
        # Check if we have enough features and data for clustering
        if len(available_features) >= 2 and len(feature_df) >= 5:
            # Cluster tracks based on audio features
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_df)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=min(5, len(feature_df)), random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Add PCA and cluster results to tracks dataframe
            tracks_df['pca_x'] = features_pca[:, 0]
            tracks_df['pca_y'] = features_pca[:, 1]
            tracks_df['cluster'] = clusters
            
            # Get cluster centers transformed into PCA space
            cluster_centers_scaled = kmeans.cluster_centers_
            cluster_centers_pca = pca.transform(cluster_centers_scaled)
        else:
            # Not enough data for meaningful clustering
            features_pca = []
            cluster_centers_pca = []
            clusters = []
        
        # Analyze mood patterns based on valence and energy
        mood_counts = {}
        if 'valence' in tracks_df.columns and 'energy' in tracks_df.columns:
            # Define mood quadrants based on valence and energy
            def get_mood(valence, energy):
                if valence >= 0.5 and energy >= 0.5:
                    return 'Happy/Euphoric'
                elif valence >= 0.5 and energy < 0.5:
                    return 'Calm/Peaceful'
                elif valence < 0.5 and energy >= 0.5:
                    return 'Angry/Intense'
                else:
                    return 'Sad/Depressed'
            
            tracks_df['mood'] = tracks_df.apply(lambda x: get_mood(x['valence'], x['energy']), axis=1)
            mood_counts = tracks_df['mood'].value_counts().to_dict()
        
        # Prepare track info with available data
        if len(available_features) >= 2 and len(feature_df) >= 5:
            # Include clustering info if available
            track_info = tracks_df[['name', 'artist']].copy()
            if 'pca_x' in tracks_df.columns:
                track_info['pca_x'] = tracks_df['pca_x']
                track_info['pca_y'] = tracks_df['pca_y']
            if 'cluster' in tracks_df.columns:
                track_info['cluster'] = tracks_df['cluster']
            if 'mood' in tracks_df.columns:
                track_info['mood'] = tracks_df['mood']
            track_info = track_info.to_dict('records')
        else:
            # Basic track info without clustering
            track_info = tracks_df[['name', 'artist']].to_dict('records')
        
        results = {
            'feature_means': feature_means,
            'feature_distributions': feature_distributions,
            'mood_counts': mood_counts,
            'pca_data': features_pca.tolist() if isinstance(features_pca, np.ndarray) else [],
            'cluster_centers': cluster_centers_pca.tolist() if isinstance(cluster_centers_pca, np.ndarray) else [],
            'clusters': clusters.tolist() if isinstance(clusters, np.ndarray) else [],
            'track_info': track_info
        }
        
        self.analysis_results['audio_features'] = results
        return results

    
    def analyze_genre_preferences(self):
        """Analyze genre preferences based on top artists"""
        if 'artists_df' not in self.analysis_results:
            self.create_artist_dataframe()
        
        artists_df = self.analysis_results['artists_df']
        
        # Extract all genres from artists
        all_genres = []
        for genres in artists_df['genres'].tolist():
            if genres:
                all_genres.extend(genres)
        
        # Count genre occurrences
        genre_counts = Counter(all_genres)
        
        # Get top genres
        top_genres = genre_counts.most_common(20)
        
        # Create genre network
        genre_network = nx.Graph()
        
        # Add nodes for each genre
        for genre, count in genre_counts.items():
            genre_network.add_node(genre, count=count)
        
        # Add edges between genres that appear for the same artist
        for genres in artists_df['genres'].tolist():
            if genres and len(genres) > 1:
                for i in range(len(genres)):
                    for j in range(i+1, len(genres)):
                        if genre_network.has_edge(genres[i], genres[j]):
                            genre_network[genres[i]][genres[j]]['weight'] += 1
                        else:
                            genre_network.add_edge(genres[i], genres[j], weight=1)
        
        # Convert network to JSON-serializable format
        network_data = {
            'nodes': [{'id': node, 'count': data['count']} for node, data in genre_network.nodes(data=True)],
            'links': [{'source': u, 'target': v, 'weight': data['weight']} 
                      for u, v, data in genre_network.edges(data=True)]
        }
        
        results = {
            'top_genres': top_genres,
            'genre_counts': dict(genre_counts),
            'network_data': network_data
        }
        
        self.analysis_results['genre_preferences'] = results
        return results
    
    def analyze_artist_connections(self):
        """Analyze connections between artists based on common genres"""
        if 'artists_df' not in self.analysis_results:
            self.create_artist_dataframe()
        
        artists_df = self.analysis_results['artists_df']
        
        # Create artist network
        artist_network = nx.Graph()
        
        # Add nodes for each artist
        for _, artist in artists_df.iterrows():
            artist_network.add_node(artist['id'], 
                                    name=artist['name'], 
                                    popularity=artist['popularity'],
                                    genres=artist['genres'])
        
        # Create dictionary of genres to artists
        genre_to_artists = {}
        for _, artist in artists_df.iterrows():
            for genre in artist['genres']:
                if genre not in genre_to_artists:
                    genre_to_artists[genre] = []
                genre_to_artists[genre].append(artist['id'])
        
        # Add edges between artists that share genres
        for genre, artist_ids in genre_to_artists.items():
            if len(artist_ids) > 1:
                for i in range(len(artist_ids)):
                    for j in range(i+1, len(artist_ids)):
                        if artist_network.has_edge(artist_ids[i], artist_ids[j]):
                            artist_network[artist_ids[i]][artist_ids[j]]['weight'] += 1
                            artist_network[artist_ids[i]][artist_ids[j]]['shared_genres'].append(genre)
                        else:
                            artist_network.add_edge(artist_ids[i], artist_ids[j], 
                                                   weight=1, 
                                                   shared_genres=[genre])
        
        # Convert network to JSON-serializable format
        network_data = {
            'nodes': [{'id': node, 
                      'name': data['name'], 
                      'popularity': data['popularity'],
                      'genres': data['genres']} 
                     for node, data in artist_network.nodes(data=True)],
            'links': [{'source': u, 
                      'target': v, 
                      'weight': data['weight'],
                      'shared_genres': data['shared_genres']} 
                     for u, v, data in artist_network.edges(data=True)]
        }
        
        results = {
            'network_data': network_data
        }
        
        self.analysis_results['artist_connections'] = results
        return results
    
    def run_all_analyses(self):
        """Run all analysis methods"""
        self.create_track_dataframe()
        self.create_artist_dataframe()
        self.analyze_listening_patterns()
        self.analyze_audio_features()
        self.analyze_genre_preferences()
        self.analyze_artist_connections()
        return self.analysis_results
    
    def save_analysis_to_file(self, filename='spotify_analysis.json'):
        """Save analysis results to JSON file"""
        try:
            # Create a copy to avoid modifying the original
            serializable_results = {}
            for key, value in self.analysis_results.items():
                # Convert DataFrames to records
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict('records')
                else:
                    serializable_results[key] = value

            with open(filename, 'w') as f:
                json.dump(serializable_results, f)
            print(f"Analysis saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving analysis to file: {e}")
            return False
            
    
    def load_analysis_from_file(self, filename='spotify_analysis.json'):
        """Load analysis results from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.analysis_results = json.load(f)
            print(f"Analysis loaded from {filename}")
            return self.analysis_results
        except Exception as e:
            print(f"Error loading analysis from file: {e}")
            return None

###########################
# 4. RECOMMENDATION ENGINE #
###########################

class SpotifyRecommender:
    """Class to generate music recommendations based on user data"""
    
    def __init__(self, spotify_client, user_data, analysis_results):
        self.sp = spotify_client
        self.user_data = user_data
        self.analysis_results = analysis_results
    
    def get_recommendations_by_top_tracks(self, time_range='medium_term', limit=20):
        """Get recommendations based on top tracks"""
        try:
            top_tracks = self.user_data.get(f'top_tracks_{time_range}', [])
            if not top_tracks:
                return []
            
            # Use top 5 tracks as seeds
            seed_tracks = [track['id'] for track in top_tracks[:5]]
            
            # Get recommendations
            recommendations = self.sp.recommendations(seed_tracks=seed_tracks, limit=limit)
            
            # Format recommendations
            formatted_recommendations = []
            for track in recommendations['tracks']:
                rec = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'recommendation_source': f'top_tracks_{time_range}'
                }
                formatted_recommendations.append(rec)
            
            return formatted_recommendations
        except Exception as e:
            print(f"Error getting recommendations by top tracks: {e}")
            return []
    
    def get_recommendations_by_top_artists(self, time_range='medium_term', limit=20):
        """Get recommendations based on top artists"""
        try:
            top_artists = self.user_data.get(f'top_artists_{time_range}', [])
            if not top_artists:
                return []
            
            # Use top 5 artists as seeds
            seed_artists = [artist['id'] for artist in top_artists[:5]]
            
            # Get recommendations
            recommendations = self.sp.recommendations(seed_artists=seed_artists, limit=limit)
            
            # Format recommendations
            formatted_recommendations = []
            for track in recommendations['tracks']:
                rec = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'recommendation_source': f'top_artists_{time_range}'
                }
                formatted_recommendations.append(rec)
            
            return formatted_recommendations
        except Exception as e:
            print(f"Error getting recommendations by top artists: {e}")
            return []
    
    def get_recommendations_by_genres(self, limit=20):
        """Get recommendations based on top genres"""
        try:
            # Get top genres from analysis
            genre_preferences = self.analysis_results.get('genre_preferences', {})
            top_genres = genre_preferences.get('top_genres', [])
            
            if not top_genres:
                return []
            
            # Use top 5 genres as seeds
            seed_genres = [genre for genre, _ in top_genres[:5]]
            
            # Get available genre seeds
            available_genres = self.sp.recommendation_genre_seeds()['genres']
            
            # Filter genres that are available in Spotify's genre seeds
            valid_seed_genres = [genre for genre in seed_genres if genre in available_genres]
            
            if not valid_seed_genres:
                return []
            
            # Get recommendations
            recommendations = self.sp.recommendations(seed_genres=valid_seed_genres[:5], limit=limit)
            
            # Format recommendations
            formatted_recommendations = []
            for track in recommendations['tracks']:
                rec = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'recommendation_source': 'top_genres'
                }
                formatted_recommendations.append(rec)
            
            return formatted_recommendations
        except Exception as e:
            print(f"Error getting recommendations by genres: {e}")
            return []
    
    def get_recommendations_by_audio_features(self, limit=20):
        """Get recommendations based on preferred audio features"""
        try:
            # Get audio feature means from analysis
            audio_features = self.analysis_results.get('audio_features', {})
            feature_means = audio_features.get('feature_means', {})
            
            if not feature_means:
                return []
            
            # Get top tracks as seeds
            top_tracks = self.user_data.get('top_tracks_medium_term', [])
            if not top_tracks:
                return []
            
            seed_tracks = [track['id'] for track in top_tracks[:5]]
            
            # Format target audio features for recommendations
            target_features = {
                'target_danceability': feature_means.get('danceability', 0.5),
                'target_energy': feature_means.get('energy', 0.5),
                'target_valence': feature_means.get('valence', 0.5),
                'target_acousticness': feature_means.get('acousticness', 0.5),
                'target_instrumentalness': feature_means.get('instrumentalness', 0.5),
            }
            
            # Get recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks, 
                limit=limit,
                **target_features
            )
            
            # Format recommendations
            formatted_recommendations = []
            for track in recommendations['tracks']:
                rec = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'recommendation_source': 'audio_features'
                }
                formatted_recommendations.append(rec)
            
            return formatted_recommendations
        except Exception as e:
            print(f"Error getting recommendations by audio features: {e}")
            return []
    
    def get_recommendations_by_mood(self, mood, limit=20):
        """Get recommendations based on mood preferences"""
        try:
            # Define target features for different moods
            mood_targets = {
                'Happy/Euphoric': {
                    'target_valence': 0.8,
                    'target_energy': 0.8,
                    'target_danceability': 0.7
                },
                'Calm/Peaceful': {
                    'target_valence': 0.7,
                    'target_energy': 0.3,
                    'target_acousticness': 0.8
                },
                'Angry/Intense': {
                    'target_valence': 0.3,
                    'target_energy': 0.9,
                    'target_loudness': 0.8
                },
                'Sad/Depressed': {
                    'target_valence': 0.2,
                    'target_energy': 0.3,
                    'target_tempo': 0.3
                }
            }
            
            # Get target features for the requested mood
            target_features = mood_targets.get(mood, mood_targets['Happy/Euphoric'])
            
            # Get top tracks as seeds
            top_tracks = self.user_data.get('top_tracks_medium_term', [])
            if not top_tracks:
                return []
            
            # Get tracks that match the requested mood
            if 'tracks_df' in self.analysis_results:
                tracks_df = self.analysis_results['tracks_df']
                mood_tracks = tracks_df[tracks_df['mood'] == mood]
                if len(mood_tracks) >= 5:
                    seed_tracks = mood_tracks['id'].tolist()[:5]
                else:
                    seed_tracks = [track['id'] for track in top_tracks[:5]]
            else:
                seed_tracks = [track['id'] for track in top_tracks[:5]]
            
            # Get recommendations
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks, 
                limit=limit,
                **target_features
            )
            
            # Format recommendations
            formatted_recommendations = []
            for track in recommendations['tracks']:
                rec = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'album_id': track['album']['id'],
                    'popularity': track['popularity'],
                    'recommendation_source': f'mood_{mood}'
                }
                formatted_recommendations.append(rec)
            
            return formatted_recommendations
        except Exception as e:
            print(f"Error getting recommendations by mood: {e}")
            return []
    
    def get_all_recommendations(self, limit=20):
        """Get all types of recommendations"""
        all_recommendations = {}
        
        # Get recommendations by top tracks for different time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            recs = self.get_recommendations_by_top_tracks(time_range, limit)
            all_recommendations[f'top_tracks_{time_range}'] = recs
        
        # Get recommendations by top artists for different time ranges
        for time_range in ['short_term', 'medium_term', 'long_term']:
            recs = self.get_recommendations_by_top_artists(time_range, limit)
            all_recommendations[f'top_artists_{time_range}'] = recs
        
        # Get recommendations by genres
        recs = self.get_recommendations_by_genres(limit)
        all_recommendations['genres'] = recs
        
        # Get recommendations by audio features
        recs = self.get_recommendations_by_audio_features(limit)
        all_recommendations['audio_features'] = recs
        
        # Get recommendations by mood
        for mood in ['Happy/Euphoric', 'Calm/Peaceful', 'Angry/Intense', 'Sad/Depressed']:
            recs = self.get_recommendations_by_mood(mood, limit)
            all_recommendations[f'mood_{mood}'] = recs
        
        return all_recommendations
    
    def save_recommendations_to_file(self, recommendations, filename='spotify_recommendations.json'):
        """Save recommendations to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(recommendations, f)
            print(f"Recommendations saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving recommendations to file: {e}")
            return False

###############################
# 5. DATA VISUALIZATION MODULE #
###############################

class SpotifyDataVisualizer:
    """Class to create visualizations for Spotify data analysis"""
    
    def __init__(self, user_data, analysis_results):
        self.user_data = user_data if user_data is not None else {}
        self.analysis_results = analysis_results if analysis_results is not None else {}
    
    def generate_charts_data(self):
        """Generate data for all charts in the dashboard"""
        charts_data = {}
        
        # 1. Listening Activity Chart (by hour)
        if 'listening_patterns' in self.analysis_results:
            patterns = self.analysis_results['listening_patterns']
            hour_counts = patterns.get('hour_counts', {})
            
            hour_data = []
            for hour in range(24):
                hour_data.append({
                    'hour': hour,
                    'count': hour_counts.get(str(hour), 0)
                })
            
            charts_data['listening_by_hour'] = hour_data
        
        # 2. Listening Activity Chart (by day)
        if 'listening_patterns' in self.analysis_results:
            patterns = self.analysis_results['listening_patterns']
            day_name_counts = patterns.get('day_name_counts', {})
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_data = []
            for day in day_order:
                day_data.append({
                    'day': day,
                    'count': day_name_counts.get(day, 0)
                })
            
            charts_data['listening_by_day'] = day_data
        
        # 3. Genre Distribution Chart
        if 'genre_preferences' in self.analysis_results:
            genre_prefs = self.analysis_results['genre_preferences']
            top_genres = genre_prefs.get('top_genres', [])
            
            genre_data = []
            for genre, count in top_genres:
                genre_data.append({
                    'genre': genre,
                    'count': count
                })
            
            charts_data['genre_distribution'] = genre_data
        
        # 4. Audio Features Radar Chart
        if 'audio_features' in self.analysis_results:
            features = self.analysis_results['audio_features']
            feature_means = features.get('feature_means', {})
            
            # Select relevant features for radar chart
            radar_features = ['danceability', 'energy', 'valence', 'acousticness', 
                             'instrumentalness', 'speechiness', 'liveness']
            
            radar_data = []
            for feature in radar_features:
                radar_data.append({
                    'feature': feature.capitalize(),
                    'value': feature_means.get(feature, 0)
                })
            
            charts_data['audio_features_radar'] = radar_data
        
        # 5. Mood Distribution Chart
        if 'audio_features' in self.analysis_results:
            features = self.analysis_results['audio_features']
            mood_counts = features.get('mood_counts', {})
            
            mood_data = []
            for mood, count in mood_counts.items():
                mood_data.append({
                    'mood': mood,
                    'count': count
                })
            
            charts_data['mood_distribution'] = mood_data
        
        # 6. Track Clustering Scatter Plot
        if 'audio_features' in self.analysis_results:
            features = self.analysis_results['audio_features']
            track_info = features.get('track_info', [])
            
            scatter_data = []
            for track in track_info:
                scatter_data.append({
                    'x': track.get('pca_x', 0),
                    'y': track.get('pca_y', 0),
                    'name': track.get('name', ''),
                    'artist': track.get('artist', ''),
                    'cluster': track.get('cluster', 0),
                    'mood': track.get('mood', '')
                })
            
            charts_data['track_clustering'] = scatter_data
        
        # 7. Artist Network Graph Data
        if 'artist_connections' in self.analysis_results:
            artist_connections = self.analysis_results['artist_connections']
            network_data = artist_connections.get('network_data', {})
            
            charts_data['artist_network'] = network_data
        
        # 8. Genre Network Graph Data
        if 'genre_preferences' in self.analysis_results:
            genre_prefs = self.analysis_results['genre_preferences']
            network_data = genre_prefs.get('network_data', {})
            
            charts_data['genre_network'] = network_data
        
        return charts_data
    
    def generate_chart_js_config(self):
        """Generate Chart.js configuration for dashboard"""
        charts_data = self.generate_charts_data()
        
        chart_configs = {}
        
        # 1. Listening Activity by Hour (Line Chart)
        if 'listening_by_hour' in charts_data:
            hour_data = charts_data['listening_by_hour']
            
            hour_labels = [str(item['hour']) + ':00' for item in hour_data]
            hour_values = [item['count'] for item in hour_data]
            
            hour_config = {
                'type': 'line',
                'data': {
                    'labels': hour_labels,
                    'datasets': [{
                        'label': 'Tracks Played',
                        'data': hour_values,
                        'backgroundColor': 'rgba(29, 185, 84, 0.2)',
                        'borderColor': 'rgba(29, 185, 84, 1)',
                        'borderWidth': 2,
                        'pointBackgroundColor': 'rgba(29, 185, 84, 1)',
                        'tension': 0.4
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Listening Activity by Hour'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Number of Tracks'
                            }
                        },
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Hour of Day'
                            }
                        }
                    }
                }
            }
            
            chart_configs['listeningByHourChart'] = hour_config
        
        # 2. Listening Activity by Day (Bar Chart)
        if 'listening_by_day' in charts_data:
            day_data = charts_data['listening_by_day']
            
            day_labels = [item['day'] for item in day_data]
            day_values = [item['count'] for item in day_data]
            
            day_config = {
                'type': 'bar',
                'data': {
                    'labels': day_labels,
                    'datasets': [{
                        'label': 'Tracks Played',
                        'data': day_values,
                        'backgroundColor': 'rgba(29, 185, 84, 0.7)',
                        'borderColor': 'rgba(29, 185, 84, 1)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Listening Activity by Day'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Number of Tracks'
                            }
                        },
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Day of Week'
                            }
                        }
                    }
                }
            }
            
            chart_configs['listeningByDayChart'] = day_config
        
        # 3. Genre Distribution (Horizontal Bar Chart)
        if 'genre_distribution' in charts_data:
            genre_data = charts_data['genre_distribution']
            # Limit to top 10 genres
            genre_data = genre_data[:10]
            
            genre_labels = [item['genre'] for item in genre_data]
            genre_values = [item['count'] for item in genre_data]
            
            genre_config = {
                'type': 'bar',
                'data': {
                    'labels': genre_labels,
                    'datasets': [{
                        'label': 'Count',
                        'data': genre_values,
                        'backgroundColor': [
                            'rgba(29, 185, 84, 0.8)',
                            'rgba(30, 215, 96, 0.8)',
                            'rgba(65, 195, 102, 0.8)',
                            'rgba(100, 175, 108, 0.8)',
                            'rgba(135, 155, 114, 0.8)',
                            'rgba(170, 135, 120, 0.8)',
                            'rgba(205, 115, 126, 0.8)',
                            'rgba(240, 95, 132, 0.8)',
                            'rgba(255, 75, 138, 0.8)',
                            'rgba(255, 55, 144, 0.8)'
                        ],
                        'borderColor': 'rgba(29, 185, 84, 1)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'indexAxis': 'y',
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Top Genres'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'x': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Number of Artists'
                            }
                        }
                    }
                }
            }
            
            chart_configs['genreDistributionChart'] = genre_config
        
        # 4. Audio Features (Radar Chart)
        if 'audio_features_radar' in charts_data:
            radar_data = charts_data['audio_features_radar']
            
            radar_labels = [item['feature'] for item in radar_data]
            radar_values = [item['value'] for item in radar_data]
            
            radar_config = {
                'type': 'radar',
                'data': {
                    'labels': radar_labels,
                    'datasets': [{
                        'label': 'Your Music Profile',
                        'data': radar_values,
                        'backgroundColor': 'rgba(29, 185, 84, 0.2)',
                        'borderColor': 'rgba(29, 185, 84, 1)',
                        'borderWidth': 2,
                        'pointBackgroundColor': 'rgba(29, 185, 84, 1)',
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Audio Features Profile'
                        }
                    },
                    'scales': {
                        'r': {
                            'min': 0,
                            'max': 1,
                            'ticks': {
                                'stepSize': 0.2
                            }
                        }
                    }
                }
            }
            
            chart_configs['audioFeaturesRadarChart'] = radar_config
        
        # 5. Mood Distribution (Pie Chart)
        if 'mood_distribution' in charts_data:
            mood_data = charts_data['mood_distribution']
            
            mood_labels = [item['mood'] for item in mood_data]
            mood_values = [item['count'] for item in mood_data]
            
            mood_config = {
                'type': 'pie',
                'data': {
                    'labels': mood_labels,
                    'datasets': [{
                        'data': mood_values,
                        'backgroundColor': [
                            'rgba(255, 99, 132, 0.7)',
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 206, 86, 0.7)',
                            'rgba(75, 192, 192, 0.7)'
                        ],
                        'borderColor': 'rgba(255, 255, 255, 0.7)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Mood Distribution'
                        }
                    }
                }
            }
            
            chart_configs['moodDistributionChart'] = mood_config
        
        # Make sure we're correctly formatting track clustering data
        if 'track_clustering' in charts_data:
            # This is what you currently have:
            scatter_data = charts_data['track_clustering']
    
            # We need to format it as expected by the chart.js configuration:
            # Create a properly formatted scatter plot dataset
            # Group tracks by cluster
            cluster_groups = {}
            for track in scatter_data:
                cluster = track.get('cluster', 0)
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                    cluster_groups[cluster].append({
                        'x': track.get('x', 0),
                        'y': track.get('y', 0),
                        'name': track.get('name', ''),
                        'artist': track.get('artist', '')
                    })
    
            # Format for chart.js
            cluster_colors = [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)'
            ]
            
            clustering_data = []
            for i, (cluster, tracks) in enumerate(cluster_groups.items()):
                clustering_data.append({
                    'cluster': cluster,
                    'color': cluster_colors[i % len(cluster_colors)],
                    'tracks': tracks
                })
                
                chart_configs['trackClusteringData'] = clustering_data
        
        return chart_configs
    
    def generate_d3_network_config(self):
        """Generate D3.js configuration for network visualizations"""
        charts_data = self.generate_charts_data()
        
        network_configs = {}
        
        # 1. Artist Network
        if 'artist_network' in charts_data:
            network_configs['artistNetwork'] = charts_data['artist_network']
        
        # 2. Genre Network
        if 'genre_network' in charts_data:
            network_configs['genreNetwork'] = charts_data['genre_network']
        
        return network_configs
    
    def save_visualizations_to_file(self, charts_data, filename='spotify_visualizations.json'):
        """Save visualization data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(charts_data, f)
            print(f"Visualization data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving visualization data to file: {e}")
            return False

def create_artist_dataframe(self):
    """Combine all artist data into a single dataframe"""
    all_artists = []
    
    # Add top artists from different time ranges
    for time_range in ['short_term', 'medium_term', 'long_term']:
        top_artists = self.user_data.get(f'top_artists_{time_range}', [])
        for artist in top_artists:
            artist['source'] = f'top_{time_range}'
            all_artists.append(artist)
    
    # Create dataframe
    artists_df = pd.DataFrame(all_artists)
    
    # Check if dataframe is empty or doesn't have 'rank' column
    if artists_df.empty or 'rank' not in artists_df.columns:
        # Create and return empty dataframe with required columns
        empty_df = pd.DataFrame(columns=['id', 'name', 'genres', 'popularity', 'source'])
        self.analysis_results['artists_df'] = empty_df
        return empty_df
    
    # Only sort if we have data and the 'rank' column exists
    artists_df = artists_df.sort_values('rank').drop_duplicates(subset='id')
    
    self.analysis_results['artists_df'] = artists_df
    return artists_df

##########################
# 6. WEB APPLICATION MODULE #
##########################

class SpotifyWebApp:
    """Class to create a web application for Spotify data analysis"""
    
    def __init__(self, user_data={}, analysis_results={}, recommendations={}):
        self.app = Flask(__name__, static_folder='static', static_url_path='/static')
        self.user_data = user_data
        self.analysis_results = analysis_results
        self.recommendations = recommendations

        self.fix_album_cover_urls()
        
        # Initialize visualizer
        self.visualizer = SpotifyDataVisualizer(user_data, analysis_results)
        
        # Setup routes
        self.setup_routes()
            
    def fix_album_cover_urls(self):
        """Fix missing album cover URLs in recommendations"""
        # Loop through all recommendation categories
        for category, rec_list in self.recommendations.items():
            for track in rec_list:
                # If album_cover_url is missing or empty
                if 'album_cover_url' not in track or not track['album_cover_url']:
                    # Assign a cover based on album name
                    album_name = track.get('album', '').lower()
                    if 'midnights' in album_name:
                        track['album_cover_url'] = '/static/images/albums/anti-hero.png'
                    elif 'after hours' in album_name:
                        track['album_cover_url'] = '/static/images/albums/blinding.png'
                    elif 'save your tears' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/savetears.jpg'
                    elif 'stay' in album_name:
                        track['album_cover_url'] = '/static/images/albums/stay.png'
                    elif 'sour' in album_name or 'driver' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/driver.png'
                    elif 'good 4 u' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/good.png'
                    elif 'montero' in album_name or 'industry baby' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/call.jpg'
                    elif 'dreamland' in album_name or 'heat waves' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/heat.png'
                    elif 'divide' in album_name or '÷' in album_name or 'shape of you' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/shape.png'
                    elif 'hot pink' in album_name or 'say so' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/say.png'
                    elif 'parachutes' in album_name or 'yellow' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/yellow.jpg'
                    elif 'scorpion' in album_name or 'god\'s plan' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/god.jpg'
                    elif 'heroes' in album_name or 'creepin' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/creep.jpg'
                    elif 'starboy' in album_name or 'die for you' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/die.png'
                    elif 'harry\'s house' in album_name or 'as it was' in track.get('name', '').lower():
                        track['album_cover_url'] = '/static/images/albums/as.png'
                    else:
                        # Default fallback
                        track['album_cover_url'] = '/static/images/albums/blinding.png'
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Home page"""
            user_profile = self.user_data.get('profile', {})
            return render_template('index.html', user=user_profile)
        
        @self.app.route('/dashboard')
        def dashboard():
            """Dashboard page with visualizations"""
            user_profile = self.user_data.get('profile', {})
            
            # Generate chart configurations
            chart_configs = self.visualizer.generate_chart_js_config()
            network_configs = self.visualizer.generate_d3_network_config()
            
            return render_template('dashboard.html', 
                                 user=user_profile,
                                 chart_configs=chart_configs,
                                 network_configs=network_configs)
        
        @self.app.route('/recommendations')
        def recommendations():
            """Recommendations page"""
            user_profile = self.user_data.get('profile', {})
            return render_template('recommendations.html', 
                                 user=user_profile, 
                                 recommendations=self.recommendations)
        
        @self.app.route('/api/data')
        def api_data():
            """API endpoint for raw data"""
            return jsonify({
                'user_data': self.user_data,
                'analysis_results': self.analysis_results,
                'recommendations': self.recommendations
            })
        
        @self.app.route('/api/charts')
        def api_charts():
            """API endpoint for chart data"""
            charts_data = self.visualizer.generate_charts_data()
            return jsonify(charts_data)
        
        @self.app.route('/api/charts/config')
        def api_chart_configs():
            """API endpoint for chart configurations"""
            chart_configs = self.visualizer.generate_chart_js_config()
            return jsonify(chart_configs)
        
        @self.app.route('/api/networks')
        def api_networks():
            """API endpoint for network data"""
            network_configs = self.visualizer.generate_d3_network_config()
            return jsonify(network_configs)
    
    def create_templates(self):
        """Create HTML templates for the web application"""
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # 1. Base template
        base_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Spotify Listening Pattern Analyzer{% endblock %}</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <!-- Custom CSS -->
    <style>
        :root {
            --spotify-green: #1DB954;
            --spotify-black: #191414;
            --spotify-white: #FFFFFF;
            --spotify-gray: #535353;
        }
        
        body {
            font-family: 'Circular', 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--spotify-black);
            color: var(--spotify-white);
        }
        
        .navbar {
            background-color: var(--spotify-black);
            border-bottom: 1px solid #333;
        }

        .nav-tabs {
            border-bottom: 1px solid #444;
        }
        
        .navbar-brand {
            color: var(--spotify-green) !important;
            font-weight: bold;
        }
        
        .nav-link {
            color: var(--spotify-white) !important;
        }
        
        .nav-link:hover {
            color: var(--spotify-green) !important;
        }
        
        .nav-tabs .nav-link {
            color: var(--spotify-white);
            transition: all 0.2s ease;
            border-radius: 0.25rem 0.25rem 0 0;
            margin-right: 4px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            border: 1px solid #444;
            border-bottom: none;
        }
        
        .nav-tabs .nav-link.active {
            color: var(--spotify-white);
            background-color: var(--spotify-green);
            border-color: var(--spotify-green);
            transform: translateY(-4px);
            border-bottom: 3px solid white;
            font-weight: bold;
        }

        .nav-tabs .nav-link:hover {
            background-color: #444;
            border-color: #555;
        }

        /* Tab content styling */
        .tab-content {
            background-color: #282828;
            border-radius: 0 0 8px 8px;
            padding: 20px;
            border: 1px solid #444;
            border-top: none;
        }

        /* Section headings */
        .section-heading {
            font-weight: bold;
            margin-bottom: 1.5rem;
            padding: 8px 16px;
            background-color: #282828;
            border-radius: 4px;
            display: inline-block;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Mood-specific background colors */
        #happy-tab {
            background-color: rgba(255, 99, 132, 0.7);
        }

        #happy-tab.active, #happy-tab:hover {
            background-color: rgba(255, 99, 132, 1);
            border-color: rgba(255, 99, 132, 1);
            color: white;
            box-shadow: 0 -2px 10px rgba(255, 99, 132, 0.5);
        }

        #calm-tab {
            background-color: rgba(54, 162, 235, 0.7);
        }

        #calm-tab.active, #calm-tab:hover {
            background-color: rgba(54, 162, 235, 1);
            border-color: rgba(54, 162, 235, 1);
            color: white;
            box-shadow: 0 -2px 10px rgba(54, 162, 235, 0.5);
        }

        #angry-tab {
            background-color: rgba(255, 206, 86, 0.7);
        }

        #angry-tab.active, #angry-tab:hover {
            background-color: rgba(255, 206, 86, 0.9);
            border-color: rgba(255, 206, 86, 1);
            color: black;
            box-shadow: 0 -2px 10px rgba(255, 206, 86, 0.5);
        }

        #sad-tab {
           background-color: rgba(75, 192, 192, 0.7);
        }

        #sad-tab.active, #sad-tab:hover {
           background-color: rgba(75, 192, 192, 0.9);
           border-color: rgba(75, 192, 192, 1);
           color: white;
           box-shadow: 0 -2px 10px rgba(75, 192, 192, 0.5);
        }
        
        .btn-spotify {
            background-color: var(--spotify-green);
            color: var(--spotify-white);
            border: none;
        }
        
        .btn-spotify:hover {
            background-color: #1ED760;
            color: var(--spotify-white);
        }
        
        .card {
            background-color: #282828;
            border: none;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .card-header {
            background-color: #333;
            border-bottom: none;
            font-weight: bold;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }
        
        .network-container {
            position: relative;
            height: 500px;
            width: 100%;
            border: 1px solid #333;
            border-radius: 8px;
        }
        
        footer {
            background-color: var(--spotify-black);
            border-top: 1px solid #333;
            color: var(--spotify-gray);
            padding: 20px 0;
            margin-top: 50px;
        }
        
        /* Additional styles for recommendation cards */
        .recommendation-card {
            transition: transform 0.3s ease;
            height: 100%;
        }
        
        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        .recommendation-img {
            height: 180px;
            object-fit: cover;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        
        /* User profile styles */
        .profile-card {
            text-align: center;
            padding: 20px;
        }
        
        .profile-img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 15px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #121212;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #535353;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--spotify-green);
        }
    </style>
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-dark sticky-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fab fa-spotify me-2"></i>Spotify Analyzer
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard"><i class="fas fa-chart-line me-1"></i>Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/recommendations"><i class="fas fa-headphones me-1"></i>Recommendations</a>
                    </li>
                </ul>
                {% if user %}
                <div class="d-flex align-items-center">
                    {% if user %}
                    <div class="d-flex align-items-center">
                        <img src="{{ user.images[0].url if user.images and user.images|length > 0 else '/static/images/profile/bbg.jpg' }}" 
                             alt="Profile" 
                             class="rounded-circle me-2" 
                             style="width: 30px; height: 30px;"
                             onerror="this.src='/static/images/profile/bbg.jpg'">
                    </div>
                    {% endif %}
                    <span class="me-3">{{ user.display_name }}</span>
                </div>
                {% endif %}
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container py-4">
        {% block content %}{% endblock %}
    </div>

    <!-- Footer -->
    <footer class="text-center py-4">
        <div class="container">
            <p>Spotify Listening Pattern Analyzer - Created with Python, Pandas, and Chart.js</p>
            <p><small>This application is not affiliated with Spotify. All Spotify data belongs to their respective owners.</small></p>
        </div>
    </footer>

    <!-- Bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
        """
        
        # 2. Index template
        index_html = """
{% extends "base.html" %}

{% block title %}Spotify Listening Pattern Analyzer - Home{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-10">
        <div class="text-center mb-5">
            <h1 class="display-4 mb-3"><i class="fab fa-spotify me-3"></i>Spotify Listening Pattern Analyzer</h1>
            <p class="lead">Discover insights about your music taste and listening habits</p>
        </div>

        {% if user %}
        <div class="card profile-card mb-5">
            <div class="card-body">
                <!-- In index.html, update the profile image element -->
                <img src="{{ user.images[0].url if user.images and user.images|length > 0 else '/static/images/profile/bbg.jpg' }}" 
                     alt="Profile" 
                     class="profile-img"
                     onerror="this.src='/static/images/profile/bbg.jpg'">
                <h2 class="mb-2">{{ user.display_name }}</h2>
                <p class="text-muted">{{ user.email }}</p>
                <div class="mt-4">
                    <a href="/dashboard" class="btn btn-spotify me-2">
                        <i class="fas fa-chart-line me-2"></i>View Your Dashboard
                    </a>
                    <a href="/recommendations" class="btn btn-outline-light">
                        <i class="fas fa-headphones me-2"></i>Get Recommendations
                    </a>
                </div>
            </div>
        </div>
        {% else %}
        <div class="card mb-5">
            <div class="card-body text-center">
                <h2 class="mb-3">Connect Your Spotify Account</h2>
                <p>Sign in with your Spotify account to analyze your listening habits and get personalized music recommendations.</p>
                <a href="/login" class="btn btn-spotify btn-lg mt-3">
                    <i class="fab fa-spotify me-2"></i>Connect with Spotify
                </a>
            </div>
        </div>
        {% endif %}

        <div class="row">
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-chart-pie fa-3x mb-3" style="color: var(--spotify-green);"></i>
                        <h4>Analyze Listening Patterns</h4>
                        <p>Discover when and how you listen to music, your favorite genres, and the audio characteristics you prefer.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-network-wired fa-3x mb-3" style="color: var(--spotify-green);"></i>
                        <h4>Visualize Connections</h4>
                        <p>See how your favorite artists and genres are connected through interactive network visualizations.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-body text-center">
                        <i class="fas fa-headphones-alt fa-3x mb-3" style="color: var(--spotify-green);"></i>
                        <h4>Get Recommendations</h4>
                        <p>Discover new music based on your listening patterns, preferred genres, and audio features.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        """
        
        # 3. Dashboard template
        dashboard_html = """
{% extends "base.html" %}

{% block title %}Spotify Listening Pattern Analyzer - Dashboard{% endblock %}

{% block content %}
<h1 class="mb-4">Your Listening Dashboard</h1>

<!-- Listening Patterns Section -->
<h2 class="mb-3">Listening Patterns</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">When You Listen</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="listeningByHourChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Days You Listen Most</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="listeningByDayChart"></canvas>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Genre and Mood Section -->
<h2 class="mb-3 mt-5">Genres & Mood</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Top Genres</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="genreDistributionChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Mood Distribution</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="moodDistributionChart"></canvas>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Audio Features Section -->
<h2 class="mb-3 mt-5">Audio Features</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Your Audio Profile</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="audioFeaturesRadarChart"></canvas>
                </div>
            </div>
            <div class="card-footer">
                <small class="text-muted">
                    <strong>Danceability:</strong> How suitable a track is for dancing<br>
                    <strong>Energy:</strong> Intensity and activity level<br>
                    <strong>Valence:</strong> Musical positiveness (happy, cheerful)<br>
                    <strong>Acousticness:</strong> Whether the track is acoustic<br>
                    <strong>Instrumentalness:</strong> Whether a track contains vocals<br>
                    <strong>Speechiness:</strong> Presence of spoken words<br>
                    <strong>Liveness:</strong> Presence of an audience
                </small>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Track Clustering</div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="trackClusteringChart"></canvas>
                </div>
            </div>
            <div class="card-footer">
                <small class="text-muted">
                    Tracks clustered by audio features using machine learning.
                    Tracks that sound similar are grouped together.
                </small>
            </div>
        </div>
    </div>
</div>

<!-- Network Visualizations Section -->
<h2 class="mb-3 mt-5">Networks</h2>
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Artist Connections</div>
            <div class="card-body">
                <div class="network-container" id="artistNetworkViz"></div>
            </div>
            <div class="card-footer">
                <small class="text-muted">
                    Network of connections between your top artists.
                    Artists are connected if they share genres.
                </small>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">Genre Map</div>
            <div class="card-body">
                <div class="network-container" id="genreNetworkViz"></div>
            </div>
            <div class="card-footer">
                <small class="text-muted">
                    Network of connections between your top genres.
                    Genres are connected if they appear together for the same artists.
                </small>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
    // Initialize charts when the page loads
    document.addEventListener('DOMContentLoaded', function() {
        // Chart.js configuration
        const chartConfigs = {{ chart_configs|tojson }};
        const networkConfigs = {{ network_configs|tojson }};
        
        // Initialize Chart.js charts
        for (const [chartId, config] of Object.entries(chartConfigs)) {
            const ctx = document.getElementById(chartId);
            if (ctx) {
                new Chart(ctx, config);
            }
        }
        
        // Initialize Track Clustering Scatter Plot
        if (chartConfigs.hasOwnProperty('trackClusteringData')) {
            const clusteringData = chartConfigs.trackClusteringData;
            const clusterCtx = document.getElementById('trackClusteringChart');
            
            if (clusterCtx) {
                new Chart(clusterCtx, {
                    type: 'scatter',
                    data: {
                        datasets: clusteringData.map((cluster, index) => ({
                            label: `Cluster ${index + 1}`,
                            data: cluster.tracks,
                            backgroundColor: cluster.color,
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }))
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Track Clustering by Audio Features'
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const track = context.raw;
                                        return `${track.name} - ${track.artist}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Component 1'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Component 2'
                                }
                            }
                        }
                    }
                });
            }
        }
        
        // Initialize D3.js network visualizations
        initializeArtistNetwork();
        initializeGenreNetwork();
        
        function initializeArtistNetwork() {
            if (!networkConfigs.hasOwnProperty('artistNetwork')) return;
            
            const artistNetwork = networkConfigs.artistNetwork;
            const container = document.getElementById('artistNetworkViz');
            
            if (!container) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // Create force simulation
            const simulation = d3.forceSimulation(artistNetwork.nodes)
                .force('link', d3.forceLink(artistNetwork.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));
            
            // Create links
            const link = svg.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(artistNetwork.links)
                .enter()
                .append('line')
                .attr('stroke', '#555')
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.sqrt(d.weight));
            
            // Create nodes
            const node = svg.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(artistNetwork.nodes)
                .enter()
                .append('g');
            
            // Add circles to nodes
            node.append('circle')
                .attr('r', d => 5 + Math.sqrt(d.popularity) / 2)
                .attr('fill', '#1DB954')
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5);
            
            // Add labels to nodes
            node.append('text')
                .text(d => d.name)
                .attr('x', 10)
                .attr('y', 3)
                .attr('fill', '#fff')
                .style('font-size', '10px');
            
            // Add tooltips
            node.append('title')
                .text(d => `${d.name}\nPopularity: ${d.popularity}\nGenres: ${d.genres.join(', ')}`);
            
            // Update positions on simulation tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('transform', d => `translate(${d.x},${d.y})`);
            });
            
            // Add drag behavior
            node.call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        function initializeGenreNetwork() {
            if (!networkConfigs.hasOwnProperty('genreNetwork')) return;
            
            const genreNetwork = networkConfigs.genreNetwork;
            const container = document.getElementById('genreNetworkViz');
            
            if (!container) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // Create force simulation
            const simulation = d3.forceSimulation(genreNetwork.nodes)
                .force('link', d3.forceLink(genreNetwork.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-200))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));
            
            // Create links
            const link = svg.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(genreNetwork.links)
                .enter()
                .append('line')
                .attr('stroke', '#555')
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.sqrt(d.weight));
            
            // Create a color scale for genres
            const color = d3.scaleOrdinal(d3.schemeCategory10);
            
            // Create nodes
            const node = svg.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(genreNetwork.nodes)
                .enter()
                .append('g');
            
            // Add circles to nodes
            node.append('circle')
                .attr('r', d => 5 + Math.sqrt(d.count))
                .attr('fill', (_, i) => color(i % 10))
                .attr('stroke', '#fff')
                .attr('stroke-width', 1.5);
            
            // Add labels to nodes
            node.append('text')
                .text(d => d.id)
                .attr('x', 10)
                .attr('y', 3)
                .attr('fill', '#fff')
                .style('font-size', '10px');
            
            // Add tooltips
            node.append('title')
                .text(d => `${d.id}\nCount: ${d.count}`);
            
            // Update positions on simulation tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('transform', d => `translate(${d.x},${d.y})`);
            });
            
            // Add drag behavior
            node.call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
    });
</script>
{% endblock %}
        """
        
        # 4. Recommendations template
        recommendations_html = """
{% extends "base.html" %}

{% block title %}Spotify Listening Pattern Analyzer - Recommendations{% endblock %}

{% block content %}
<h1 class="mb-4">Your Music Recommendations</h1>

<!-- Recommendations based on top tracks -->
<section class="mb-5">
    <h2 class="mb-3">Based on Your Top Tracks</h2>
    <div class="row">
        {% for rec in recommendations.top_tracks_medium_term[:8] %}
        <div class="col-md-3 col-sm-6 mb-4">
            <div class="card recommendation-card">
                <!-- In recommendations.html for album covers -->
                <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/albums/blinding.png' }}" 
                     class="recommendation-img" 
                     alt="{{ rec.name }}"
                     onerror="this.src='/static/images/albums/blinding.png'">
                <div class="card-body">
                    <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                    <p class="card-text text-truncate">{{ rec.artist }}</p>
                    <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                        <i class="fab fa-spotify me-1"></i> Listen
                    </a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</section>

<!-- Recommendations based on mood -->
<section class="mb-5">
    <h2 class="mb-3">Based on Mood</h2>
    
    <!-- Mood Tabs -->
    <ul class="nav nav-tabs mb-3" id="moodTabs" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="happy-tab" data-bs-toggle="tab" data-bs-target="#happy" type="button" role="tab">
                Happy/Euphoric
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="calm-tab" data-bs-toggle="tab" data-bs-target="#calm" type="button" role="tab">
                Calm/Peaceful
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="angry-tab" data-bs-toggle="tab" data-bs-target="#angry" type="button" role="tab">
                Angry/Intense
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="sad-tab" data-bs-toggle="tab" data-bs-target="#sad" type="button" role="tab">
                Sad/Depressed
            </button>
        </li>
    </ul>
    
    <!-- Mood Tab Contents -->
    <div class="tab-content" id="moodTabContent">
        <!-- Happy/Euphoric -->
        <div class="tab-pane fade show active" id="happy" role="tabpanel" aria-labelledby="happy-tab">
            <div class="row">
                {% for rec in recommendations.mood_Happy_Euphoric[:8] %}
                <div class="col-md-3 col-sm-6 mb-4">
                    <div class="card recommendation-card">
                        <!-- For Happy/Euphoric section -->
                        <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/moods/happy.png' }}" 
                             class="recommendation-img" 
                             alt="{{ rec.name }}"
                             onerror="this.src='/static/images/moods/happy.png'">    
                        <div class="card-body">
                            <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                            <p class="card-text text-truncate">{{ rec.artist }}</p>
                            <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                                <i class="fab fa-spotify me-1"></i> Listen
                            </a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Calm/Peaceful -->
        <div class="tab-pane fade" id="calm" role="tabpanel" aria-labelledby="calm-tab">
            <div class="row">
                {% for rec in recommendations.mood_Calm_Peaceful[:8] %}
                <div class="col-md-3 col-sm-6 mb-4">
                    <div class="card recommendation-card">
                        <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/moods/calm.png' }}" 
                             class="recommendation-img" 
                             alt="{{ rec.name }}"
                             onerror="this.src='/static/images/moods/calm.png'"> 
                        <div class="card-body">
                            <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                            <p class="card-text text-truncate">{{ rec.artist }}</p>
                            <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                                <i class="fab fa-spotify me-1"></i> Listen
                            </a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Angry/Intense -->
        <div class="tab-pane fade" id="angry" role="tabpanel" aria-labelledby="angry-tab">
            <div class="row">
                {% for rec in recommendations.mood_Angry_Intense[:8] %}
                <div class="col-md-3 col-sm-6 mb-4">
                    <div class="card recommendation-card">
                        <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/moods/angry.png' }}" 
                             class="recommendation-img" 
                             alt="{{ rec.name }}"
                             onerror="this.src='/static/images/moods/angry.png'"> 
                        <div class="card-body">
                            <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                            <p class="card-text text-truncate">{{ rec.artist }}</p>
                            <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                                <i class="fab fa-spotify me-1"></i> Listen
                            </a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Sad/Depressed -->
        <div class="tab-pane fade" id="sad" role="tabpanel" aria-labelledby="sad-tab">
            <div class="row">
                {% for rec in recommendations.mood_Sad_Depressed[:8] %}
                <div class="col-md-3 col-sm-6 mb-4">
                    <div class="card recommendation-card">
                        <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/moods/sad.png' }}" 
                             class="recommendation-img" 
                             alt="{{ rec.name }}"
                             onerror="this.src='/static/images/moods/sad.png'"> 
                        <div class="card-body">
                            <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                            <p class="card-text text-truncate">{{ rec.artist }}</p>
                            <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                                <i class="fab fa-spotify me-1"></i> Listen
                            </a>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
</section>

<!-- Recommendations based on genres -->
<section class="mb-5">
    <h2 class="mb-3">Based on Your Favorite Genres</h2>
    <div class="row">
        {% for rec in recommendations.genres[:8] %}
        <div class="col-md-3 col-sm-6 mb-4">
            <div class="card recommendation-card">
                <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/genres/default.png' }}" 
                     class="recommendation-img" 
                     alt="{{ rec.name }}"
                     onerror="this.src='/static/images/genres/default.png'">
                <div class="card-body">
                    <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                    <p class="card-text text-truncate">{{ rec.artist }}</p>
                    <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                        <i class="fab fa-spotify me-1"></i> Listen
                    </a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</section>

<!-- Recommendations based on audio features -->
<section class="mb-5">
    <h2 class="mb-3">Based on Your Audio Profile</h2>
    <div class="row">
        {% for rec in recommendations.audio_features[:8] %}
        <div class="col-md-3 col-sm-6 mb-4">
            <div class="card recommendation-card">
                <img src="{{ rec.album_cover_url if rec.album_cover_url else '/static/images/features/default.png' }}" 
                     class="recommendation-img" 
                     alt="{{ rec.name }}"
                     onerror="this.src='/static/images/features/default.png'">
                <div class="card-body">
                    <h5 class="card-title text-truncate">{{ rec.name }}</h5>
                    <p class="card-text text-truncate">{{ rec.artist }}</p>
                    <a href="https://open.spotify.com/track/{{ rec.id }}" target="_blank" class="btn btn-sm btn-spotify w-100">
                        <i class="fab fa-spotify me-1"></i> Listen
                    </a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</section>
{% endblock %}

{% block extra_js %}
<script>
    // Enable Bootstrap tabs
    document.addEventListener('DOMContentLoaded', function() {
        const tabElements = document.querySelectorAll('#moodTabs .nav-link');
        // Enable Bootstrap tabs
        tabElements.forEach(function(tabEl) {
            const tabTrigger = new bootstrap.Tab(tabEl);

            tabEl.addEventListener('click', function(event) {
                event.preventDefault();
                
                tabElements.forEach(tab => {
                    tab.classList.remove('active');
                });

                this.classList.add('active');

                tabTrigger.show();
            });
        });

        const activeTab = document.querySelector('#happy-tab');
        if (activeTab) {
            activeTab.classList.add('active');
        }

        const triggerTabList = [].slice.call(document.querySelectorAll('#moodTabs button'));
        triggerTabList.forEach(function(triggerEl) {
            const tabTrigger = new bootstrap.Tab(triggerEl);
            triggerEl.addEventListener('click', function(event) {
                event.preventDefault();
                tabTrigger.show();
            });
        });
    });
</script>
{% endblock %}
        """
        
        # Write templates to files
        with open('templates/base.html', 'w') as f:
            f.write(base_html)
        
        with open('templates/index.html', 'w') as f:
            f.write(index_html)
        
        with open('templates/dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        with open('templates/recommendations.html', 'w') as f:
            f.write(recommendations_html)
        
        print("Templates created successfully")
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)

#####################
# 7. MAIN WORKFLOW #
#####################

def main():
    """Main function to run the Spotify Listening Pattern Analyzer"""
    print("Starting Spotify Listening Pattern Analyzer...")

    use_sample_data = True  # Set to False to use real Spotify data
    
    if use_sample_data:
        # Check if sample data file exists, if not generate it
        sample_file = 'sample_spotify_data.json'
        if not os.path.exists(sample_file):
            print("Generating sample Spotify data...")
            from sample_data import save_sample_data
            user_data = save_sample_data(sample_file)
        else:
            print("Loading sample Spotify data...")
            with open(sample_file, 'r') as f:
                user_data = json.load(f)
        
        # Create a dummy Spotify client for recommendations
        sp = None  # We won't use the API with sample data
    else:
        
        # Step 1: Setup Spotify client
        sp = setup_spotify_client()
        if not sp:
            print("Failed to setup Spotify client. Please check your credentials.")
            return
    
        # Step 2: Collect data
        collector = SpotifyDataCollector(sp)
    
        # Check if data file exists
        if os.path.exists('spotify_data.json'):
            print("Found existing data file. Loading data...")
            user_data = collector.load_data_from_file()
            
        else:
            print("Collecting data from Spotify API...")
            user_data = collector.collect_all_data()
            collector.save_data_to_file()
    
    # Step 3: Analyze data
    analyzer = SpotifyDataAnalyzer(user_data)
    
    # Check if analysis file exists
    analysis_file = 'sample_analysis.json' if use_sample_data else 'spotify_analysis.json'
    if os.path.exists(analysis_file):
        print(f"Found existing analysis file. Loading analysis...")
        analysis_results = analyzer.load_analysis_from_file(analysis_file)
        if analysis_results is None:
            print("Failed to load analysis, generating new analysis...")
            analysis_results = analyzer.run_all_analyses()
            analyzer.save_analysis_to_file(analysis_file)
    else:
        print("Analyzing Spotify data...")
        analysis_results = analyzer.run_all_analyses()
        analyzer.save_analysis_to_file(analysis_file)
    
    # Step 4: Generate recommendations
    if use_sample_data:
        print("Generating mock recommendations...")
        # Create a simple mock recommendations structure
        recommendations = {}
        # Add recommendations for different categories
        for category in ['top_tracks_short_term', 'top_tracks_medium_term', 'top_tracks_long_term',
                        'top_artists_short_term', 'top_artists_medium_term', 'top_artists_long_term',
                        'genres', 'audio_features']:
            # Pick 10 random tracks as recommendations
            recommendations[category] = random.sample(user_data['saved_tracks'], min(10, len(user_data['saved_tracks'])))
        
        # Add mood-based recommendations
        recommendations['mood_Happy_Euphoric'] = random.sample(user_data['saved_tracks'], min(10, len(user_data['saved_tracks'])))
        recommendations['mood_Calm_Peaceful'] = random.sample(user_data['saved_tracks'], min(10, len(user_data['saved_tracks'])))
        recommendations['mood_Angry_Intense'] = random.sample(user_data['saved_tracks'], min(10, len(user_data['saved_tracks'])))
        recommendations['mood_Sad_Depressed'] = random.sample(user_data['saved_tracks'], min(10, len(user_data['saved_tracks'])))
        
        # Save mock recommendations
        with open('sample_recommendations.json', 'w') as f:
            json.dump(recommendations, f)
    else:
        # Use real recommendations from Spotify API
        print("Generating music recommendations...")
        recommender = SpotifyRecommender(sp, user_data, analysis_results)
        recommendations = recommender.get_all_recommendations()
        recommender.save_recommendations_to_file(recommendations)
        recommender.save_recommendations_to_file(recommendations)
    
    # Step 5: Setup and run web application
    print("Setting up web application...")
    web_app = SpotifyWebApp(user_data, analysis_results, recommendations)
    web_app.create_templates()
    
    print("Starting web server...")
    web_app.run(port=5051)

if __name__ == "__main__":
    main()