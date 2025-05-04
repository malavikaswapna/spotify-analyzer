import random
import json
import datetime
from collections import Counter

def generate_sample_data():
    """Generate sample Spotify data for testing"""
    
    # Sample user profile
    profile = {
        "display_name": "Sample User",
        "email": "user@example.com",
        "id": "sampleuser123",
        "images": [{"url": "static/images/profile/bbg.jpg"}],
        "followers": {"total": 42}
    }
    
    # Sample artists with realistic data
    artists = [
        {"id": "0TnOYISbd1XYRBk9myaseg", "name": "Taylor Swift", "genres": ["pop", "country pop"], "popularity": 98, "followers": {"total": 45000000}},
        {"id": "1Xyo4u8uXC1ZmMpatF05PJ", "name": "The Weeknd", "genres": ["canadian pop", "r&b"], "popularity": 97, "followers": {"total": 35000000}},
        {"id": "3TVXtAsR1Inumwj472S9r4", "name": "Drake", "genres": ["canadian hip hop", "rap"], "popularity": 96, "followers": {"total": 37000000}},
        {"id": "06HL4z0CvFAxyc27GXpf02", "name": "BTS", "genres": ["k-pop", "pop"], "popularity": 92, "followers": {"total": 30000000}},
        {"id": "1vCWHaC5f2uS3yhpwWbIA6", "name": "Kendrick Lamar", "genres": ["conscious hip hop", "hip hop", "rap"], "popularity": 91, "followers": {"total": 12000000}},
        {"id": "7jVv8c5Fj3E9VhNjxT4snq", "name": "Lil Nas X", "genres": ["pop rap", "pop"], "popularity": 89, "followers": {"total": 8500000}},
        {"id": "3WrFJ7ztbogyGnTHbHJFl2", "name": "Billie Eilish", "genres": ["electropop", "pop"], "popularity": 88, "followers": {"total": 10000000}},
        {"id": "6eUKZXaKkcviH0Ku9w2n3V", "name": "Ed Sheeran", "genres": ["pop", "uk pop"], "popularity": 90, "followers": {"total": 28000000}},
        {"id": "4gzpq5DPGxSnKTe4SA8HAU", "name": "Coldplay", "genres": ["permanent wave", "pop"], "popularity": 86, "followers": {"total": 25000000}},
        {"id": "246dkjvS1zLTtiykXe5h60", "name": "Post Malone", "genres": ["dfw rap", "pop", "rap"], "popularity": 87, "followers": {"total": 15000000}},
        {"id": "6KImCVD70vtIoJWnq6nGn3", "name": "Harry Styles", "genres": ["pop", "uk pop"], "popularity": 85, "followers": {"total": 12000000}},
        {"id": "4YRxDV8wJFPHPTeXepOstw", "name": "Olivia Rodrigo", "genres": ["pop"], "popularity": 84, "followers": {"total": 7500000}},
        {"id": "1uNFoZAHBGtllmzznpCI3s", "name": "Justin Bieber", "genres": ["canadian pop", "pop"], "popularity": 83, "followers": {"total": 30000000}},
        {"id": "5pKCCKE2ajJHZ9KAiaK11H", "name": "Rihanna", "genres": ["barbadian pop", "pop", "r&b"], "popularity": 89, "followers": {"total": 32000000}},
        {"id": "0du5cEVh5yTK9QJze8zA0C", "name": "Bruno Mars", "genres": ["pop"], "popularity": 82, "followers": {"total": 22000000}},
    ]
    
    # Sample tracks with realistic data
    tracks = [
        {"id": "4iJyoBOLtHqaGxP12qzhQI", "name": "Anti-Hero", "artist": "Taylor Swift", "artist_id": "0TnOYISbd1XYRBk9myaseg", "album": "Midnights", "album_id": "151w1FgRZfnKZA9FEcg9Z3", "popularity": 95, "duration_ms": 200672, "album_cover_url": "/static/images/albums/anti-hero.png"},
        {"id": "1bDbXMyjaUIooNwFE9wn0N", "name": "Blinding Lights", "artist": "The Weeknd", "artist_id": "1Xyo4u8uXC1ZmMpatF05PJ", "album": "After Hours", "album_id": "4yP0hdKOZPNshxUOjY0cZj", "popularity": 94, "duration_ms": 200040, "album_cover_url": "/static/images/albums/blinding.png"},
        {"id": "5QO79kh1waicV47BqGRL3g", "name": "Save Your Tears", "artist": "The Weeknd", "artist_id": "1Xyo4u8uXC1ZmMpatF05PJ", "album": "After Hours", "album_id": "4yP0hdKOZPNshxUOjY0cZj", "popularity": 93, "duration_ms": 215627, "album_cover_url": "/static/images/albums/savetears.jpg"},
        {"id": "4QLAtpLNUsHEYrcHXmMIZZ", "name": "STAY", "artist": "Justin Bieber", "artist_id": "1uNFoZAHBGtllmzznpCI3s", "album": "STAY", "album_id": "6M3FYnL1FQvnYyNJfm9vvw", "popularity": 90, "duration_ms": 141805, "album_cover_url": "/static/images/albums/stay.png"},
        {"id": "5wANPM4fQCJwkGd4rN57mH", "name": "drivers license", "artist": "Olivia Rodrigo", "artist_id": "4YRxDV8wJFPHPTeXepOstw", "album": "SOUR", "album_id": "6s84u2TUpR3wdUv4NgKA2j", "popularity": 89, "duration_ms": 242013, "album_cover_url": "/static/images/albums/driver.png"},
        {"id": "4ZtFanR9U6ndgddUvNcjcG", "name": "good 4 u", "artist": "Olivia Rodrigo", "artist_id": "4YRxDV8wJFPHPTeXepOstw", "album": "SOUR", "album_id": "6s84u2TUpR3wdUv4NgKA2j", "popularity": 88, "duration_ms": 178147, "album_cover_url": "/static/images/albums/good.png"},
        {"id": "5PjdY0CKGZdEuoNab3yDmX", "name": "INDUSTRY BABY", "artist": "Lil Nas X", "artist_id": "7jVv8c5Fj3E9VhNjxT4snq", "album": "MONTERO", "album_id": "6pOiDiuDQqrmo5DbG0ZubR", "popularity": 86, "duration_ms": 212000, "album_cover_url": "/static/images/albums/call.jpg"},
        {"id": "02MWAaffLxlfxAUY7c5dvx", "name": "Heat Waves", "artist": "Glass Animals", "artist_id": "4yvcSjfu4PC0CYQyLy4wSq", "album": "Dreamland", "album_id": "5bfgWVCnIs1XKZ6aJzLaPf", "popularity": 87, "duration_ms": 238805, "album_cover_url": "/static/images/albums/heat.png"},
        {"id": "7qiZfU4dY1lWllzX7mPBI3", "name": "Shape of You", "artist": "Ed Sheeran", "artist_id": "6eUKZXaKkcviH0Ku9w2n3V", "album": "÷", "album_id": "3T4tUhGYeRNVUGevb0wThu", "popularity": 83, "duration_ms": 233713, "album_cover_url": "/static/images/albums/shape.png"},
        {"id": "5uCax9HTNlzGybIStD3vDh", "name": "Say So", "artist": "Doja Cat", "artist_id": "5cj0lLjcoR7YOSnhnX0Po5", "album": "Hot Pink", "album_id": "7aOE6PORwbKURcfQMvZT5e", "popularity": 85, "duration_ms": 237893, "album_cover_url": "/static/images/albums/say.png"},
        {"id": "0e4Anw4QE7GKtZwhRWbiSK", "name": "Yellow", "artist": "Coldplay", "artist_id": "4gzpq5DPGxSnKTe4SA8HAU", "album": "Parachutes", "album_id": "6ZG5lRT77aJ3btmArcykra", "popularity": 80, "duration_ms": 271240,"album_cover_url": "/static/images/albums/yellow.jpg"},
        {"id": "6DCZcSspjsKoFjzjrWoCdn", "name": "God's Plan", "artist": "Drake", "artist_id": "3TVXtAsR1Inumwj472S9r4", "album": "Scorpion", "album_id": "1ATL5GLyefJaxhQzSPVrLX", "popularity": 82, "duration_ms": 198885, "album_cover_url": "/static/images/albums/god.jpg"},
        {"id": "6ft4hAq6yde8jPZY2i5zLr", "name": "Creepin'", "artist": "The Weeknd", "artist_id": "1Xyo4u8uXC1ZmMpatF05PJ", "album": "Heroes & Villains", "album_id": "6AlH8K1aEtZLEervpW9mVG", "popularity": 83, "duration_ms": 220928, "album_cover_url": "/static/images/albums/creep.jpg"},
        {"id": "2LBqCSwhJGcFQeTHMVGwy3", "name": "Die For You", "artist": "The Weeknd", "artist_id": "1Xyo4u8uXC1ZmMpatF05PJ", "album": "Starboy", "album_id": "2ODvWsOgouMbaA5xf0RkJe", "popularity": 85, "duration_ms": 260253, "album_cover_url": "/static/images/albums/die.png"},
        {"id": "4Dvkj6JhhA12EX05fT7y2e", "name": "As It Was", "artist": "Harry Styles", "artist_id": "6KImCVD70vtIoJWnq6nGn3", "album": "Harry's House", "album_id": "5r36AJ6VOJtp00oxSkBZ5h", "popularity": 93, "duration_ms": 167303, "album_cover_url": "/static/images/albums/as.png"},
    ]
    
    # Generate audio features for tracks (realistic values)
    audio_features = []
    for track in tracks:
        feature = {
            "id": track["id"],
            "danceability": random.uniform(0.3, 0.9),
            "energy": random.uniform(0.3, 0.9),
            "key": random.randint(0, 11),
            "loudness": random.uniform(-12, -2),
            "mode": random.randint(0, 1),
            "speechiness": random.uniform(0.02, 0.2),
            "acousticness": random.uniform(0.01, 0.8),
            "instrumentalness": random.uniform(0, 0.5),
            "liveness": random.uniform(0.05, 0.5),
            "valence": random.uniform(0.1, 0.9),
            "tempo": random.uniform(70, 180),
            "type": "audio_features",
            "duration_ms": track["duration_ms"],
            "time_signature": 4
        }
        audio_features.append(feature)
    
    # Generate listening timestamps over the past month
    now = datetime.datetime.now()
    recently_played = []
    for _ in range(50):
        # Pick a random track
        track = random.choice(tracks)
        
        # Random timestamp within the last month
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        played_at = now - datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        played_at_str = played_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        recently_played.append({
            "track": track,
            "played_at": played_at_str
        })
    
    # Create top tracks for different time ranges
    top_tracks = {}
    for time_range in ["short_term", "medium_term", "long_term"]:
        # Shuffle tracks and pick a subset
        time_tracks = random.sample(tracks, min(len(tracks), 10))
        top_tracks[time_range] = [
            {"rank": i+1, **track} for i, track in enumerate(time_tracks)
        ]
    
    # Create top artists for different time ranges
    top_artists = {}
    for time_range in ["short_term", "medium_term", "long_term"]:
        # Shuffle artists and pick a subset
        time_artists = random.sample(artists, min(len(artists), 10))
        top_artists[time_range] = [
            {"rank": i+1, **artist} for i, artist in enumerate(time_artists)
        ]
    
    # Combine everything into user_data
    user_data = {
        "profile": profile,
        "audio_features": audio_features,
        "recently_played": recently_played,
        "saved_tracks": random.sample(tracks, 10)  # 10 random saved tracks
    }
    
    # Add top tracks and artists for each time range
    for time_range in ["short_term", "medium_term", "long_term"]:
        user_data[f"top_tracks_{time_range}"] = top_tracks[time_range]
        user_data[f"top_artists_{time_range}"] = top_artists[time_range]
    
    return user_data

def save_sample_data(filename="sample_spotify_data.json"):
    """Generate and save sample data to a file"""
    data = generate_sample_data()
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Sample data saved to {filename}")
    return data

if __name__ == "__main__":
    save_sample_data()