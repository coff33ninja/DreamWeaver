import random
import requests
import os

class ChaosEngine:
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY") # Use environment variable
        if not self.nasa_api_key:
            print("Warning: NASA API key not set, using DEMO_KEY. Limited requests allowed.")

    def random_factor(self):
        return random.random() * 10

    def apply_chaos(self, narration, character_texts):
        if random.random() < 0.1:
            try:
                response = requests.get(f"https://api.nasa.gov/neo/rest/v1/feed?api_key={self.nasa_api_key}", timeout=5)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                event = response.json().get("near_earth_objects", {}).get("today", [{"name": "Asteroid incoming!"}])[0].get("name", "Unknown cosmic event!")
                narration += f"\nChaos: {event}"
            except requests.exceptions.RequestException as e:
                print(f"Error fetching NASA data for chaos event: {e}")
                narration += "\nChaos: A mysterious disturbance ripples through the fabric of reality!"
            for character in character_texts:
                character_texts[character] = f"Chaos strikes: {character_texts[character]}"
        return narration, character_texts
