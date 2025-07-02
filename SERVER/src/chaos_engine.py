import random
import requests
import os
import logging

logger = logging.getLogger("dreamweaver_server")

class ChaosEngine:
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        if not self.nasa_api_key or self.nasa_api_key == "DEMO_KEY":
            logger.warning("ChaosEngine: NASA API key not set or using DEMO_KEY. Actual NASA data for chaos events will be limited or unavailable.")
        else:
            logger.info("ChaosEngine: NASA API key found. Will use it for fetching cosmic events.")
        self.event_triggered_this_turn = False # To ensure only one major chaos event per call if desired

    def random_factor(self):
        """Generates a random factor, typically between 0 and 10."""
        return random.random() * 10

    def apply_chaos(self, narration, character_texts):
        """
        Potentially modifies the narration and character texts by introducing a chaos event,
        such as a cosmic event fetched from NASA's API.
        """
        self.event_triggered_this_turn = False
        # Example: 10% chance to trigger a NASA event if API key is not DEMO_KEY
        if self.nasa_api_key != "DEMO_KEY" and random.random() < 0.1:
            try:
                logger.info("ChaosEngine: Attempting to fetch NASA NEO data for a chaos event...")
                # Fetch today's Near Earth Objects
                # For a real app, you might want to vary the date or query type
                response = requests.get(f"https://api.nasa.gov/neo/rest/v1/feed/today?detailed=false&api_key={self.nasa_api_key}", timeout=5)
                response.raise_for_status()
                data = response.json()

                near_earth_objects = data.get("near_earth_objects", {})
                if near_earth_objects:
                    # Get a list of all NEOs for today (dates are keys in near_earth_objects)
                    all_neos_today = []
                    for date_key in near_earth_objects.keys(): # Should only be one date for 'today' feed
                        all_neos_today.extend(near_earth_objects[date_key])

                    if all_neos_today:
                        selected_neo = random.choice(all_neos_today)
                        event_name = selected_neo.get("name", "Unnamed cosmic entity")
                        miss_distance_km = float(selected_neo.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("kilometers", "unknown"))
                        velocity_km_s = float(selected_neo.get("close_approach_data", [{}])[0].get("relative_velocity", {}).get("kilometers_per_second", "unknown"))

                        event_description = (f"A cosmic event unfolds! {event_name} makes a close approach, "
                                             f"missing Earth by approximately {miss_distance_km:,.0f} km, "
                                             f"traveling at {velocity_km_s:,.1f} km/s.")
                        narration += f"\n[CHAOS EVENT] {event_description}"
                        logger.info(f"ChaosEngine: NASA NEO event triggered: {event_description}")
                        self.event_triggered_this_turn = True
                    else:
                        logger.info("ChaosEngine: No Near Earth Objects reported by NASA for today.")
                        narration += "\n[CHAOS EVENT] The cosmos is eerily quiet today..."
                else:
                    logger.info("ChaosEngine: NASA NEO data feed was empty or in an unexpected format.")
                    narration += "\n[CHAOS EVENT] The stars themselves seem to hold their breath..."


            except requests.exceptions.Timeout:
                logger.warning("ChaosEngine: Timeout fetching NASA data for chaos event.")
                narration += "\n[CHAOS EVENT] A distortion in spacetime makes external data feeds temporarily unavailable!"
            except requests.exceptions.RequestException as e:
                logger.error(f"ChaosEngine: Error fetching NASA data for chaos event: {e}", exc_info=True)
                narration += "\n[CHAOS EVENT] A mysterious disturbance ripples through the fabric of reality, disrupting sensor arrays!"
            except json.JSONDecodeError as e_json:
                logger.error(f"ChaosEngine: Error decoding NASA API response: {e_json}", exc_info=True)
                narration += "\n[CHAOS EVENT] Cosmic rays garble incoming transmissions!"

        # Generic chaos effect on character texts if no major event or as an additional layer
        if not self.event_triggered_this_turn and random.random() < 0.05 : # 5% chance for minor text chaos
            logger.info("ChaosEngine: Applying minor textual chaos to character responses.")
            for character in list(character_texts.keys()): # Iterate over a copy of keys if modifying dict
                original_text = character_texts[character]
                if random.random() < 0.5:
                    character_texts[character] = "".join(random.sample(original_text, len(original_text))) # Scramble text
                else:
                    character_texts[character] = original_text.upper() + "!!" # SHOUT
                logger.debug(f"ChaosEngine: Text for {character} changed from '{original_text[:30]}...' to '{character_texts[character][:30]}...'")

        return narration, character_texts

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing Chaos Engine...")
    ce = ChaosEngine()

    # Test with NASA API Key (if available)
    if ce.nasa_api_key != "DEMO_KEY":
        logger.info("Testing with actual NASA API Key.")
    else:
        logger.info("Testing with DEMO_KEY for NASA API (expect limited or no actual events).")

    test_narration = "The adventurers entered the dark cave."
    test_char_texts = {"Alice": "I hope we find the treasure.", "Bob": "Watch out for traps!"}

    for i in range(5): # Run a few times to see different outcomes
        logger.info(f"\n--- Chaos Test Iteration {i+1} ---")
        updated_narration, updated_char_texts = ce.apply_chaos(test_narration, test_char_texts.copy())
        logger.info(f"Updated Narration: {updated_narration}")
        logger.info(f"Updated Character Texts: {updated_char_texts}")

    logger.info("Chaos Engine test complete.")
