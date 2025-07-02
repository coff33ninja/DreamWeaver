import random
import requests
import os
import json
import logging

logger = logging.getLogger("dreamweaver_server")


class ChaosEngine:
    def __init__(self):
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        if not self.nasa_api_key or self.nasa_api_key == "DEMO_KEY":
            logger.warning(
                "ChaosEngine: NASA API key not set or using DEMO_KEY. Actual NASA data for chaos events will be limited or unavailable."
            )
        else:
            logger.info(
                "ChaosEngine: NASA API key found. Will use it for fetching cosmic events."
            )
        self.event_triggered_this_turn = (
            False  # To ensure only one major chaos event per call if desired
        )

    def random_factor(self):
        """Generates a random factor, typically between 0 and 10."""
        return random.random() * 10

    def _parse_neo_data(self, neo_data: dict) -> str:
        """Helper to parse a single NEO record and return a descriptive string."""
        event_name = neo_data.get("name", "Unnamed cosmic entity")
        miss_distance_km = float(
            neo_data.get("close_approach_data", [{}])[0]
            .get("miss_distance", {})
            .get("kilometers", "unknown")
        )
        velocity_km_s = float(
            neo_data.get("close_approach_data", [{}])[0]
            .get("relative_velocity", {})
            .get("kilometers_per_second", "unknown")
        )

        return (
            f"A cosmic event unfolds! {event_name} makes a close approach, "
            f"missing Earth by approximately {miss_distance_km:,.0f} km, "
            f"traveling at {velocity_km_s:,.1f} km/s."
        )

    def _get_nasa_neo_event(self) -> str | None:
        """
        Fetches and formats a NASA Near-Earth Object event.
        Returns a descriptive string for the event, a fallback message, or None on failure.
        """
        try:
            logger.info(
                "ChaosEngine: Attempting to fetch NASA NEO data for a chaos event..."
            )
            # Fetch today's Near Earth Objects
            response = requests.get(
                f"https://api.nasa.gov/neo/rest/v1/feed/today?detailed=false&api_key={self.nasa_api_key}",
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            if near_earth_objects := data.get("near_earth_objects", {}):
                if all_neos_today := [
                    neo
                    for date_key in near_earth_objects
                    for neo in near_earth_objects[date_key]
                ]:
                    return self._parse_neo_data(random.choice(all_neos_today))
                logger.info(
                    "ChaosEngine: No Near Earth Objects reported by NASA for today."
                )
                return "The cosmos is eerily quiet today..."
            else:
                logger.info(
                    "ChaosEngine: NASA NEO data feed was empty or in an unexpected format."
                )
                return "The stars themselves seem to hold their breath..."

        except requests.exceptions.Timeout:
            logger.warning("ChaosEngine: Timeout fetching NASA data for chaos event.")
            return "A distortion in spacetime makes external data feeds temporarily unavailable!"
        except requests.exceptions.RequestException as e:
            logger.error(
                f"ChaosEngine: Error fetching NASA data for chaos event: {e}",
                exc_info=True,
            )
            return "A mysterious disturbance ripples through the fabric of reality, disrupting sensor arrays!"
        except json.JSONDecodeError as e_json:
            logger.error(
                f"ChaosEngine: Error decoding NASA API response: {e_json}",
                exc_info=True,
            )
            return "Cosmic rays garble incoming transmissions!"

    def _apply_textual_chaos(self, character_texts: dict) -> dict:
        """Applies minor textual chaos to character responses."""
        logger.info("ChaosEngine: Applying minor textual chaos to character responses.")
        for character in list(
            character_texts.keys()
        ):  # Iterate over a copy of keys if modifying dict
            original_text = character_texts[character]
            if random.random() < 0.5:
                character_texts[character] = "".join(
                    random.sample(original_text, len(original_text))
                )  # Scramble text
            else:
                character_texts[character] = f"{original_text.upper()}!!"
            logger.debug(
                f"ChaosEngine: Text for {character} changed from '{original_text[:30]}...' to '{character_texts[character][:30]}...'"
            )
        return character_texts

    def apply_chaos(self, narration, character_texts):
        """
        Potentially modifies the narration and character texts by introducing a chaos event,
        such as a cosmic event fetched from NASA's API.
        """
        # Example: 10% chance to trigger a NASA event if API key is not DEMO_KEY
        if self.nasa_api_key != "DEMO_KEY" and random.random() < 0.1:
            if event_description := self._get_nasa_neo_event():
                narration += f"\n[CHAOS EVENT] {event_description}"
                logger.info(
                    f"ChaosEngine: Major chaos event triggered: {event_description}"
                )
                return (
                    narration,
                    character_texts,
                )  # Return early if a major event happened

        # Generic chaos effect on character texts if no major event or as an additional layer
        if random.random() < 0.05:  # 5% chance for minor text chaos
            character_texts = self._apply_textual_chaos(character_texts)

        return narration, character_texts


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing Chaos Engine...")
    ce = ChaosEngine()

    # Test with NASA API Key (if available)
    if ce.nasa_api_key != "DEMO_KEY":
        logger.info("Testing with actual NASA API Key.")
    else:
        logger.info(
            "Testing with DEMO_KEY for NASA API (expect limited or no actual events)."
        )

    test_narration = "The adventurers entered the dark cave."
    test_char_texts = {
        "Alice": "I hope we find the treasure.",
        "Bob": "Watch out for traps!",
    }

    for i in range(5):  # Run a few times to see different outcomes
        logger.info(f"\n--- Chaos Test Iteration {i+1} ---")
        updated_narration, updated_char_texts = ce.apply_chaos(
            test_narration, test_char_texts.copy()
        )
        logger.info(f"Updated Narration: {updated_narration}")
        logger.info(f"Updated Character Texts: {updated_char_texts}")

    logger.info("Chaos Engine test complete.")
