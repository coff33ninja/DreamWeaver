import unittest
from unittest.mock import patch, MagicMock
from src.chaos_engine import ChaosEngine


class TestChaosEngine(unittest.TestCase):
    def setUp(self):
        self.chaos_engine = ChaosEngine()
        self.test_narration = "The adventurers entered the dark cave."
        self.test_char_texts = {
            "Alice": "I hope we find the treasure.",
            "Bob": "Watch out for traps!",
        }

    @patch("random.random")
    def test_no_major_event_demo_key(self, mock_random):
        """Test when NASA API key is DEMO_KEY and no major chaos event occurs."""
        self.chaos_engine.nasa_api_key = "DEMO_KEY"
        mock_random.side_effect = [0.2, 0.2]  # Ensure no chaos event is triggered
        updated_narration, updated_char_texts = self.chaos_engine.apply_chaos(
            self.test_narration, self.test_char_texts.copy()
        )
        self.assertEqual(updated_narration, self.test_narration)
        self.assertEqual(updated_char_texts, self.test_char_texts)

    @patch("random.random")
    @patch("requests.get")
    def test_major_event_nasa_api(self, mock_requests, mock_random):
        """Test when NASA API key is valid and a major chaos event is triggered."""
        self.chaos_engine.nasa_api_key = "VALID_API_KEY"
        mock_random.side_effect = [0.05]  # Trigger NASA event
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "near_earth_objects": {
                "2023-10-01": [
                    {
                        "name": "Asteroid X",
                        "close_approach_data": [
                            {
                                "miss_distance": {"kilometers": "50000"},
                                "relative_velocity": {"kilometers_per_second": "25"},
                            }
                        ],
                    }
                ]
            }
        }
        mock_requests.return_value = mock_response
        updated_narration, updated_char_texts = self.chaos_engine.apply_chaos(
            self.test_narration, self.test_char_texts.copy()
        )
        self.assertIn("[CHAOS EVENT]", updated_narration)
        self.assertIn("Asteroid X", updated_narration)

    @patch("random.random")
    def test_minor_text_chaos_scramble(self, mock_random):
        """Test minor chaos effect: scrambling character texts."""
        mock_random.side_effect = [
            0.2,
            0.04,
            0.4,
        ]  # Trigger minor chaos and scramble text
        updated_narration, updated_char_texts = self.chaos_engine.apply_chaos(
            self.test_narration, self.test_char_texts.copy()
        )
        self.assertNotEqual(updated_char_texts["Alice"], self.test_char_texts["Alice"])
        self.assertNotEqual(updated_char_texts["Bob"], self.test_char_texts["Bob"])

    @patch("random.random")
    def test_minor_text_chaos_shout(self, mock_random):
        """Test minor chaos effect: shouting character texts."""
        mock_random.side_effect = [0.2, 0.04, 0.6]  # Trigger minor chaos and shout text
        updated_narration, updated_char_texts = self.chaos_engine.apply_chaos(
            self.test_narration, self.test_char_texts.copy()
        )
        self.assertEqual(
            updated_char_texts["Alice"], self.test_char_texts["Alice"].upper() + "!!"
        )
        self.assertEqual(
            updated_char_texts["Bob"], self.test_char_texts["Bob"].upper() + "!!"
        )

    @patch("requests.get")
    def test_nasa_api_exception_handling(self, mock_requests):
        """Test exception handling for NASA API requests."""
        self.chaos_engine.nasa_api_key = "VALID_API_KEY"
        mock_requests.side_effect = Exception("API Error")
        updated_narration, updated_char_texts = self.chaos_engine.apply_chaos(
            self.test_narration, self.test_char_texts.copy()
        )
        self.assertIn("[CHAOS EVENT]", updated_narration)
        self.assertIn("disrupting sensor arrays", updated_narration)


if __name__ == "__main__":
    unittest.main()
