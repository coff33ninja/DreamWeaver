import pytest
import sqlite3
import os
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Add SERVER/src to sys.path to allow importing database
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from database import Database

# Test database will be in-memory
TEST_DB_PATH = ":memory:"

@pytest.fixture
def db():
    """Fixture to create and tear down an in-memory database for each test."""
    # Patch threading.local to simplify testing single-threaded scenarios
    # or to ensure connection is not shared unexpectedly across test parameterizations if an issue arises.
    # For basic sqlite3 :memory:, direct use might be fine.
    # If issues with _thread_local occur, this is where to patch it.
    # with patch('threading.local') as mock_thread_local:
    #     mock_thread_local.return_value = MagicMock()
    database = Database(TEST_DB_PATH)
    yield database
    database.close()

class TestDatabaseInitialization:
    def test_db_creation_and_schema(self, db):
        """Test that the database is created and tables exist."""
        conn = db._get_conn()
        cursor = conn.cursor()

        tables_to_check = ["characters", "story_log", "training_data", "client_tokens"]
        for table_name in tables_to_check:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            assert cursor.fetchone() is not None, f"Table {table_name} should exist."

        # Check WAL mode
        cursor.execute("PRAGMA journal_mode;")
        assert cursor.fetchone()['journal_mode'].lower() == 'wal', "WAL mode should be enabled."

    def test_ensure_column_functionality(self, db):
        """Test the _ensure_column helper adds a column if it doesn't exist."""
        # Add a test column to an existing table
        db._ensure_column("characters", "test_new_col", "TEXT")
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(characters);")
        columns = [row['name'] for row in cursor.fetchall()]
        assert "test_new_col" in columns

        # Call again, should not fail
        db._ensure_column("characters", "test_new_col", "TEXT")
        cursor.execute("PRAGMA table_info(characters);")
        columns_after = [row['name'] for row in cursor.fetchall()]
        assert columns.count("test_new_col") == columns_after.count("test_new_col")


    def test_ensure_index_functionality(self, db):
        """Test the _ensure_index helper creates an index."""
        db._ensure_index("test_idx_characters_name", "characters", "name")
        conn = db._get_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA index_list(characters);")
        indexes = [row['name'] for row in cursor.fetchall()]
        assert "test_idx_characters_name" in indexes

        # Call again, should not fail (CREATE INDEX IF NOT EXISTS)
        db._ensure_index("test_idx_characters_name", "characters", "name")


class TestCharacterManagement:
    def test_save_and_get_character(self, db):
        """Test saving a new character and retrieving it."""
        db.save_character(
            name="Hero", personality="Brave", goals="Save world", backstory="Mysterious",
            tts="gtts", tts_model="en", reference_audio_filename=None,
            Actor_id="Actor2", llm_model="tinyllama"
        )
        char = db.get_character("Actor2")
        assert char is not None
        assert char["name"] == "Hero"
        assert char["Actor_id"] == "Actor2"
        assert char["llm_model"] == "tinyllama"
        assert char["tts"] == "gtts"

    def test_get_non_existent_character(self, db):
        """Test getting a character that does not exist."""
        char = db.get_character("NonExistentActor")
        assert char is None

    def test_update_existing_character(self, db):
        """Test updating an existing character's details."""
        db.save_character(
            name="Hero", personality="Brave", goals="Save world", backstory="Mysterious",
            tts="gtts", tts_model="en", reference_audio_filename=None,
            Actor_id="Actor3", llm_model="tinyllama"
        )
        db.save_character( # Update
            name="SuperHero", personality="Very Brave", goals="Save Universe", backstory="Known",
            tts="xttsv2", tts_model="custom", reference_audio_filename="hero.wav",
            Actor_id="Actor3", llm_model="megallama"
        )
        char = db.get_character("Actor3")
        assert char["name"] == "SuperHero"
        assert char["personality"] == "Very Brave"
        assert char["tts"] == "xttsv2"
        assert char["llm_model"] == "megallama"
        assert char["reference_audio_filename"] == "hero.wav"

    def test_get_characters_by_ids(self, db):
        """Test retrieving multiple characters by their Actor_ids."""
        chars_data = [
            {"name": "CharA", "Actor_id": "ActorA", "personality": "A", "goals": "A", "backstory": "A", "tts": "gtts", "tts_model": "en", "reference_audio_filename": None, "llm_model": "modelA"},
            {"name": "CharB", "Actor_id": "ActorB", "personality": "B", "goals": "B", "backstory": "B", "tts": "xttsv2", "tts_model": "modelB", "reference_audio_filename": "b.wav", "llm_model": "modelB"},
            {"name": "CharC", "Actor_id": "ActorC", "personality": "C", "goals": "C", "backstory": "C", "tts": "gtts", "tts_model": "es", "reference_audio_filename": None, "llm_model": "modelC"},
        ]
        for char_data in chars_data:
            db.save_character(**char_data)

        retrieved_chars = db.get_characters_by_ids(["ActorA", "ActorC", "NonExistent"])
        assert len(retrieved_chars) == 2
        assert "ActorA" in retrieved_chars
        assert "ActorC" in retrieved_chars
        assert retrieved_chars["ActorA"]["name"] == "CharA"
        assert retrieved_chars["ActorC"]["llm_model"] == "modelC"
        assert "NonExistent" not in retrieved_chars

    def test_get_characters_by_ids_empty_list(self, db):
        """Test get_characters_by_ids with an empty list of IDs."""
        retrieved_chars = db.get_characters_by_ids([])
        assert retrieved_chars == {}

    def test_get_characters_by_ids_no_matches(self, db):
        """Test get_characters_by_ids when no IDs match."""
        db.save_character(name="CharA", Actor_id="ActorA", personality="A", goals="A", backstory="A", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="modelA")
        retrieved_chars = db.get_characters_by_ids(["NonExistent1", "NonExistent2"])
        assert retrieved_chars == {}

# Placeholder for other test classes
class TestStoryLogManagement:
    def test_save_story_entry_and_get_history(self, db):
        db.save_story_entry("Narrator", "Chapter 1 begins.", "narrator_audio_ch1.wav")
        db.save_story_entry("Hero", "I am ready!", None) # No audio for character here

        history = db.get_story_history()
        assert len(history) == 2
        assert history[0]["speaker"] == "Narrator"
        assert history[0]["text"] == "Chapter 1 begins."
        assert history[0]["narrator_audio_path"] == "narrator_audio_ch1.wav"
        assert history[1]["speaker"] == "Hero"
        assert history[1]["text"] == "I am ready!"
        assert history[1]["narrator_audio_path"] is None # Or check for actual value if applicable
        assert "timestamp" in history[0]

    def test_save_story_multiple_entries(self, db):
        db.save_story("Narrator action.", {"Char1": "Response 1", "Char2": "Response 2"}, "narrator_multi.wav")
        history = db.get_story_history()
        assert len(history) == 3
        assert history[0]["speaker"] == "Narrator"
        assert history[0]["narrator_audio_path"] == "narrator_multi.wav"
        assert history[1]["speaker"] == "Char1"
        assert history[2]["speaker"] == "Char2"

    def test_get_empty_story_history(self, db):
        history = db.get_story_history()
        assert history == []

    def test_update_story_entry(self, db):
        db.save_story_entry("Narrator", "Original text.", "original_audio.wav")
        entry_id = db.get_story_history()[0]['id']

        db.update_story_entry(entry_id, new_text="Updated text.")
        updated_entry = db.get_story_history()[0]
        assert updated_entry['text'] == "Updated text."
        assert updated_entry['narrator_audio_path'] == "original_audio.wav" # Audio path not changed

        db.update_story_entry(entry_id, new_audio_path="updated_audio.wav")
        updated_entry = db.get_story_history()[0]
        assert updated_entry['text'] == "Updated text." # Text not changed
        assert updated_entry['narrator_audio_path'] == "updated_audio.wav"

        db.update_story_entry(entry_id, new_text="Final text.", new_audio_path="final_audio.wav")
        updated_entry = db.get_story_history()[0]
        assert updated_entry['text'] == "Final text."
        assert updated_entry['narrator_audio_path'] == "final_audio.wav"

        # Test updating non-existent entry (should not fail, just do nothing)
        assert db.update_story_entry(999, new_text="No such entry") is True # _execute_query returns cursor
        history = db.get_story_history()
        assert len(history) == 1 # No new entry added

        # Test no fields to update
        assert db.update_story_entry(entry_id) is False


class TestTrainingDataManagement:
    def test_save_and_get_training_data(self, db):
        # Need a character for foreign key constraint
        db.save_character(name="Trainer", Actor_id="TrainerActor", personality="Trainer", goals="Train", backstory="Trainer", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")

        dataset1 = {"input": "Hello", "output": "Hi there!"}
        dataset2 = {"input": "How are you?", "output": "I am fine."}
        db.save_training_data(dataset1, "TrainerActor")
        db.save_training_data(dataset2, "TrainerActor")

        data = db.get_training_data_for_Actor("TrainerActor")
        assert len(data) == 2
        assert dataset1 in data
        assert dataset2 in data

    def test_get_training_data_for_non_existent_actor(self, db):
        data = db.get_training_data_for_Actor("NonExistentTrainer")
        assert data == []

    def test_get_training_data_when_no_data_exists(self, db):
        db.save_character(name="NoDataActor", Actor_id="NoDataActorId", personality="ND", goals="ND", backstory="ND", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        data = db.get_training_data_for_Actor("NoDataActorId")
        assert data == []


class TestClientTokenManagement:
    ACTOR_ID = "Client1"
    TOKEN = "test_token_123"

    def setup_method(self, method):
        """Ensure Actor1 exists for some tests that might depend on it implicitly."""
        # This is a bit of a workaround for tests that might interact with Actor1 indirectly
        # or if save_client_token for Actor1 needs Actor1 character to exist.
        # db_instance = Database(TEST_DB_PATH) # Get a fresh instance if needed
        # if not db_instance.get_character("Actor1"):
        #     db_instance.save_character(name="ServerChar", personality="Host", goals="Manage story", backstory="Server's own character",
        #                         tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, Actor_id="Actor1", llm_model=None)
        # db_instance.close()
        pass


    def test_save_and_get_client_token_details(self, db):
        db.save_character(name=self.ACTOR_ID, Actor_id=self.ACTOR_ID, personality="Test", goals="Test", backstory="Test", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        db.save_client_token(self.ACTOR_ID, self.TOKEN)

        details = db.get_client_token_details(self.ACTOR_ID)
        assert details is not None
        assert details["token"] == self.TOKEN
        assert details["status"] == "Registered"
        assert details["ip_address"] is None
        assert details["client_port"] is None
        assert "last_seen" in details # Should have a timestamp

    def test_save_client_token_for_actor1_creates_placeholder(self, db):
        """Test that saving a token for Actor1 creates a placeholder character if it doesn't exist."""
        assert db.get_character("Actor1") is None # Ensure Actor1 doesn't exist yet
        db.save_client_token("Actor1", "actor1_token")

        actor1_char = db.get_character("Actor1")
        assert actor1_char is not None
        assert actor1_char["name"] == "ServerChar" # Default name from save_client_token

        actor1_token_details = db.get_client_token_details("Actor1")
        assert actor1_token_details is not None
        assert actor1_token_details["token"] == "actor1_token"


    def test_get_primary_token(self, db):
        db.save_character(name=self.ACTOR_ID, Actor_id=self.ACTOR_ID, personality="Test", goals="Test", backstory="Test", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        db.save_client_token(self.ACTOR_ID, self.TOKEN)
        token = db.get_primary_token(self.ACTOR_ID)
        assert token == self.TOKEN

    def test_get_primary_token_non_existent(self, db):
        token = db.get_primary_token("NonExistentClient")
        assert token is None

    def test_update_client_session_token(self, db):
        db.save_character(name=self.ACTOR_ID, Actor_id=self.ACTOR_ID, personality="Test", goals="Test", backstory="Test", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        db.save_client_token(self.ACTOR_ID, self.TOKEN)

        session_token = "session_abc_123"
        expiry_time = datetime.now(timezone.utc) + timedelta(hours=1)
        db.update_client_session_token(self.ACTOR_ID, session_token, expiry_time)

        details = db.get_client_token_details(self.ACTOR_ID)
        assert details["session_token"] == session_token
        assert datetime.fromisoformat(details["session_token_expiry"]) == expiry_time

        # Clear session token
        db.update_client_session_token(self.ACTOR_ID, None, None)
        details_cleared = db.get_client_token_details(self.ACTOR_ID)
        assert details_cleared["session_token"] is None
        assert details_cleared["session_token_expiry"] is None


    def test_register_client(self, db):
        db.save_character(name=self.ACTOR_ID, Actor_id=self.ACTOR_ID, personality="Test", goals="Test", backstory="Test", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        db.save_client_token(self.ACTOR_ID, self.TOKEN) # Client must have a token first

        ip = "127.0.0.1"
        port = 8001
        db.register_client(self.ACTOR_ID, ip, port)

        details = db.get_client_token_details(self.ACTOR_ID)
        assert details["ip_address"] == ip
        assert details["client_port"] == port
        assert details["status"] == "Online_Heartbeat" # register_client sets this
        assert "last_seen" in details


    def test_update_client_status(self, db):
        db.save_character(name=self.ACTOR_ID, Actor_id=self.ACTOR_ID, personality="Test", goals="Test", backstory="Test", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="test")
        db.save_client_token(self.ACTOR_ID, self.TOKEN)

        db.update_client_status(self.ACTOR_ID, "Online_Responsive")
        details = db.get_client_token_details(self.ACTOR_ID)
        assert details["status"] == "Online_Responsive"

        # Test with explicit last_seen
        custom_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        db.update_client_status(self.ACTOR_ID, "Offline", custom_time)
        details_offline = db.get_client_token_details(self.ACTOR_ID)
        assert details_offline["status"] == "Offline"
        assert details_offline["last_seen"] == custom_time


    def test_get_clients_for_story_progression(self, db):
        # Setup clients
        actors_data = [
            {"Actor_id": "ClientResponsive", "status": "Online_Responsive", "token": "t1"},
            {"Actor_id": "ClientHeartbeat", "status": "Online_Heartbeat", "token": "t2"},
            {"Actor_id": "ClientOffline", "status": "Offline", "token": "t3"},
            {"Actor_id": "ClientResponsiveOld", "status": "Online_Responsive", "token": "t4"}, # This one will be too old
            {"Actor_id": "Actor1", "status": "Online_Responsive", "token": "t_actor1"}, # Should be excluded
        ]
        for data in actors_data:
            db.save_character(name=data["Actor_id"], Actor_id=data["Actor_id"], personality="P", goals="G", backstory="B", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="m")
            db.save_client_token(data["Actor_id"], data["token"])
            # For ClientResponsiveOld, set last_seen to be too old
            last_seen = datetime.now(timezone.utc)
            if data["Actor_id"] == "ClientResponsiveOld":
                last_seen -= timedelta(days=1) # Make it old

            db.update_client_status(data["Actor_id"], data["status"], last_seen.isoformat())
            if data["status"] == "Online_Responsive": # register_client also sets ip/port
                 db.register_client(data["Actor_id"], "1.2.3.4", 1234)
                 db.update_client_status(data["Actor_id"], data["status"], last_seen.isoformat()) # re-apply status after register

        responsive_clients = db.get_clients_for_story_progression()
        assert len(responsive_clients) == 1
        assert responsive_clients[0]["Actor_id"] == "ClientResponsive"

    def test_get_all_client_statuses(self, db):
        actors_data = [
            {"Actor_id": "ClientA", "status": "Online_Responsive", "token": "ta"},
            {"Actor_id": "ClientB", "status": "Offline", "token": "tb"},
            {"Actor_id": "Actor1", "status": "Online_Responsive", "token": "t_actor1_all"},
        ]
        for data in actors_data:
            db.save_character(name=data["Actor_id"], Actor_id=data["Actor_id"], personality="P", goals="G", backstory="B", tts="gtts", tts_model="en", reference_audio_filename=None, llm_model="m")
            db.save_client_token(data["Actor_id"], data["token"])
            db.update_client_status(data["Actor_id"], data["status"])

        all_statuses = db.get_all_client_statuses()
        assert len(all_statuses) == 2 # Actor1 should be excluded
        actor_ids_retrieved = [client['Actor_id'] for client in all_statuses]
        assert "ClientA" in actor_ids_retrieved
        assert "ClientB" in actor_ids_retrieved
        assert "Actor1" not in actor_ids_retrieved
        # Check ordering (default is by Actor_id ASC)
        assert actor_ids_retrieved[0] == "ClientA"
        assert actor_ids_retrieved[1] == "ClientB"

class TestDatabaseConnection:
    def test_close_connection(self, db):
        """Test that the database connection can be closed."""
        # Connection is typically established in fixture.
        # We can try to close it and then attempt an operation that would fail.
        conn_before_close = db._thread_local.conn
        db.close()
        assert not hasattr(db._thread_local, 'conn')

        # Verify that a new connection is made if _get_conn is called after close
        # This implicitly tests that the old connection was indeed removed from _thread_local
        new_conn = db._get_conn()
        assert new_conn is not None
        assert new_conn != conn_before_close # Should be a new connection object

    # __del__ is hard to test deterministically.
    # We rely on the fact that close() is called by __del__.
    # A more complex test might involve weak PReferences or gc.collect(),
    # but that's often overkill for this kind of check.
