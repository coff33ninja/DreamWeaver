import sqlite3
from datetime import datetime, timezone # For explicit timezone handling

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path) # self.conn will be used by other methods
            self.conn.row_factory = sqlite3.Row # Access columns by name
        except sqlite3.Error as e:
            print(f"Error connecting to database at {db_path}: {e}")
            raise # Re-raise the exception if connection fails
        self.create_tables()

    def _execute_query(self, query, params=None, commit=False, fetchone=False, fetchall=False):
        """Helper function for database operations."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            if commit:
                self.conn.commit()
            if fetchone:
                return cursor.fetchone()
            if fetchall:
                return cursor.fetchall()
            return cursor # For operations like lastrowid or rowcount
        except sqlite3.Error as e:
            print(f"Database error: {e}\nQuery: {query}\nParams: {params}")
            # For critical errors, you might want to rollback or handle more gracefully
            # self.conn.rollback()
            return None # Or raise the exception

    def create_tables(self):
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                personality TEXT,
                goals TEXT,
                backstory TEXT,
                tts TEXT,
                tts_model TEXT,
                reference_audio_filename TEXT,
                pc_id TEXT NOT NULL UNIQUE -- Renamed 'pc' to 'pc_id' for clarity
            )
        """)
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS story_log ( -- Renamed 'story' to 'story_log'
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                narrator_audio_path TEXT, -- Added field for narrator's audio
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pc_id TEXT NOT NULL, -- Renamed 'pc' to 'pc_id'
                input_text TEXT NOT NULL, -- Renamed 'input' to 'input_text'
                output_text TEXT NOT NULL, -- Renamed 'output' to 'output_text'
                FOREIGN KEY (pc_id) REFERENCES characters(pc_id)
            )
        """)
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS client_tokens ( -- Renamed 'tokens' to 'client_tokens'
                pc_id TEXT PRIMARY KEY, -- Renamed 'pc' to 'pc_id'
                token TEXT NOT NULL UNIQUE,
                ip_address TEXT,
                client_port INTEGER, -- Added client_port
                last_seen DATETIME,
                status TEXT DEFAULT 'Offline', -- Added status field e.g. Online, Offline, Error
                FOREIGN KEY (pc_id) REFERENCES characters(pc_id)
            )
        """)
        # Add client_port column if it doesn't exist (for existing databases)
        try:
            self._execute_query("ALTER TABLE client_tokens ADD COLUMN client_port INTEGER;")
            print("Database: Added 'client_port' column to 'client_tokens' table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                print(f"Database: Note - Could not add 'client_port' (may already exist or other issue): {e}")
        # Add status column if it doesn't exist
        try:
            self._execute_query("ALTER TABLE client_tokens ADD COLUMN status TEXT DEFAULT 'Offline';")
            print("Database: Added 'status' column to 'client_tokens' table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                print(f"Database: Note - Could not add 'status' (may already exist or other issue): {e}")

        # Add narrator_audio_path column to story_log if it doesn't exist
        try:
            self._execute_query("ALTER TABLE story_log ADD COLUMN narrator_audio_path TEXT;")
            print("Database: Added 'narrator_audio_path' column to 'story_log' table.")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                 print(f"Database: Note - Could not add 'narrator_audio_path' (may already exist or other issue): {e}")
        self.conn.commit()


    def save_character(self, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id):
        query = """
            INSERT INTO characters (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pc_id) DO UPDATE SET
                name=excluded.name,
                personality=excluded.personality,
                goals=excluded.goals,
                backstory=excluded.backstory,
                tts=excluded.tts,
                tts_model=excluded.tts_model,
                reference_audio_filename=excluded.reference_audio_filename
        """
        self._execute_query(query, (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id), commit=True)
        print(f"Character {name} for pc_id {pc_id} saved/updated.")

    def get_character(self, pc_id):
        row = self._execute_query("SELECT * FROM characters WHERE pc_id=?", (pc_id,), fetchone=True)
        return dict(row) if row else None

    def save_story_entry(self, speaker, text, narrator_audio_path=None):
        query = "INSERT INTO story_log (speaker, text, narrator_audio_path, timestamp) VALUES (?, ?, ?, ?)"
        self._execute_query(query, (speaker, text, narrator_audio_path, datetime.now(timezone.utc).isoformat()), commit=True)

    def save_story(self, narration_text, character_texts, narrator_audio_path=None):
        self.save_story_entry("Narrator", narration_text, narrator_audio_path)
        for char_name, char_text in character_texts.items():
            self.save_story_entry(char_name, char_text)
        print("Story progress saved.")

    def get_story_history(self):
        rows = self._execute_query("SELECT speaker, text, timestamp, narrator_audio_path FROM story_log ORDER BY timestamp ASC", fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def save_training_data(self, dataset, pc_id):
        query = "INSERT INTO training_data (pc_id, input_text, output_text) VALUES (?, ?, ?)"
        self._execute_query(query, (pc_id, dataset["input"], dataset["output"]), commit=True)

    def get_training_data_for_pc(self, pc_id):
        rows = self._execute_query("SELECT input_text, output_text FROM training_data WHERE pc_id=?", (pc_id,), fetchall=True)
        return [{"input": row["input_text"], "output": row["output_text"]} for row in rows] if rows else []

    def save_client_token(self, pc_id, token):
        query = """
            INSERT INTO client_tokens (pc_id, token, status) VALUES (?, ?, 'Registered')
            ON CONFLICT(pc_id) DO UPDATE SET token=excluded.token, status='Registered'
        """
        self._execute_query(query, (pc_id, token), commit=True)
        print(f"Token for {pc_id} saved/updated.")


    def get_client_token_details(self, pc_id):
        row = self._execute_query("SELECT token, status FROM client_tokens WHERE pc_id=?", (pc_id,), fetchone=True)
        return dict(row) if row else None

    def get_token(self, pc_id): # Kept for compatibility if some parts still use it simply
        details = self.get_client_token_details(pc_id)
        return details['token'] if details else None

    def register_client(self, pc_id, ip_address, client_port):
        """Updates IP, port, and last_seen for a client. Sets status to 'Online'."""
        timestamp_utc_iso = datetime.now(timezone.utc).isoformat()
        query = """
            UPDATE client_tokens
            SET ip_address = ?, client_port = ?, last_seen = ?, status = 'Online'
            WHERE pc_id = ?
        """
        self._execute_query(query, (ip_address, client_port, timestamp_utc_iso, pc_id), commit=True)
        print(f"Client {pc_id} registered from {ip_address}:{client_port}. Status: Online.")


    def update_client_status(self, pc_id, status, last_seen_iso=None):
        """Updates the status and optionally last_seen for a client."""
        if last_seen_iso is None:
            last_seen_iso = datetime.now(timezone.utc).isoformat()

        query = "UPDATE client_tokens SET status = ?, last_seen = ? WHERE pc_id = ?"
        self._execute_query(query, (status, last_seen_iso, pc_id), commit=True)
        # print(f"Status for {pc_id} updated to {status}.")


    def get_active_clients(self):
        # Consider clients active if Online and seen recently (e.g., within 2 * HEARTBEAT_INTERVAL)
        # HEARTBEAT_INTERVAL_SECONDS is defined in character_client.py, assume 60s for now.
        # Let's say 150 seconds (2.5 * 60) to be safe.
        five_minutes_ago = (datetime.now(timezone.utc) - timedelta(seconds=150)).isoformat() # Corrected import
        from datetime import timedelta # Import timedelta here or at the top of the file

        query = """
            SELECT pc_id, ip_address, client_port
            FROM client_tokens
            WHERE status = 'Online' AND last_seen > ? AND pc_id != 'PC1'
        """
        rows = self._execute_query(query, (five_minutes_ago,), fetchall=True)
        return [dict(row) for row in rows] if rows else []


    def get_all_client_statuses(self):
        query = "SELECT pc_id, ip_address, client_port, last_seen, status FROM client_tokens WHERE pc_id != 'PC1' ORDER BY pc_id ASC"
        rows = self._execute_query(query, fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def __del__(self):
        self.close()

    # Example method to add a character if it doesn't exist, then save token
    def ensure_character_and_save_token(self, pc_id, token, char_details=None):
        char = self.get_character(pc_id)
        if not char and char_details:
            self.save_character(
                name=char_details.get('name', pc_id),
                personality=char_details.get('personality', 'N/A'),
                goals=char_details.get('goals', 'N/A'),
                backstory=char_details.get('backstory', 'N/A'),
                tts=char_details.get('tts', 'piper'),
                tts_model=char_details.get('tts_model', 'en_US-ryan-high'),
                reference_audio_filename=char_details.get('reference_audio_filename'),
                pc_id=pc_id
            )
        elif not char and pc_id == "PC1": # Default for PC1 if not found
             self.save_character(pc_id, "Server Character", "To narrate and assist.", "Exists within the server.", "piper", "en_US-ryan-high", None, "PC1")


        self.save_client_token(pc_id, token)
