import sqlite3
import threading
from datetime import datetime, timezone, timedelta

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self._thread_local = threading.local()
        # The initial connection for the main thread is made here to ensure the schema exists.
        # Other threads will create their own connections on first use.
        self._get_conn()
        self._ensure_schema()
        print("Database schema ensured.")

    def _get_conn(self):
        """
        Gets a database connection for the current thread.
        If one doesn't exist, it creates it.
        """
        if not hasattr(self._thread_local, 'conn'):
            try:
                # Each thread gets its own connection. No need for check_same_thread=False.
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                self._thread_local.conn = conn
            except sqlite3.Error as e:
                print(f"Error connecting to database in thread {threading.get_ident()}: {e}")
                raise
        return self._thread_local.conn

    def _execute_query(self, query, params=None, commit=False, fetchone=False, fetchall=False):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            if commit:
                conn.commit()
            if fetchone:
                return cursor.fetchone()
            if fetchall:
                return cursor.fetchall()
            return cursor
        except sqlite3.Error as e:
            print(f"Database error: {e}\nQuery: {query}\nParams: {params}")
            return None

    def _ensure_column(self, table_name, column_name, column_definition):
        """Helper to add a column if it doesn't exist."""
        try:
            # Check if column exists (more robust way for SQLite)
            cursor = self._execute_query(f"PRAGMA table_info({table_name});", fetchall=True)
            if cursor is None: # Error in _execute_query
                print(f"Database: Could not get table_info for {table_name}. Column '{column_name}' check skipped.")
                return

            columns = [row['name'] for row in cursor]
            if column_name not in columns:
                self._execute_query(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition};", commit=True)
                print(f"Database: Added '{column_name}' column to '{table_name}' table.")
            # else: # For debugging
            #    print(f"Database: Column '{column_name}' already exists in '{table_name}'.")

        except sqlite3.OperationalError as e: # Should be caught by _execute_query mostly
            print(f"Database: Error ensuring column '{column_name}' in '{table_name}': {e}")


    def _ensure_schema(self):
        """Ensures all tables and specified columns exist."""
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                personality TEXT, goals TEXT, backstory TEXT, tts TEXT, tts_model TEXT,
                reference_audio_filename TEXT,
                pc_id TEXT NOT NULL UNIQUE,
                llm_model TEXT -- Added for character-specific LLM model choice
            )
        """)
        self._ensure_column("characters", "llm_model", "TEXT")


        self._execute_query("""
            CREATE TABLE IF NOT EXISTS story_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker TEXT NOT NULL, text TEXT NOT NULL,
                narrator_audio_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._ensure_column("story_log", "narrator_audio_path", "TEXT")

        self._execute_query("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pc_id TEXT NOT NULL, input_text TEXT NOT NULL, output_text TEXT NOT NULL,
                FOREIGN KEY (pc_id) REFERENCES characters(pc_id)
            )
        """)

        # Define new client statuses
        # 'Registered': Token created, client not yet connected.
        # 'Offline': Client was connected, now presumed disconnected (no recent heartbeat).
        # 'Online_Heartbeat': Client is sending heartbeats. API responsiveness unknown.
        # 'Online_Responsive': Client is sending heartbeats AND server confirmed API is responsive.
        # 'Error_API': Client heartbeating, but its API is not responding to server checks.
        # 'Error_Unreachable': Client cannot be reached at all (failed registration, multiple failed heartbeats).
        # 'Deactivated': Admin manually deactivated this client slot.
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS client_tokens (
                pc_id TEXT PRIMARY KEY,
                token TEXT NOT NULL UNIQUE,
                ip_address TEXT,
                client_port INTEGER,
                last_seen DATETIME,
                status TEXT DEFAULT 'Registered', -- Default for new tokens
                FOREIGN KEY (pc_id) REFERENCES characters(pc_id)
            )
        """)
        self._ensure_column("client_tokens", "client_port", "INTEGER")
        self._ensure_column("client_tokens", "status", "TEXT DEFAULT 'Registered'") # Ensure default is set if column added

        self._get_conn().commit()

    def save_character(self, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id, llm_model=None):
        query = """
            INSERT INTO characters (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id, llm_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pc_id) DO UPDATE SET
                name=excluded.name, personality=excluded.personality, goals=excluded.goals, backstory=excluded.backstory,
                tts=excluded.tts, tts_model=excluded.tts_model, reference_audio_filename=excluded.reference_audio_filename,
                llm_model=excluded.llm_model
        """
        self._execute_query(query, (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id, llm_model), commit=True)
        print(f"Character '{name}' for pc_id '{pc_id}' saved/updated.")

    def get_character(self, pc_id):
        # Ensure llm_model is selected
        row = self._execute_query("SELECT id, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc_id, llm_model FROM characters WHERE pc_id=?", (pc_id,), fetchone=True)
        return dict(row) if row else None

    def save_story_entry(self, speaker, text, narrator_audio_path=None):
        query = "INSERT INTO story_log (speaker, text, narrator_audio_path, timestamp) VALUES (?, ?, ?, ?)"
        self._execute_query(query, (speaker, text, narrator_audio_path, datetime.now(timezone.utc).isoformat()), commit=True)

    def save_story(self, narration_text, character_texts, narrator_audio_path=None):
        self.save_story_entry("Narrator", narration_text, narrator_audio_path)
        for char_name, char_text in character_texts.items():
            self.save_story_entry(char_name, char_text)
        # print("Story progress saved.") # Can be too verbose

    def get_story_history(self):
        rows = self._execute_query("SELECT id, speaker, text, timestamp, narrator_audio_path FROM story_log ORDER BY timestamp ASC", fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def save_training_data(self, dataset, pc_id):
        query = "INSERT INTO training_data (pc_id, input_text, output_text) VALUES (?, ?, ?)"
        self._execute_query(query, (pc_id, dataset["input"], dataset["output"]), commit=True)

    def get_training_data_for_pc(self, pc_id):
        rows = self._execute_query("SELECT input_text, output_text FROM training_data WHERE pc_id=?", (pc_id,), fetchall=True)
        return [{"input": row["input_text"], "output": row["output_text"]} for row in rows] if rows else []

    def save_client_token(self, pc_id, token):
        """Saves a new token, defaults status to 'Registered'."""
        # Ensure character exists or create a placeholder if it's PC1
        if pc_id == "PC1" and not self.get_character("PC1"):
            self.save_character(name="ServerChar", personality="Host", goals="Manage story", backstory="Server's own character",
                                tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, pc_id="PC1", llm_model=None)

        query = """
            INSERT INTO client_tokens (pc_id, token, status, last_seen) VALUES (?, ?, 'Registered', ?)
            ON CONFLICT(pc_id) DO UPDATE SET token=excluded.token, status='Registered', last_seen=excluded.last_seen
        """
        self._execute_query(query, (pc_id, token, datetime.now(timezone.utc).isoformat()), commit=True)
        print(f"Token for '{pc_id}' saved. Status: Registered.")

    def get_client_token_details(self, pc_id):
        row = self._execute_query("SELECT token, status, ip_address, client_port, last_seen FROM client_tokens WHERE pc_id=?", (pc_id,), fetchone=True)
        return dict(row) if row else None

    def get_token(self, pc_id):
        details = self.get_client_token_details(pc_id)
        return details['token'] if details else None

    def register_client(self, pc_id, ip_address, client_port):
        """Updates IP, port, and last_seen for a client. Sets status to 'Online_Heartbeat' initially."""
        # Server will verify API responsiveness separately to move to 'Online_Responsive'
        timestamp_utc_iso = datetime.now(timezone.utc).isoformat()
        query = """
            UPDATE client_tokens
            SET ip_address = ?, client_port = ?, last_seen = ?, status = 'Online_Heartbeat'
            WHERE pc_id = ?
        """
        self._execute_query(query, (ip_address, client_port, timestamp_utc_iso, pc_id), commit=True)
        print(f"Client '{pc_id}' registered from {ip_address}:{client_port}. Status: Online_Heartbeat.")

    def update_client_status(self, pc_id, new_status, last_seen_iso=None):
        """Updates the status and optionally last_seen for a client."""
        if last_seen_iso is None:
            last_seen_iso = datetime.now(timezone.utc).isoformat()

        query = "UPDATE client_tokens SET status = ?, last_seen = ? WHERE pc_id = ?"
        self._execute_query(query, (new_status, last_seen_iso, pc_id), commit=True)
        # print(f"Status for {pc_id} updated to {new_status}.")


    def get_clients_for_story_progression(self):
        """
        Gets clients that are fully responsive and ready for story interaction.
        Excludes PC1 as it's handled by CharacterServer.
        """
        # Define "recent" more dynamically, e.g., 2.5 * heartbeat interval (assuming 60s)
        # This could be passed from config or ClientManager if it knows the heartbeat interval
        recent_threshold = (datetime.now(timezone.utc) - timedelta(seconds=150)).isoformat()
        query = """
            SELECT pc_id, ip_address, client_port
            FROM client_tokens
            WHERE status = 'Online_Responsive' AND last_seen >= ? AND pc_id != 'PC1'
        """
        rows = self._execute_query(query, (recent_threshold,), fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def get_all_client_statuses(self):
        """Retrieves all client details for dashboard or admin purposes."""
        query = "SELECT pc_id, ip_address, client_port, last_seen, status FROM client_tokens WHERE pc_id != 'PC1' ORDER BY pc_id ASC"
        rows = self._execute_query(query, fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def close(self):
        if hasattr(self._thread_local, 'conn'):
            self._thread_local.conn.close()
            del self._thread_local.conn

    def __del__(self):
        self.close()
