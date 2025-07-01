import sqlite3
import threading
from datetime import datetime, timezone, timedelta

class Database:
    def __init__(self, db_path):
        """
        Initialize the Database instance with the specified SQLite file path, set up thread-local storage for connections, and ensure the required database schema exists.
        """
        self.db_path = db_path
        self._thread_local = threading.local()
        # The initial connection for the main thread is made here to ensure the schema exists.
        # Other threads will create their own connections on first use.
        self._get_conn()
        self._ensure_schema()
        print("Database schema ensured.")

    def _get_conn(self):
        """
        Retrieve or create a SQLite database connection specific to the current thread.
        
        Returns:
            sqlite3.Connection: The thread-local SQLite connection object.
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
        """
        Executes a SQL query with optional parameters and supports committing transactions and fetching results.
        
        Parameters:
            query (str): The SQL query to execute.
            params (tuple or list, optional): Parameters to substitute into the SQL query.
            commit (bool): Whether to commit the transaction after executing the query.
            fetchone (bool): If True, returns a single result row as a dictionary.
            fetchall (bool): If True, returns all result rows as a list of dictionaries.
        
        Returns:
            The query result based on fetch mode: a single row, a list of rows, the cursor object, or None if an error occurs.
        """
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
        """
        Ensure that a specified column exists in a given table, adding it with the provided definition if missing.
        
        Parameters:
            table_name (str): Name of the table to check or modify.
            column_name (str): Name of the column to ensure exists.
            column_definition (str): SQL definition for the column if it needs to be added.
        """
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
        """
        Ensures that all required database tables and columns exist, creating or updating them as necessary for schema consistency.
        
        This includes creating the `characters`, `story_log`, `training_data`, and `client_tokens` tables with appropriate fields and constraints, and adding any missing columns to support new features or schema changes.
        """
        self._execute_query("""
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                personality TEXT, goals TEXT, backstory TEXT, tts TEXT, tts_model TEXT,
                reference_audio_filename TEXT,
                Actor_id TEXT NOT NULL UNIQUE,
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
                Actor_id TEXT NOT NULL, input_text TEXT NOT NULL, output_text TEXT NOT NULL,
                FOREIGN KEY (Actor_id) REFERENCES characters(Actor_id)
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
                Actor_id TEXT PRIMARY KEY,
                token TEXT NOT NULL UNIQUE,
                ip_address TEXT,
                client_port INTEGER,
                last_seen DATETIME,
                status TEXT DEFAULT 'Registered', -- Default for new tokens
                FOREIGN KEY (Actor_id) REFERENCES characters(Actor_id)
            )
        """)
        self._ensure_column("client_tokens", "client_port", "INTEGER")
        self._ensure_column("client_tokens", "status", "TEXT DEFAULT 'Registered'") # Ensure default is set if column added

        self._get_conn().commit()

    def save_character(self, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, Actor_id, llm_model=None):
        """
        Insert a new character or update an existing character record identified by Actor_id with the provided attributes.
        
        If a character with the given Actor_id already exists, its details are updated with the new values.
        """
        query = """
            INSERT INTO characters (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, Actor_id, llm_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Actor_id) DO UPDATE SET
                name=excluded.name, personality=excluded.personality, goals=excluded.goals, backstory=excluded.backstory,
                tts=excluded.tts, tts_model=excluded.tts_model, reference_audio_filename=excluded.reference_audio_filename,
                llm_model=excluded.llm_model
        """
        self._execute_query(query, (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, Actor_id, llm_model), commit=True)
        print(f"Character '{name}' for Actor_id '{Actor_id}' saved/updated.")

    def get_character(self, Actor_id):
        # Ensure llm_model is selected
        """
        Retrieve a character record by its Actor_id.
        
        Returns:
            dict: A dictionary containing the character's details if found, otherwise None.
        """
        row = self._execute_query("SELECT id, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, Actor_id, llm_model FROM characters WHERE Actor_id=?", (Actor_id,), fetchone=True)
        return dict(row) if row else None

    def save_story_entry(self, speaker, text, narrator_audio_path=None):
        """
        Insert a new entry into the story log with the specified speaker, text, optional narrator audio path, and the current UTC timestamp.
        """
        query = "INSERT INTO story_log (speaker, text, narrator_audio_path, timestamp) VALUES (?, ?, ?, ?)"
        self._execute_query(query, (speaker, text, narrator_audio_path, datetime.now(timezone.utc).isoformat()), commit=True)

    def save_story(self, narration_text, character_texts, narrator_audio_path=None):
        """
        Save a story entry consisting of a narrator's text and multiple character texts to the story log.
        
        Parameters:
            narration_text (str): The text narrated by the narrator.
            character_texts (dict): A mapping of character names to their respective dialogue texts.
            narrator_audio_path (str, optional): Path to the narrator's audio file, if available.
        """
        self.save_story_entry("Narrator", narration_text, narrator_audio_path)
        for char_name, char_text in character_texts.items():
            self.save_story_entry(char_name, char_text)
        # print("Story progress saved.") # Can be too verbose

    def get_story_history(self):
        """
        Retrieve the complete story log history as a list of entries.
        
        Returns:
            List[dict]: A list of dictionaries, each containing the ID, speaker, text, timestamp, and narrator audio path for a story log entry, ordered by timestamp.
        """
        rows = self._execute_query("SELECT id, speaker, text, timestamp, narrator_audio_path FROM story_log ORDER BY timestamp ASC", fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def save_training_data(self, dataset, Actor_id):
        """
        Insert a new training data entry associated with a specific Actor.
        
        Parameters:
            dataset (dict): A dictionary containing 'input' and 'output' text fields.
            Actor_id (str): The unique identifier of the actor to associate with the training data.
        """
        query = "INSERT INTO training_data (Actor_id, input_text, output_text) VALUES (?, ?, ?)"
        self._execute_query(query, (Actor_id, dataset["input"], dataset["output"]), commit=True)

    def get_training_data_for_Actor(self, Actor_id):
        """
        Retrieve all training data entries associated with a specific Actor.
        
        Parameters:
            Actor_id (str): The unique identifier of the Actor whose training data is to be retrieved.
        
        Returns:
            List[dict]: A list of dictionaries, each containing 'input' and 'output' keys representing the input and output text pairs for the Actor. Returns an empty list if no data is found.
        """
        rows = self._execute_query("SELECT input_text, output_text FROM training_data WHERE Actor_id=?", (Actor_id,), fetchall=True)
        return [{"input": row["input_text"], "output": row["output_text"]} for row in rows] if rows else []

    def save_client_token(self, Actor_id, token):
        """
        Save or update a client token for the specified Actor_id, setting status to 'Registered' and updating the last seen timestamp.
        
        If the Actor_id is "Actor1" and no corresponding character exists, a placeholder character is created.
        """
        # Ensure character exists or create a placeholder if it's Actor1
        if Actor_id == "Actor1" and not self.get_character("Actor1"):
            self.save_character(name="ServerChar", personality="Host", goals="Manage story", backstory="Server's own character",
                                tts="piper", tts_model="en_US-ryan-high", reference_audio_filename=None, Actor_id="Actor1", llm_model=None)

        query = """
            INSERT INTO client_tokens (Actor_id, token, status, last_seen) VALUES (?, ?, 'Registered', ?)
            ON CONFLICT(Actor_id) DO UPDATE SET token=excluded.token, status='Registered', last_seen=excluded.last_seen
        """
        self._execute_query(query, (Actor_id, token, datetime.now(timezone.utc).isoformat()), commit=True)
        print(f"Token for '{Actor_id}' saved. Status: Registered.")

    def get_client_token_details(self, Actor_id):
        """
        Retrieve detailed client token information for a given Actor ID.
        
        Returns:
            dict: A dictionary containing the token, status, IP address, client port, and last seen timestamp if found; otherwise, None.
        """
        row = self._execute_query("SELECT token, status, ip_address, client_port, last_seen FROM client_tokens WHERE Actor_id=?", (Actor_id,), fetchone=True)
        return dict(row) if row else None

    def get_token(self, Actor_id):
        """
        Retrieve the token string associated with the specified Actor ID.
        
        Returns:
            str or None: The token if found, otherwise None.
        """
        details = self.get_client_token_details(Actor_id)
        return details['token'] if details else None

    def register_client(self, Actor_id, ip_address, client_port):
        """
        Registers or updates a client by setting its IP address, port, last seen timestamp, and status to 'Online_Heartbeat'.
        
        Parameters:
            Actor_id (str): Unique identifier for the client.
            ip_address (str): IP address of the client.
            client_port (int): Port number used by the client.
        """
        # Server will verify API responsiveness separately to move to 'Online_Responsive'
        timestamp_utc_iso = datetime.now(timezone.utc).isoformat()
        query = """
            UPDATE client_tokens
            SET ip_address = ?, client_port = ?, last_seen = ?, status = 'Online_Heartbeat'
            WHERE Actor_id = ?
        """
        self._execute_query(query, (ip_address, client_port, timestamp_utc_iso, Actor_id), commit=True)
        print(f"Client '{Actor_id}' registered from {ip_address}:{client_port}. Status: Online_Heartbeat.")

    def update_client_status(self, Actor_id, new_status, last_seen_iso=None):
        """
        Update the status and optionally the last seen timestamp for a client token.
        
        Parameters:
            Actor_id (str): The unique identifier for the client.
            new_status (str): The new status to set for the client.
            last_seen_iso (str, optional): ISO 8601 formatted timestamp for the last seen time. If not provided, the current UTC time is used.
        """
        if last_seen_iso is None:
            last_seen_iso = datetime.now(timezone.utc).isoformat()

        query = "UPDATE client_tokens SET status = ?, last_seen = ? WHERE Actor_id = ?"
        self._execute_query(query, (new_status, last_seen_iso, Actor_id), commit=True)
        # print(f"Status for {Actor_id} updated to {new_status}.")


    def get_clients_for_story_progression(self):
        """
        Retrieve a list of clients that are currently responsive and eligible for story progression, excluding "Actor1".
        
        Returns:
            List[dict]: A list of dictionaries containing 'Actor_id', 'ip_address', and 'client_port' for each responsive client.
        """
        # Define "recent" more dynamically, e.g., 2.5 * heartbeat interval (assuming 60s)
        # This could be passed from config or ClientManager if it knows the heartbeat interval
        recent_threshold = (datetime.now(timezone.utc) - timedelta(seconds=150)).isoformat()
        query = """
            SELECT Actor_id, ip_address, client_port
            FROM client_tokens
            WHERE status = 'Online_Responsive' AND last_seen >= ? AND Actor_id != 'Actor1'
        """
        rows = self._execute_query(query, (recent_threshold,), fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def get_all_client_statuses(self):
        """
        Retrieve all client token records except for 'Actor1', including status and connection details.
        
        Returns:
            List[dict]: A list of dictionaries containing Actor_id, IP address, client port, last seen timestamp, and status for each client.
        """
        query = "SELECT Actor_id, ip_address, client_port, last_seen, status FROM client_tokens WHERE Actor_id != 'Actor1' ORDER BY Actor_id ASC"
        rows = self._execute_query(query, fetchall=True)
        return [dict(row) for row in rows] if rows else []

    def close(self):
        """
        Closes the current thread's SQLite database connection if it exists.
        """
        if hasattr(self._thread_local, 'conn'):
            self._thread_local.conn.close()
            del self._thread_local.conn

    def __del__(self):
        """
        Destructor that ensures the database connection is closed when the object is deleted.
        """
        self.close()

    def update_story_entry(self, entry_id, new_text=None, new_audio_path=None):
        """
        Update the text and/or narrator audio path of a story log entry by its ID.
        
        Parameters:
            entry_id (int): The unique identifier of the story log entry to update.
            new_text (str, optional): The new text to set for the entry.
            new_audio_path (str, optional): The new narrator audio file path to set.
        
        Returns:
            bool: True if an update was performed, False if no fields were provided to update.
        """
        sets = []
        params = []
        if new_text is not None:
            sets.append("text = ?")
            params.append(new_text)
        if new_audio_path is not None:
            sets.append("narrator_audio_path = ?")
            params.append(new_audio_path)
        if not sets:
            return False
        params.append(entry_id)
        query = f"UPDATE story_log SET {', '.join(sets)} WHERE id = ?"
        self._execute_query(query, tuple(params), commit=True)
        return True
