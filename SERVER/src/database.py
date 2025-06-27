import sqlite3

class Database:
    def __init__(self, db_path):
        try:
            self.conn = sqlite3.connect(db_path)
        except sqlite3.Error as e:
            print(f"Error connecting to database at {db_path}: {e}")
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                name TEXT, personality TEXT, goals TEXT, backstory TEXT, tts TEXT, tts_model TEXT, reference_audio_filename TEXT, pc TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS story (
                speaker TEXT, text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                pc TEXT, input TEXT, output TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                pc TEXT PRIMARY KEY,
                token TEXT,
                ip_address TEXT,
                last_seen DATETIME,
                active BOOLEAN DEFAULT 1
            )
        """)
        self.conn.commit()

    def save_character(self, name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc):
        self.conn.execute("INSERT INTO characters (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (name, personality, goals, backstory, tts, tts_model, reference_audio_filename, pc))
        self.conn.commit()

    def get_character(self, pc):
        cursor = self.conn.execute("SELECT name, personality, goals, backstory, tts, tts_model, reference_audio_filename FROM characters WHERE pc=?", (pc,))
        row = cursor.fetchone()
        return {"name": row[0], "personality": row[1], "goals": row[2], "backstory": row[3], "tts": row[4], "tts_model": row[5], "reference_audio_filename": row[6], "pc": pc} if row else None

    def save_story(self, narration, character_texts):
        self.conn.execute("INSERT INTO story (speaker, text) VALUES (?, ?)", ("Narrator", narration))
        for name, text in character_texts.items():
            self.conn.execute("INSERT INTO story (speaker, text) VALUES (?, ?)", (name, text))
        self.conn.commit()

    def get_story_history(self):
        """Retrieves all story entries from the database."""
        cursor = self.conn.execute("SELECT speaker, text, timestamp FROM story ORDER BY timestamp ASC")
        return cursor.fetchall()

    def save_training_data(self, dataset, pc):
        self.conn.execute("INSERT INTO training_data (pc, input, output) VALUES (?, ?, ?)", (pc, dataset["input"], dataset["output"]))
        self.conn.commit()

    def get_training_data_for_pc(self, pc):
        """Retrieves all training data for a specific PC."""
        cursor = self.conn.execute("SELECT input, output FROM training_data WHERE pc=?", (pc,))
        return [{"input": row[0], "output": row[1]} for row in cursor.fetchall()]

    def save_token(self, pc, token):
        # Use INSERT OR REPLACE to handle creating or updating a character's token
        self.conn.execute("INSERT OR REPLACE INTO tokens (pc, token, active) VALUES (?, ?, ?)", (pc, token, 1))
        self.conn.commit()

    def get_token(self, pc):
        cursor = self.conn.execute("SELECT token FROM tokens WHERE pc=? AND active=1", (pc,))
        row = cursor.fetchone()
        return row[0] if row else None

    def register_client(self, pc, ip_address):
        """Updates the IP address and last_seen timestamp for a client."""
        self.conn.execute("""
            UPDATE tokens
            SET ip_address = ?, last_seen = CURRENT_TIMESTAMP
            WHERE pc = ? AND active = 1
        """, (ip_address, pc))
        self.conn.commit()

    def get_active_clients(self):
        # An active client is one that has registered its IP and has been seen in the last 5 minutes.
        cursor = self.conn.execute("SELECT pc, ip_address FROM tokens WHERE active=1 AND ip_address IS NOT NULL AND last_seen > datetime('now', '-5 minutes')")
        return [{"pc": row[0], "ip_address": row[1]} for row in cursor.fetchall() if row[0] != "PC1"]

    def deactivate_client(self, pc):
        self.conn.execute("UPDATE tokens SET active=0 WHERE pc=?", (pc,))
        self.conn.commit()

    def get_all_client_statuses(self):
        """Retrieves status for all configured clients."""
        cursor = self.conn.execute("SELECT pc, ip_address, last_seen, active FROM tokens WHERE pc != 'PC1' ORDER BY pc ASC")
        return [{"pc": row[0], "ip_address": row[1], "last_seen": row[2], "active": row[3]} for row in cursor.fetchall()]
