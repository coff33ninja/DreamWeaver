# DreamWeaver - A Decentralized, AI-Powered Storytelling Network

Dream Weaver is a scalable, decentralized storytelling network where a human narrator guides a story populated by a hive of AI-driven characters. Each character runs on a separate machine, possesses a unique personality and voice, and evolves over time through local fine-tuning of its own Large Language Model (LLM).

The system is managed from a central server that hosts the narrator's interface, synchronizes the story state, and provides a real-time dashboard to monitor the health of the character hive.

---

## Key Features

*   **Decentralized Character Hive**: Characters run as independent clients on separate machines (PC2, PC3, ...), each with its own LLM, TTS engine, and configurable API endpoint.
*   **Dynamic Character Creation & Evolution**: Add or remove characters on the fly via the Gradio UI. Characters evolve as their local LLMs are fine-tuned with new dialogue, and the training data is backed up to the central server.
*   **Robust Client Discovery & Communication**: Clients register their listening port with the server and maintain their connection via a periodic heartbeat. The server communicates with clients on their specified ports.
*   **Flexible TTS Engine**: Supports multiple Text-to-Speech services, including Piper, Google TTS (requires setup), and Coqui's XTTS-v2 for high-quality voice cloning from a reference audio file.
*   **Interactive Gradio Interface**: A central web UI for:
    *   Narrating the story via microphone (using Whisper STT).
    *   Creating and configuring characters, including their PC assignments.
    *   Generating secure access tokens for clients.
    *   Reviewing the complete story history in a chatbot format.
    *   Managing story checkpoints.
    *   Exporting the story to JSON or TXT.
*   **Story Checkpoints & Export**: Save the entire story state (database and server-side model adapters) at any point. Load previous checkpoints to explore different narrative branches.
*   **Real-time Monitoring Dashboard**: A web-based dashboard to monitor server CPU/memory and the online/offline status (including IP and port) of all connected clients.
*   **Chaos Mode**: Injects random events into the story, driven by external APIs (like NASA) or random seeds, to create unexpected twists.
*   **Hardware Integration**: An Arduino-controlled LED strip on the server can reflect the story's mood.

---

## Architecture

*   **Server (PC1)**: The central hub.
    *   Runs FastAPI backend (default `0.0.0.0:8000`) and Gradio UI (default `localhost:7860`).
    *   Manages client registration (including client IP and port), serves character traits, stores training data.
    *   SQLite database stores all persistent data. Paths are configurable via environment variables (see Setup).
*   **Clients (PC2, PC3, ...)**: Character nodes.
    *   Each client runs `CharacterClient/main.py`.
    *   Runs its own FastAPI server on a configurable host and port (defaults to `0.0.0.0:8001`).
    *   Registers with the central server, providing its `pc_id`, `token`, and listening `client_port`.
    *   Receives narration, generates responses using its local LLM & TTS, and sends audio data back to the server.

---

## Project Structure

The project root (e.g., `DreamWeaver/`) is the base for default data and checkpoint paths.

```
DreamWeaver/
├── SERVER/
│   ├── src/
│   │   ├── config.py         # Centralized server path and settings
│   │   └── ...               # Other server modules
│   ├── requirements.txt      # Server dependencies
│   └── main.py               # Server entry point
│
├── CharacterClient/
│   ├── src/
│   │   ├── character_client.py # Client FastAPI app and logic
│   │   ├── llm_engine.py     # Client LLM (placeholder)
│   │   ├── tts_manager.py    # Client TTS (placeholder)
│   │   └── requirements.txt  # Client dependencies
│   ├── main.py               # Client entry point (argparse for config)
│   └── README.md             # Client-specific README (TODO)
│
├── data/                     # Default: [ProjectRoot]/data/
│   ├── dream_weaver.db       # Default: [data]/dream_weaver.db
│   └── ...                   # Other data subdirectories (audio, models)
│
├── checkpoints/              # Default: [ProjectRoot]/checkpoints/
│
└── README.md                 # This file
```

---

## Setup and Installation

### Prerequisites
*   Python 3.9+
*   Git
*   NVIDIA GPU with CUDA recommended for server and clients running LLMs/TTS.

### 1. Server Setup (PC1)

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/your_username/DreamWeaver.git # Replace with your repo URL
    cd DreamWeaver
    ```
2.  **Install Server Dependencies:**
    ```bash
    cd SERVER
    pip install -r requirements.txt
    cd ..
    ```
3.  **Configure Server Environment Variables (Recommended):**
    Create `.env` in `DreamWeaver/SERVER/` or set system-wide.
    *   `DREAMWEAVER_DATA_PATH`: Base for data, db, audio, models. (Default: `[ProjectRoot]/data/`)
    *   `DB_PATH`: Path to `dream_weaver.db`. (Default: `[DREAMWEAVER_DATA_PATH]/dream_weaver.db`)
    *   `DREAMWEAVER_MODEL_PATH`: For LLM/TTS models. (Default: `[DREAMWEAVER_DATA_PATH]/models/`)
    *   `DREAMWEAVER_CHECKPOINT_PATH`: For checkpoints. (Default: `[ProjectRoot]/checkpoints/`)
    *   `ARDUINO_SERIAL_PORT`: E.g., `COM3` or `/dev/ttyUSB0`.
    *   `NASA_API_KEY`: Default: `DEMO_KEY`.

4.  **Launch Server:**
    From `DreamWeaver/` root:
    ```bash
    python SERVER/main.py
    ```
    *   Gradio UI: `http://localhost:7860`
    *   FastAPI Server: `http://0.0.0.0:8000`
    *   Dashboard: `http://localhost:8000/dashboard`

### 2. Client Setup (PC2, PC3, ...)

1.  **Copy `CharacterClient` Directory:** Transfer the `CharacterClient` folder to each client machine.
2.  **Install Client Dependencies:**
    On each client machine, in its `CharacterClient` directory:
    ```bash
    pip install -r src/requirements.txt
    ```
3.  **Configure and Launch Client:**
    Clients are configured via command-line arguments or environment variables. Command-line arguments take precedence.
    From the `CharacterClient` directory on the client machine:
    ```bash
    python main.py --pc_id <YourClientID> --token <YourClientToken> --server_url <ServerFastAPI_URL> [--client_host <IP_Client_Binds_To>] [--client_port <Port_Client_Listens_On>]
    ```
    **Required Arguments:**
    *   `--pc_id <YourClientID>`: Unique ID for this client (e.g., `PC2`). Must match the ID used on the server's Gradio UI to create the character and generate the token. (Env: `CLIENT_PC_ID`)
    *   `--token <YourClientToken>`: Access token obtained from the server's Gradio UI for this `pc_id`. (Env: `CLIENT_TOKEN`)
    *   `--server_url <ServerFastAPI_URL>`: Full URL of the central server's FastAPI. (E.g., `http://<server_ip>:8000`). (Env: `SERVER_URL`, Default: `http://127.0.0.1:8000`)

    **Optional Arguments:**
    *   `--client_host <IP_Client_Binds_To>`: The IP address the client's own API server will bind to. (Env: `CLIENT_HOST`, Default: `0.0.0.0`)
    *   `--client_port <Port_Client_Listens_On>`: The port the client's API server will listen on. This port is registered with the central server. **Ensure this is unique per client if multiple clients run on the same machine, and different from the main server's port (8000).** (Env: `CLIENT_PORT`, Default: `8001`)
    *   `--client_model_path <Path>`: (Placeholder for future use) Path for client-specific models. (Env: `DREAMWEAVER_CLIENT_MODEL_PATH`)

    **Example:**
    ```bash
    python main.py --pc_id PC2 --token abc123xyz --server_url http://192.168.1.100:8000 --client_port 8002
    ```
    The client will register its `pc_id`, IP address, and `client_port` (e.g., 8002) with the server. The server will then use `http://<client_ip>:8002/character` to communicate with this specific client.

---

## Usage Workflow

1.  **Start Server:** `python SERVER/main.py` on PC1.
2.  **Create Characters (Gradio UI):** Access `http://<server_ip>:7860`. Create characters, assign them to `pc_id`s (e.g., `PC1`, `PC2`), and note the generated tokens for clients.
3.  **Start Clients:** On each client machine, run `python main.py` from its `CharacterClient` directory with the appropriate arguments (see Client Setup).
4.  **Monitor Hive (Dashboard):** `http://<server_ip>:8000/dashboard` to check client statuses (IP, port, online/offline).
5.  **Narrate Story (Gradio UI):** Use the "Story Progression" tab. Server sends requests to clients at their registered IP and port.

---

## Contributing

Fork, branch, commit, push, PR. Contributions welcome!

---

## License

MIT License. (Include LICENSE file if applicable).
