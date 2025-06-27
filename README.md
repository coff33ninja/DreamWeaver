# DreamWeaver - A Decentralized, AI-Powered Storytelling Network

Dream Weaver is a scalable, decentralized storytelling network where a human narrator guides a story populated by a hive of AI-driven characters. Each character runs on a separate machine (or process), possesses a unique personality and voice, and evolves over time through local fine-tuning of its own Large Language Model (LLM).

The system is managed from a central server that hosts the narrator's interface, synchronizes the story state, and provides a real-time dashboard to monitor the health and status of the character hive.

---

## Key Features

*   **Decentralized Character Hive**: Characters run as independent `CharacterClient` instances, each with its own LLM, TTS engine, and configurable API endpoint. Clients manage their own model files and temporary data.
*   **Dynamic Character Creation & Evolution**: Add or remove characters on the fly via the Gradio UI. Characters can (eventually) evolve as their local LLMs are fine-tuned with new dialogue (current client fine-tuning is placeholder). Training data is backed up to the central server.
*   **Robust Client Discovery & Communication**: Clients register their listening port with the server and maintain their connection via a periodic heartbeat. The server performs health checks on clients and communicates with responsive ones on their specified ports.
*   **Flexible TTS Engine**: Both server and clients support multiple Text-to-Speech services, including Piper and Coqui's XTTS-v2 (for high-quality voice cloning from a reference audio file), and gTTS.
*   **Interactive Gradio Interface**: A central web UI for:
    *   Narrating the story via microphone (using Whisper STT).
    *   Creating and configuring characters (name, personality, voice, LLM model, PC assignment).
    *   Generating secure access tokens for clients.
    *   Reviewing the complete story history.
    *   Managing story checkpoints.
    *   Exporting the story to JSON or TXT.
*   **Enhanced Client Status Monitoring**: The dashboard displays detailed client statuses: `Registered`, `Offline`, `Online_Heartbeat` (heartbeating but API not confirmed), `Online_Responsive` (fully ready), `Error_API` (API issues), `Error_Unreachable`, `Deactivated`.
*   **Resilient Communication**: Basic retry mechanisms implemented for server-to-client and client-to-server (registration, heartbeat) communications.

---

## Architecture

*   **Server (PC1)**: The central hub.
    *   Runs FastAPI backend (default `0.0.0.0:8000`) and Gradio UI (default `localhost:7860`).
    *   Manages client registration (IP, port, status), serves character traits, stores training data.
    *   SQLite database (`dream_weaver.db`) for persistent data. Server paths configurable via environment variables (see Server Setup).
*   **Clients (PC2, PC3, ...)**: Character nodes.
    *   Each client runs `CharacterClient/main.py`.
    *   Manages its own data (models, logs, temp files) within `CharacterClient/data/` by default (configurable via `DREAMWEAVER_CLIENT_DATA_PATH`).
    *   Runs its own FastAPI server on a configurable host/port (defaults to `0.0.0.0:8001`).
    *   Registers with the central server, providing its `pc_id`, `token`, and listening `client_port`.
    *   Receives narration, generates responses using its local LLM & TTS, and sends audio data back to the server.

---

## Project Structure

```
DreamWeaver/
├── SERVER/
│   ├── src/
│   │   ├── config.py         # Server path/settings config
│   │   ├── database.py       # Handles DB schema & operations
│   │   ├── client_manager.py # Manages client interactions, health checks
│   │   ├── csm.py            # Central State Manager for story flow
│   │   └── ...               # Other server modules (FastAPI, Gradio, LLM, TTS)
│   ├── requirements.txt      # Server dependencies
│   └── main.py               # Server entry point
│
├── CharacterClient/
│   ├── src/
│   │   ├── config.py         # Client path/settings config
│   │   ├── character_client.py # Client FastAPI app, CharacterClient class
│   │   ├── llm_engine.py     # Client LLM engine
│   │   ├── tts_manager.py    # Client TTS manager
│   │   └── requirements.txt  # Client dependencies
│   ├── main.py               # Client entry point (argparse for config)
│   └── data/                   # Default root for client-specific data (created automatically)
│       ├── models/             # For LLMs, TTS models, adapters
│       │   ├── llm/
│       │   │   ├── base_models/
│       │   │   └── adapters/
│       │   └── tts/
│       │       ├── piper/
│       │       └── reference_voices/
│       ├── logs/               # Placeholder for client logs
│       └── temp_audio/         # Temporary audio files synthesized by client
│
├── data/                     # Default for SERVER data (if not overridden by env vars)
│   ├── dream_weaver.db       # Server's SQLite database
│   ├── audio/                # Server-side audio storage
│   └── models/               # Server-side model storage
│
├── checkpoints/              # Default for SERVER checkpoints
│
└── README.md                 # This file
```

### Client-Side Data Storage (`CharacterClient/data/`)
*   **`models/llm/base_models/`**: Cached base LLM models downloaded by Hugging Face Transformers.
*   **`models/llm/adapters/[pc_id]/[model_name]/`**: LoRA adapters for client-specific fine-tuned LLMs.
*   **`models/llm/training_data_local/[pc_id]/`**: JSON samples of training data (input/output pairs) saved by the client's LLM `fine_tune` placeholder.
*   **`models/tts/piper/[model_name]/`**: Piper TTS voice model files.
*   **`models/tts/reference_voices/`**: Downloaded reference audio files for XTTSv2 voice cloning, prefixed with `pc_id`.
*   **Coqui TTS Models**: Stored within `CLIENT_TTS_MODELS_PATH` (usually `CharacterClient/data/models/tts/`) as per Coqui's caching, guided by the `TTS_HOME` environment variable set by the client's `TTSManager`.
*   **`logs/`**: Intended for client-specific log files (logging not yet fully implemented).
*   **`temp_audio/`**: Temporary WAV files generated by the client's TTS before being sent to the server.

---

## Setup and Installation

### Prerequisites
*   Python 3.9+
*   Git
*   NVIDIA GPU with CUDA recommended for server and clients for optimal LLM/TTS performance.

### 1. Server Setup (PC1)

1.  **Clone Repository:** `git clone <your_repo_url> && cd DreamWeaver`
2.  **Install Server Dependencies:** `cd SERVER && pip install -r requirements.txt && cd ..`
3.  **Configure Server Environment Variables (Recommended):**
    Create `.env` in `DreamWeaver/SERVER/` or set system-wide.
    *   `DREAMWEAVER_DATA_PATH`: Base for server data files. (Default: `[ProjectRoot]/data/`)
    *   `DB_PATH`: Path to `dream_weaver.db`. (Default: `[DREAMWEAVER_DATA_PATH]/dream_weaver.db`)
    *   `DREAMWEAVER_MODEL_PATH`: For server's LLM/TTS models. (Default: `[DREAMWEAVER_DATA_PATH]/models/`)
    *   `DREAMWEAVER_CHECKPOINT_PATH`: For story checkpoints. (Default: `[ProjectRoot]/checkpoints/`)
4.  **Launch Server:** From `DreamWeaver/` root: `python SERVER/main.py`
    *   Gradio UI: `http://localhost:7860`
    *   FastAPI & Dashboard: `http://0.0.0.0:8000` (Dashboard at `/dashboard`)

### 2. Client Setup (PC2, PC3, ...)

1.  **Copy `CharacterClient` Directory:** To each client machine.
2.  **Install Client Dependencies:** On each client, in its `CharacterClient` dir: `pip install -r src/requirements.txt`
3.  **Configure Client Data Paths (Optional Environment Variables):**
    Before launching a client, you can optionally set these to change where it stores its data:
    *   `DREAMWEAVER_CLIENT_DATA_PATH`: Overrides the default `CharacterClient/data/` for all client data.
    *   `DREAMWEAVER_CLIENT_MODELS_PATH`: Overrides `[CLIENT_DATA_PATH]/models/` for model storage.
4.  **Launch Client (Command-Line):**
    From the `CharacterClient` directory on the client machine:
    ```bash
    python main.py --pc_id <YourClientID> --token <ClientToken> --server_url <ServerURL> [options]
    ```
    **Required Arguments:**
    *   `--pc_id <YourClientID>`: E.g., `PC2`. Must match ID from Gradio UI. (Env: `CLIENT_PC_ID`)
    *   `--token <ClientToken>`: Token from Gradio UI for this `pc_id`. (Env: `CLIENT_TOKEN`)
    *   `--server_url <ServerURL>`: Server's FastAPI URL (e.g., `http://<server_ip>:8000`). (Env: `SERVER_URL`, Default: `http://127.0.0.1:8000`)

    **Optional Arguments:**
    *   `--client_host <IP>`: Client API binds to this IP. (Env: `CLIENT_HOST`, Default: `0.0.0.0`)
    *   `--client_port <Port>`: Client API listens on this port. Registered with server. **Must be unique if multiple clients on one host.** (Env: `CLIENT_PORT`, Default: `8001`)

    **Example:**
    `python main.py --pc_id PC2 --token xyz789 --server_url http://192.168.1.100:8000 --client_port 8002`

---

## Client Statuses on Dashboard

The server dashboard now displays more detailed client statuses:
*   **Registered**: Token created, client has not yet connected/registered its IP and port.
*   **Offline**: Client was previously connected but is no longer sending heartbeats or is unreachable.
*   **Online_Heartbeat**: Client is sending heartbeats, but its API responsiveness hasn't been confirmed by a direct health check from the server yet.
*   **Online_Responsive**: Client is heartbeating, and its `/health` endpoint is responsive to the server. Ready for story interaction.
*   **Error_API**: Client is heartbeating, but its `/health` endpoint failed or returned an error/degraded status, or an error occurred during character interaction.
*   **Error_API_Degraded**: Client's `/health` endpoint responded but indicated one of its subsystems (LLM/TTS) is not ready.
*   **Error_Unreachable**: Server cannot connect to the client's IP/Port (e.g., for health check or character interaction).
*   **Deactivated**: Client has been manually marked as deactivated by an administrator (feature not yet implemented, but status exists).

---
## Usage Workflow (Brief)

1.  Start Server.
2.  Use Gradio UI to create characters for `PC1` (server) and any clients (e.g., `PC2`, `PC3`). Note client tokens.
3.  Start each `CharacterClient` with its `pc_id`, `token`, and `server_url`. Ensure unique `client_port` if on same host.
4.  Monitor client statuses on the server's Dashboard. Look for `Online_Responsive`.
5.  Narrate story via Gradio. Server communicates with `Online_Responsive` clients.

---
(Contributing and License sections remain the same)
