# DreamWeaver - Server

This directory contains the server application for the DreamWeaver storytelling network. The server acts as the central hub, managing story progression, character interactions, and client connections.

## Features

*   **Centralized Story Management**: Orchestrates the narrative flow and interactions between the narrator and AI characters.
*   **Client Coordination**: Manages registration, health checks, and communication with `CharacterClient` instances.
*   **Gradio Interface**: Provides a web-based UI for:
    *   Narrating the story (speech-to-text input).
    *   Creating and configuring characters (personality, voice, LLM, client assignment).
    *   Generating access tokens for clients.
    *   Viewing story history.
    *   Managing story checkpoints.
    *   Exporting the story.
*   **FastAPI Backend**: Exposes API endpoints for client communication and dashboard functionality.
*   **Dashboard**: Offers a real-time view of connected clients and their statuses.
*   **Database Integration**: Uses SQLite (`dream_weaver.db`) to store persistent data like story content, character configurations, and client information.
*   **Flexible TTS Engine**: Supports multiple Text-to-Speech services for the narrator and server-generated audio.
*   **Asynchronous Operations**: Handles long-running tasks asynchronously to maintain UI responsiveness.

## Directory Structure

*   **`src/`**: Contains the core server source code.
    *   **`config.py`**: Manages server-specific paths for data, models, checkpoints, and database. Paths are configurable via environment variables.
    *   **`database.py`**: Handles database schema creation and operations.
    *   **`client_manager.py`**: Manages client connections, status updates, and health checks.
    *   **`csm.py`** (Central State Manager): Likely manages the overall state of the story and interactions.
    *   **`character_server.py`**: Contains the main server logic, possibly including the FastAPI application setup.
    *   **`gradio_interface.py`**: Implements the Gradio web UI.
    *   **`server_api.py`**: Defines FastAPI endpoints for client-server communication.
    *   **`llm_engine.py`**: Manages LLM functionalities for the server (e.g., for narrator's AI assistant if any, or processing text).
    *   **`tts_manager.py`**: Manages TTS functionalities for the server.
    *   **`checkpoint_manager.py`**: Handles saving and loading of story checkpoints.
    *   **`narrator.py`**: Logic related to the narrator's input and role.
    *   **`chaos_engine.py`**: (Purpose to be inferred - might introduce dynamic events or challenges).
    *   **`dashboard.py`**: Code for the server dashboard.
    *   **`env_manager.py`**: Manages environment variable loading and access.
    *   **`hardware.py`**: (Purpose to be inferred - might involve hardware monitoring or specific hardware acceleration).
*   **`main.py`**: Entry point for launching the server application.
*   **`requirements.txt`**: Python dependencies for the server.
*   **`data/`** (Default location, path configurable via `DREAMWEAVER_DATA_PATH` env var):
    *   **`dream_weaver.db`**: SQLite database file.
    *   **`audio/`**: Storage for server-side audio (e.g., narrator recordings, character responses).
    *   **`models/`**: Storage for server-side models (e.g., STT models, narrator TTS models).
*   **`logs/`**: Contains server-side log files (e.g., `server.log`). Useful for debugging and monitoring server activity. (Created automatically within the `SERVER` directory).
*   **`checkpoints/`** (Default location, path configurable via `DREAMWEAVER_CHECKPOINT_PATH` env var): Storage for story checkpoints.
*   **`tests/`**: Contains tests for the server components.

## Logging

The server now implements more structured logging:
*   Logs are output to both the console and a rotating file located at `SERVER/logs/server.log`.
*   Log entries include timestamp, logger name, level, module, and message.
*   This provides better insights into server operations and aids in troubleshooting.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   Git (if cloning from the DreamWeaver repository)
    *   An NVIDIA GPU with CUDA is recommended for optimal performance if using local LLMs/TTS on the server.

2.  **Clone the Repository (if not already done):**
    ```bash
    git clone <your_repo_url>
    cd DreamWeaver/SERVER
    ```

3.  **Install Dependencies:**
    Navigate to the `SERVER` directory and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    You may need to install system dependencies for some functionalities (e.g., CUDA toolkits for GPU support).

4.  **Configuration (Environment Variables):**
    It's recommended to configure the server using environment variables. You can create a `.env` file in the `DreamWeaver/SERVER/` directory or set them system-wide.
    *   `DREAMWEAVER_DATA_PATH`: Base path for all server-related data files.
        *   Default: `[ProjectRoot]/data/` (i.e., `DreamWeaver/data/`)
    *   `DB_PATH`: Full path to the `dream_weaver.db` SQLite database file.
        *   Default: `[DREAMWEAVER_DATA_PATH]/dream_weaver.db`
    *   `DREAMWEAVER_MODEL_PATH`: Path for storing server-specific models (LLM, TTS, STT).
        *   Default: `[DREAMWEAVER_DATA_PATH]/models/`
    *   `DREAMWEAVER_CHECKPOINT_PATH`: Path for storing story checkpoints.
        *   Default: `[ProjectRoot]/checkpoints/`
    *   `GRADIO_SERVER_NAME`: Host for the Gradio UI.
        *   Default: `127.0.0.1` (localhost)
    *   `GRADIO_SERVER_PORT`: Port for the Gradio UI.
        *   Default: `7860`
    *   `API_SERVER_HOST`: Host for the FastAPI backend.
        *   Default: `0.0.0.0` (accessible from network)
    *   `API_SERVER_PORT`: Port for the FastAPI backend.
        *   Default: `8000`

5.  **Launch the Server:**
    From the root directory of the DreamWeaver project (`DreamWeaver/`):
    ```bash
    python SERVER/main.py
    ```
    Or, if you are already in the `DreamWeaver/SERVER/` directory:
    ```bash
    python main.py
    ```

    *   The Gradio UI should then be accessible at `http://<GRADIO_SERVER_NAME>:<GRADIO_SERVER_PORT>` (e.g., `http://127.0.0.1:7860`).
    *   The FastAPI backend (and dashboard) will be available at `http://<API_SERVER_HOST>:<API_SERVER_PORT>` (e.g., `http://0.0.0.0:8000`, with the dashboard likely at `/dashboard`).

## Operation

Once launched, the server will:
1.  Initialize its database if it doesn't exist.
2.  Start the FastAPI backend to listen for client registrations and API calls.
3.  Launch the Gradio web interface for narrator interaction and system management.
4.  Be ready to accept connections from `CharacterClient` instances.
5.  Manage the story progression based on narrator input and character responses.

Refer to the main project README for details on the overall DreamWeaver system architecture and `CharacterClient` setup.
