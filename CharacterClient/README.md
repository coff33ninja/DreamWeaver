# DreamWeaver - Character Client

This directory contains the `CharacterClient` application, which allows an AI-driven character to participate in the DreamWeaver storytelling network. Each client instance runs independently, managing its own Large Language Model (LLM) and Text-to-Speech (TTS) engine.

## Features

*   Connects to a central DreamWeaver server to receive character traits and story narration.
*   Generates character responses using a local LLM.
*   Synthesizes speech for its responses using a local TTS engine.
*   Sends generated audio back to the server for playback in the main story.
*   Manages its own model files (LLM base models, LoRA adapters, TTS models, reference voices).
*   Stores temporary data and (optionally) logs locally.
*   Registers with the server, providing its listening IP and port for communication.
*   Sends periodic heartbeats to the server to maintain "online" status.
*   Exposes a `/health` endpoint for server-side API responsiveness checks.

## Directory Structure

*   **`src/`**: Contains the core client source code.
    *   **`config.py`**: Manages client-specific paths for data, models, logs, and temporary files. Paths are configurable via environment variables.
    *   **`character_client.py`**: Main client logic, including the `CharacterClient` class and the FastAPI application that receives requests from the server.
    *   **`llm_engine.py`**: Manages the client's local LLM, including model loading (with quantization), adapter management (loading, placeholder for saving), and text generation.
    *   **`tts_manager.py`**: Manages the client's local TTS engine, supporting services like Piper, Coqui XTTSv2, and gTTS. Handles model downloading and speech synthesis.
    *   **`requirements.txt`**: Python dependencies for the client.
*   **`main.py`**: Entry point for launching the client. Handles command-line argument parsing for configuration.
*   **`data/`** (Default location, created automatically if it doesn't exist. Path configurable via `DREAMWEAVER_CLIENT_DATA_PATH` env var):
    *   **`models/`**: Root directory for all models used by this client. (Path configurable via `DREAMWEAVER_CLIENT_MODELS_PATH` env var).
        *   `llm/base_models/`: Cached base LLM models.
        *   `llm/adapters/[Actor_id]/[model_name]/`: LoRA adapters for fine-tuned LLMs.
        *   `llm/training_data_local/[Actor_id]/`: Locally saved training data samples (JSON).
        *   `tts/piper/[model_name]/`: Piper TTS voice models.
        *   `tts/reference_voices/`: Downloaded reference audio for XTTSv2, prefixed with `Actor_id`.
        *   (Coqui TTS models are also stored under `models/tts/` or a similar structure, guided by the `TTS_HOME` env var set by `tts_manager.py`).
    *   **`logs/`**: Contains client-specific log files (e.g., `client.log`). This directory is created automatically if it doesn't exist, under the `CLIENT_DATA_PATH`. The default location is `CharacterClient/data/logs/client.log`.
    *   **`temp_audio/`**: Temporary audio files synthesized by the client before being sent to the server.

## Logging

The client now implements structured logging:
*   Logs are output to both the console and a rotating file located at `[CLIENT_DATA_PATH]/logs/client.log` (default: `CharacterClient/data/logs/client.log`).
*   Log entries include timestamp, logger name, level, module, and message.
*   This helps in monitoring client activity and diagnosing issues.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   Git (if cloning from a repository)
    *   An NVIDIA GPU with CUDA is highly recommended for running local LLMs and TTS efficiently. CPU fallback is possible but will be very slow.

2.  **Obtain Client Files:**
    Copy the entire `CharacterClient` directory to the machine where this client instance will run.

3.  **Install Dependencies:**
    Navigate to the `CharacterClient` directory and install the required Python packages:
    ```bash
    pip install -r src/requirements.txt
    ```
    You may need to install system dependencies for some TTS engines (e.g., `espeak` for Piper on some systems, or CUDA toolkits for GPU support).

4.  **Configuration & Launch:**
    The client is configured and launched using command-line arguments or by setting environment variables. The necessary `CLIENT_Actor_ID`, `CLIENT_TOKEN`, and `SERVER_URL` can be easily obtained by downloading a client configuration file (`.env` format) from the server's Gradio UI (Character Management tab). You can save this downloaded file as `.env` in the `CharacterClient` root directory, and the client will automatically load these settings if not overridden by command-line arguments.

    **Command-Line Usage (Overrides .env file):**
    ```bash
    python main.py --Actor_id <YourClientID> --token <YourClientToken> --server_url <ServerFastAPI_URL> [options]
    ```

    **Environment Variables (Loaded from `.env` file or set manually):**
    *   `CLIENT_Actor_ID`: (Required) Your unique client ID.
    *   `CLIENT_TOKEN`: (Required) Your access token.
    *   `SERVER_URL`: (Required) The server's FastAPI URL.
    *   `CLIENT_HOST`: (Optional) Host for the client's API server (Default: `0.0.0.0`).
    *   `CLIENT_PORT`: (Optional) Port for the client's API server (Default: `8001`).
    *   `DREAMWEAVER_CLIENT_DATA_PATH`: (Optional) Root path for client data.
    *   `DREAMWEAVER_CLIENT_MODELS_PATH`: (Optional) Path for client models.


    **Required Arguments (if not using .env or environment variables):**
    *   `--Actor_id <YourClientID>`: A unique identifier for this client instance (e.g., `Actor2`, `MyCharacterClient`). This ID must match the `Actor_id` used when creating the character and generating its token on the central DreamWeaver server's Gradio UI.
        *   Environment Variable: `CLIENT_Actor_ID`
    *   `--token <YourClientToken>`: The access token generated by the DreamWeaver server for this specific `Actor_id`.
        *   Environment Variable: `CLIENT_TOKEN`
    *   `--server_url <ServerFastAPI_URL>`: The full URL of the central DreamWeaver server's FastAPI backend (e.g., `http://192.168.1.100:8000`).
        *   Environment Variable: `SERVER_URL` (Default: `http://127.0.0.1:8000`)

    **Optional Arguments:**
    *   `--client_host <IP_Address>`: The IP address that this client's own API server will bind to. Use `0.0.0.0` to be accessible from other machines on the network, or `127.0.0.1` for local access only.
        *   Environment Variable: `CLIENT_HOST` (Default: `0.0.0.0`)
    *   `--client_port <PortNumber>`: The port that this client's API server will listen on. This port is registered with the central server so it knows how to contact this client. **Ensure this port is unique if running multiple clients on the same machine, and different from the main server's port (default 8000).**
        *   Environment Variable: `CLIENT_PORT` (Default: `8001`)

    **Optional Environment Variables for Data Paths:**
    These allow you to customize where the client stores its data, models, and logs. If not set, defaults relative to the `CharacterClient` directory will be used (typically creating a `CharacterClient/data/` subdirectory).
    *   `DREAMWEAVER_CLIENT_DATA_PATH`: Overrides the root path for all client data (default: `CharacterClient/data/`).
    *   `DREAMWEAVER_CLIENT_MODELS_PATH`: Overrides the path for all models (default: `[CLIENT_DATA_PATH]/models/`).

    **Example Launch Command:**
    ```bash
    python main.py --Actor_id Actor_Elara --token mysecretelaratoken --server_url http://dreamserver.local:8000 --client_port 8002
    ```

5.  **Operation:**
    Once launched, the client will:
    1.  Attempt to register with the central server using its `Actor_id`, `token`, and `client_port`.
    2.  Fetch its character traits (personality, voice settings, LLM model if specified) from the server.
    3.  Initialize its local LLM and TTS engines. This may involve downloading models if they are not already present in its local model cache (`CharacterClient/data/models/`).
    4.  Start sending periodic heartbeats to the server.
    5.  Listen for requests on its `/character` endpoint from the server.
    6.  Expose a `/health` endpoint for the server to check its API responsiveness.

## Development Notes

*   **LLM Fine-Tuning:** The `LLMEngine.fine_tune()` method is currently a placeholder. It saves training data locally but does not yet implement the actual model fine-tuning process.
*   **Logging:** Comprehensive logging to files in `CharacterClient/data/logs/` is planned but not fully implemented. Current logging is primarily to the console.
*   **Error Handling:** Basic error handling and retry mechanisms for server communication are in place. Further enhancements can be made for robustness.

This README provides a guide to setting up and running an individual Character Client. Refer to the main project README for details on the DreamWeaver server and overall system architecture.
