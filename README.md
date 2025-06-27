# DreamWeaver - A Decentralized, AI-Powered Storytelling Network

Dream Weaver is a scalable, decentralized storytelling network where a human narrator guides a story populated by a hive of AI-driven characters. Each character runs on a separate machine, possesses a unique personality and voice, and evolves over time through local fine-tuning of its own Large Language Model (LLM).

The system is managed from a central server that hosts the narrator's interface, synchronizes the story state, and provides a real-time dashboard to monitor the health of the character hive.

---

## Key Features

*   **Decentralized Character Hive**: Characters run as independent clients on separate machines (PC2, PC3, ...), each with its own LLM (e.g., TinyLLaMA) and TTS engine.
*   **Dynamic Character Creation & Evolution**: Add or remove characters on the fly via the Gradio UI. Characters evolve as their local LLMs are fine-tuned with new dialogue, and the training data is backed up to the central server.
*   **Robust Client Discovery**: Clients register with the server and maintain their connection via a periodic heartbeat, ensuring the server always knows which characters are active.
*   **Flexible TTS Engine**: Supports multiple Text-to-Speech services, including Piper, Google TTS, and Coqui's XTTS-v2 for high-quality voice cloning from a reference audio file.
*   **Interactive Gradio Interface**: A central web UI for:
    *   Narrating the story via microphone (using Whisper STT).
    *   Creating and configuring characters.
    *   Generating secure access tokens for clients.
    *   Reviewing the complete story history in a chatbot format.
    *   Managing story checkpoints.
    *   Exporting the story to JSON or TXT.
*   **Story Checkpoints & Export**: Save the entire story state (database and server-side model adapters) at any point. Load previous checkpoints to explore different narrative branches.
*   **Real-time Monitoring Dashboard**: A web-based dashboard to monitor server CPU/memory and the online/offline status of all connected clients.
*   **Chaos Mode**: Injects random events into the story, driven by external APIs (like NASA) or random seeds, to create unexpected twists.
*   **Hardware Integration**: An Arduino-controlled LED strip on the server can reflect the story's mood, with optional support for character-specific LEDs on client machines.

---

## Architecture

The project follows a client-server model:

*   **Server (PC1)**: The central hub.
    *   **FastAPI Backend**: Manages client registration, serves character traits, stores training data, and provides API endpoints for the dashboard and reference audio.
    *   **Gradio UI**: The primary user interface for the narrator and system administrator.
    *   **SQLite Database**: A single `dream_weaver.db` file stores all story dialogue, character traits, client tokens, and training data.
    *   **Narrator Module**: Uses Whisper STT for transcription.
    *   **PC1 Character**: Can optionally run a character directly on the server.

*   **Clients (PC2, PC3, ...)**: The character nodes.
    *   Each client runs the `character_client.py` script.
    *   Connects to the server using a unique token to receive its personality and configuration.
    *   Runs its own LLM for dialogue generation and a TTS engine for speech synthesis.
    *   Sends generated audio back to the server for playback, creating a unified audio experience.
    *   Periodically sends a heartbeat to the server to stay "online".

---

## Project Structure

```
SERVER/
├── src/
│   ├── __init__.py
│   ├── csm.py              # Central State Manager
│   ├── narrator.py         # STT for narration
│   ├── character_server.py # Manages the character on PC1
│   ├── client_manager.py   # Handles client communication
│   ├── llm_engine.py       # LLM management and fine-tuning (server-side)
│   ├── tts_manager.py      # TTS selection and synthesis
│   ├── gradio_interface.py # Gradio backend/frontend
│   ├── server_api.py       # FastAPI endpoints for clients
│   ├── dashboard.py        # FastAPI endpoints and HTML for the dashboard
│   ├── checkpoint_manager.py # Logic for saving/loading story state
│   ├── chaos_engine.py     # Random events and glitches
│   ├── hardware.py         # Arduino LED strip control
│   └── database.py         # SQLite management
├── data/
│   ├── dream_weaver.db     # Main SQLite database
│   ├── audio/              # Stores all generated audio
│   │   ├── narrator/
│   │   ├── characters/
│   │   └── reference_voices/ # Stores uploaded voice cloning samples
│   └── models/             # LLM and TTS model weights
│       └── adapters/       # Saved LoRA adapters for fine-tuned models
├── checkpoints/            # Saved story checkpoints
├── requirements.txt        # Server dependencies
└── main.py                 # Server entry point

CharacterClient/            # Client directory
├── src/
│   ├── character_client.py # The script for all character clients
│   ├── llm_engine.py       # LLM management (client-side)
│   ├── tts_manager.py      # TTS management (client-side)
│   └── requirements.txt    # Client dependencies
├── main.py                 # Client entry point
└── README.md
```

---

## Setup and Installation

### Prerequisites
*   Python 3.9+
*   Git
*   An NVIDIA GPU with CUDA is highly recommended for running the LLMs and TTS engines efficiently.

### 1. Server Setup (PC1)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/coff33ninja/DreamWeaver
    cd DreamWeaver
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables (Optional):**
    You can set these system-wide or create a `.env` file (and use a library like `python-dotenv` to load it).
    *   `DB_PATH`: Path to the SQLite database file. (Default: `E:/DreamWeaver/data/dream_weaver.db`)
    *   `ARDUINO_SERIAL_PORT`: COM port for the Arduino. (Default: `COM3`)
    *   `NASA_API_KEY`: Your API key for the NASA API for Chaos Mode. (Default: `DEMO_KEY`)
    *   `DREAMWEAVER_MODEL_PATH`: Base path for storing downloaded TTS models. (Default: `E:/DreamWeaver/data/models`)

4.  **Launch the Server:**
    ```bash
    python main.py
    ```
    *   The Gradio UI will be available at `http://localhost:7860`.
    *   The FastAPI server will run on `http://0.0.0.0:8000`.
    *   The Monitoring Dashboard will be at `http://localhost:8000/dashboard`.

### 2. Client Setup (PC2, PC3, ...)

1.  **Copy the `client` directory** to each machine that will run a character.

2.  **Install dependencies:**
    ```bash
    cd client
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    *   `SERVER_URL`: The full URL of the server's FastAPI instance (e.g., `http://192.168.1.101:8000`).
    *   `CLIENT_PC_ID`: The unique ID for this client (e.g., `PC2`, `PC3`). This **must** match the PC ID assigned in the Gradio UI.

4.  **Get an Access Token:**
    *   On the server's Gradio UI, go to the "Character Creation" section.
    *   Create a new character, assigning it to the correct Client PC ID (e.g., `PC2`).
    *   A unique token will be generated. Copy this token.

5.  **Launch the Client:**
    Run the client script from the command line, passing the token as an argument.
    ```bash
    python character_client.py <your_character_token>
    ```
    The client will register with the server and begin listening for narration.

---

## Usage Workflow

1.  **Start the Server**: Run `python main.py` on PC1.
2.  **Create Characters**: Open the Gradio UI (`http://localhost:7860`). Create a character for the server (PC1) and characters for each client machine (PC2, PC3, etc.). For XTTS-v2, you can upload a reference voice file.
3.  **Start Clients**: On each client machine, run `python character_client.py <token>` with the corresponding token from the Gradio UI.
4.  **Monitor the Hive**: Open the dashboard (`http://localhost:8000/dashboard`) to confirm all clients are "Online".
5.  **Tell the Story**: In the Gradio UI's "Story" section, use the microphone to narrate. The server will transcribe your speech, and all active characters will generate and play back their responses in sequence.
6.  **Review and Manage**:
    *   Use the "Story Playback" tab to review the conversation history.
    *   Use the "Checkpoints" tab to save the current state of the story.
    *   Use the "Export Story" tab to save a copy of the narrative to a file.

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
