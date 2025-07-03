import os
import torch
import asyncio  # Added asyncio
from typing import Optional
import logging
import types

from .config import (
    CLIENT_TTS_MODELS_PATH,
    CLIENT_TEMP_AUDIO_PATH,
    ensure_client_directories,
)

logger = logging.getLogger("dreamweaver_client")
ensure_client_directories()  # This in config.py should also be logged if it prints

try:
    import gtts
except ImportError:
    gtts = None
    logger.info(
        "Client TTSManager: gtts library not found. gTTS service will not be available."
    )
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None
    logger.info(
        "Client TTSManager: Coqui TTS library not found. XTTSv2 service will not be available."
    )


class TTSManager:
    def __init__(
        self,
        tts_service_name: str,
        model_name: Optional[str] = None,
        speaker_wav_path: Optional[str] = None,
        language: Optional[str] = "en",
    ):
        """
        Initialize a TTSManager instance for the specified text-to-speech service and model.

        Parameters:
            tts_service_name (str): Name of the TTS backend service to use (e.g., "gtts", "xttsv2").
            model_name (Optional[str]): Identifier or path for the TTS model, if applicable.
            speaker_wav_path (Optional[str]): Path to a speaker WAV file for voice cloning, if supported.
            language (Optional[str]): Language code for synthesis (default is "en").

        Initializes environment variables and directories required for the selected TTS service and performs blocking setup.
        """
        self.service_name = tts_service_name
        self.model_name = model_name or ""
        self.speaker_wav_path = speaker_wav_path or ""
        self.language = language or "en"
        self.tts_instance = None
        self.is_initialized = False

        os.environ["TTS_HOME"] = CLIENT_TTS_MODELS_PATH  # For Coqui TTS
        try:
            os.makedirs(
                os.path.join(CLIENT_TTS_MODELS_PATH, "tts_models"), exist_ok=True
            )
        except OSError as e:
            logger.error(
                f"Client TTSManager: Could not create tts_models directory {os.path.join(CLIENT_TTS_MODELS_PATH, 'tts_models')}: {e}",
                exc_info=True,
            )
            # Depending on severity, might want to set self.is_initialized = False or raise

        logger.info(
            f"Client TTSManager: Initializing for service '{self.service_name}', model '{self.model_name}', lang '{self.language}', speaker_wav: '{self.speaker_wav_path}'"
        )
        self._initialize_service_blocking()

    def _initialize_service_blocking(self):
        """
        Synchronously initializes the selected TTS backend and loads the required model.

        Sets up the TTS instance for the specified service (`gtts` or `xttsv2`) and marks the manager as initialized if successful. Logs errors or warnings if the service or model is unavailable.
        """
        if self.service_name == "gtts":
            if gtts:
                self.tts_instance = self._gtts_synthesize_blocking
                self.is_initialized = True
                logger.info("Client TTSManager: gTTS service configured.")
            else:
                logger.error(
                    "Client TTSManager: Error - gTTS library not found. gTTS service unavailable."
                )
            return

        if not self.model_name:  # This check is more relevant for CoquiTTS
            logger.warning(
                f"Client TTSManager: No model name specified for service '{self.service_name}'. Initialization may fail or use defaults."
            )
            # Allow to proceed, specific service logic will handle if model_name is critical.

        model_path_or_name = self._get_or_download_model_blocking(
            self.service_name, self.model_name
        )
        if (
            not model_path_or_name and self.service_name == "xttsv2"
        ):  # If xttsv2 and model_name was expected but not resolved
            logger.error(
                f"Client TTSManager: Could not determine model path or name for '{self.model_name}' for service '{self.service_name}'. Cannot initialize."
            )
            return

        if self.service_name == "xttsv2":
            if CoquiTTS:
                try:
                    logger.info(
                        f"Client TTSManager: Initializing Coqui XTTSv2 with model: {model_path_or_name or self.model_name}"
                    )
                    self.tts_instance = CoquiTTS(
                        model_name=model_path_or_name or self.model_name,
                        progress_bar=True,
                    )  # Use self.model_name if model_path_or_name is None (e.g. not used by _get_or_download)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.tts_instance.to(device)
                    self.is_initialized = True
                    logger.info(
                        f"Client TTSManager: Coqui XTTSv2 initialized with {model_path_or_name or self.model_name}. Device: {device}"
                    )
                except Exception as e:
                    logger.error(
                        f"Client TTSManager: Error initializing Coqui XTTSv2 model {model_path_or_name or self.model_name}: {e}",
                        exc_info=True,
                    )
            else:
                logger.error(
                    "Client TTSManager: Error - Coqui TTS library not found. XTTSv2 service unavailable."
                )
        elif self.service_name != "gtts":  # gtts is handled above
            logger.error(
                f"Client TTSManager: Unsupported TTS service '{self.service_name}'."
            )

    def _gtts_synthesize_blocking(self, text: str, output_file_path: str, lang: str):
        """
        Synchronously synthesizes speech from text using gTTS and saves the output to a file.

        Parameters:
            text (str): The text to be converted to speech.
            output_file_path (str): The file path where the synthesized audio will be saved.
            lang (str): The language code for the speech synthesis.
        """
        if gtts and hasattr(gtts, "gTTS"):
            logger.debug(
                f"Client TTSManager (gTTS): Synthesizing text '{text[:30]}...' to {output_file_path} in lang '{lang}'"
            )
            gtts.gTTS(text=text, lang=lang).save(output_file_path)
        else:
            logger.error(
                "Client TTSManager: gTTS library not available or gTTS class missing. Cannot synthesize with gTTS."
            )

    def _xttsv2_synthesize_blocking(
        self,
        text: str,
        output_file_path: str,
        speaker_wav: Optional[str] = None,
        lang: str = "en",
    ):
        """
        Synthesize speech from text using the XTTSv2 backend and save it to a file.

        If a valid speaker WAV file is provided and exists, it is used for voice cloning; otherwise, the default voice is used. The language is set to the specified value if supported by the model, otherwise the first available language is used.

        Parameters:
            text (str): The input text to synthesize.
            output_file_path (str): The file path where the synthesized audio will be saved.
            speaker_wav (Optional[str]): Path to a speaker WAV file for voice cloning. If not provided or invalid, the default voice is used.
            lang (str): The language code for synthesis. Defaults to "en".
        """
        # Only proceed if tts_instance is a valid object (not a MethodType)
        if (
            self.tts_instance is None
            or isinstance(self.tts_instance, types.MethodType)
            or not hasattr(self.tts_instance, "tts_to_file")
            or not callable(getattr(self.tts_instance, "tts_to_file", None))
        ):
            logger.error(
                "Client TTSManager (XTTSv2): XTTSv2 instance is not available or invalid. Cannot synthesize."
            )
            return

        lang_to_use = lang or "en"
        logger.debug(
            f"Client TTSManager (XTTSv2): Preparing to synthesize text '{text[:30]}...' to {output_file_path}. Speaker: '{speaker_wav}', Lang: '{lang_to_use}'."
        )

        # Only check languages if attribute exists and is not a method
        languages = getattr(self.tts_instance, "languages", None)
        if languages and isinstance(languages, (list, tuple)):
            if lang_to_use not in languages:
                lang_to_use = languages[0]
        speaker_to_use = speaker_wav or self.speaker_wav_path
        # Only call tts_to_file if tts_instance is not a MethodType (i.e., not gTTS)
        if not isinstance(self.tts_instance, types.MethodType):
            if (
                speaker_to_use
                and isinstance(speaker_to_use, str)
                and os.path.exists(speaker_to_use)
            ):
                self.tts_instance.tts_to_file(
                    text=text,
                    speaker_wav=speaker_to_use,
                    language=lang_to_use,
                    file_path=output_file_path,
                )
                logger.debug(
                    f"Client TTSManager (XTTSv2): Synthesized with speaker wav {speaker_to_use}."
                )
            else:
                if speaker_to_use:
                    logger.warning(
                        f"Client TTSManager (XTTSv2): speaker_wav '{speaker_to_use}' not found or invalid. Using default voice for lang {lang_to_use}."
                    )
                else:
                    logger.debug(
                        f"Client TTSManager (XTTSv2): Synthesizing with default voice for lang {lang_to_use}."
                    )
                self.tts_instance.tts_to_file(
                    text=text, language=lang_to_use, file_path=output_file_path
                )
        else:
            logger.error("Client TTSManager: tts_to_file called on a method instance (gTTS). This should not happen.")
            return

    async def synthesize(
        self,
        text: str,
        output_filename_no_path: str,
        speaker_wav_for_synthesis: Optional[str] = None,
    ) -> str | None:
        """
        Asynchronously synthesizes speech from text and saves the audio to a file using the configured TTS backend.

        Parameters:
            text (str): The text to synthesize into speech.
            output_filename_no_path (str): The filename (without path) for the output audio file.
            speaker_wav_for_synthesis (Optional[str]): Optional path to a speaker WAV file for voice cloning (used by some TTS backends).

        Returns:
            str | None: The full path to the generated audio file, or None if synthesis fails or the service is not initialized.
        """
        if not self.is_initialized or not self.tts_instance:
            logger.error(
                f"Client TTSManager ({self.service_name}): Not initialized. Cannot synthesize text: '{text[:30]}...'."
            )
            return None

        # CLIENT_TEMP_AUDIO_PATH should be ensured by config.ensure_client_directories()
        # but a local check for the specific output dir doesn't hurt.
        output_full_path = os.path.join(CLIENT_TEMP_AUDIO_PATH, output_filename_no_path)
        try:
            os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        except OSError as e_mkdir:
            logger.error(
                f"Client TTSManager: Could not create directory for temporary audio file {output_full_path}: {e_mkdir}",
                exc_info=True,
            )
            return None

        logger.info(
            f"Client TTSManager ({self.service_name}): Synthesizing text '{text[:30]}...' to {output_full_path}"
        )
        try:
            if self.service_name == "gtts":
                await asyncio.to_thread(
                    self._gtts_synthesize_blocking,
                    text,
                    output_full_path,
                    self.language,
                )
            elif self.service_name == "xttsv2":
                speaker_wav = (
                    speaker_wav_for_synthesis or ""
                )  # Default to empty string if None
                await asyncio.to_thread(
                    self._xttsv2_synthesize_blocking,
                    text,
                    output_full_path,
                    speaker_wav,
                    self.language,
                )
            else:
                logger.error(
                    f"Client TTSManager: No async synthesis method for unsupported service '{self.service_name}'."
                )
                return None  # Or raise ValueError

            if (
                os.path.exists(output_full_path)
                and os.path.getsize(output_full_path) > 0
            ):
                logger.info(
                    f"Client TTSManager: Successfully synthesized audio to {output_full_path}"
                )
                return output_full_path
            else:
                logger.error(
                    f"Client TTSManager: Synthesis completed but output file {output_full_path} is missing or empty."
                )
                return None
        except Exception as e:
            logger.error(
                f"Client TTSManager: Error during async TTS synthesis with {self.service_name} for '{text[:30]}...': {e}",
                exc_info=True,
            )
            if os.path.exists(
                output_full_path
            ):  # Attempt to clean up failed/partial file
                try:
                    os.remove(output_full_path)
                    logger.info(
                        f"Client TTSManager: Removed partial/failed output file {output_full_path}"
                    )
                except OSError as e_remove:
                    logger.error(
                        f"Client TTSManager: Error removing partial/failed output file {output_full_path}: {e_remove}",
                        exc_info=True,
                    )
            return None

    def _get_or_download_model_blocking(self, service_name: str, model_identifier: str):
        """
        Return the model identifier for the specified service, creating the target directory if needed.

        For the "xttsv2" service, this method ensures the model directory exists and returns the provided model identifier. For other services, it returns None.
        """
        target_dir_base = os.path.join(CLIENT_TTS_MODELS_PATH, service_name.lower())
        try:
            os.makedirs(target_dir_base, exist_ok=True)
        except OSError as e:
            logger.error(
                f"Client TTSManager: Could not create model directory {target_dir_base} for service {service_name}: {e}",
                exc_info=True,
            )
            # This might be critical depending on the service
            if service_name == "xttsv2":  # If we can't make the dir, Coqui might fail.
                return None

        if service_name == "xttsv2":
            logger.debug(
                f"Client TTSManager: Model identifier for XTTSv2 is '{model_identifier}'. Coqui will handle download if needed to TTS_HOME={os.getenv('TTS_HOME')}"
            )
            return model_identifier
        # For gTTS or other services not requiring explicit model file download via this manager
        logger.debug(
            f"Client TTSManager: No specific model download logic for service '{service_name}' in this manager. Identifier: '{model_identifier}'"
        )
        return model_identifier  # Return identifier for other services too, they might use it directly.

    @staticmethod
    def list_services():
        """
        Return a list of available TTS service names based on installed libraries.

        Returns:
            services (list of str): Names of supported TTS services available in the current environment.
        """
        services = []
        if gtts:
            services.append("gtts")
        if CoquiTTS:
            services.append("xttsv2")
        return services

    @staticmethod
    def discover_models():
        """
        Scan the TTS models directory for available models for all supported backends.

        Returns:
            dict: {service_name: [model_names]}
        """
        models = {}
        if os.path.exists(CLIENT_TTS_MODELS_PATH):
            for service in os.listdir(CLIENT_TTS_MODELS_PATH):
                service_path = os.path.join(CLIENT_TTS_MODELS_PATH, service)
                if os.path.isdir(service_path):
                    models[service] = []
                    for model in os.listdir(service_path):
                        model_path = os.path.join(service_path, model)
                        if os.path.isdir(model_path) or model.endswith(('.pth', '.onnx', '.bin')):
                            models[service].append(model)
        # Always include default XTTSv2 if CoquiTTS is available
        if CoquiTTS:
            models.setdefault('xttsv2', []).append('tts_models/multilingual/multi-dataset/xtts_v2')
        return models

    @staticmethod
    def get_available_models(service_name: str):
        """
        Return a list of available model identifiers for the specified TTS service.

        Parameters:
            service_name (str): The name of the TTS service (e.g., "gtts", "xttsv2", "piper").

        Returns:
            list: List of model identifiers or UI hints relevant to the service.
        """
        if service_name == "gtts":
            return ["N/A (uses language codes)"]
        discovered = TTSManager.discover_models().get(service_name, [])
        # Always include default XTTSv2 if CoquiTTS is available
        if service_name == "xttsv2" and CoquiTTS:
            if "tts_models/multilingual/multi-dataset/xtts_v2" not in discovered:
                discovered.append("tts_models/multilingual/multi-dataset/xtts_v2")
        return discovered

    @staticmethod
    def get_available_voices(service_name: str, model_name: Optional[str] = None):
        """
        Return a list of available voices/speakers for a given service/model.

        Parameters:
            service_name (str): The TTS backend name.
            model_name (str): The model identifier (if required by backend).

        Returns:
            list: List of available voices/speakers, or an empty list if not supported.
        """
        voices = []
        if service_name == "gtts":
            # gTTS supports language codes as "voices"
            try:
                import gtts.lang
                voices = list(gtts.lang.tts_langs().keys())
            except Exception:
                voices = ["en"]
        elif service_name == "xttsv2" and CoquiTTS:
            try:
                tts = CoquiTTS(model_name=model_name or "tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                if hasattr(tts, "speakers") and tts.speakers:
                    voices = list(tts.speakers)
                elif hasattr(tts, "languages") and tts.languages:
                    voices = list(tts.languages)
            except Exception:
                voices = []
        # Future: add Piper or other TTS backends here
        return voices

    @staticmethod
    def get_default_model(service_name: str):
        """
        Return the default model for a given service, for UI selection.
        """
        models = TTSManager.get_available_models(service_name)
        return models[0] if models else None

    @staticmethod
    def get_default_voice(service_name: str, model_name: Optional[str] = None):
        """
        Return the default voice for a given service/model, for UI selection.
        """
        voices = TTSManager.get_available_voices(service_name, model_name)
        return voices[0] if voices else None


if __name__ == "__main__":

    async def test_async_tts_manager():
        """
        Asynchronously tests the TTSManager with available TTS services and outputs synthesized audio files.

        This function initializes TTSManager instances for each supported backend (gTTS and XTTSv2), synthesizes sample speech asynchronously, and prints the output file paths. It demonstrates both default and language-specific synthesis, and prepares the output directory for test results.
        """
        logger.info("--- Client TTSManager Async Test ---")
        test_output_dir = CLIENT_TEMP_AUDIO_PATH  # Use configured temp path
        try:
            os.makedirs(test_output_dir, exist_ok=True)
            logger.info(f"Test outputs will be in: {test_output_dir}")
        except OSError as e:
            logger.error(
                f"Could not create test output directory {test_output_dir}: {e}. Aborting test.",
                exc_info=True,
            )
            return

        if "gtts" in TTSManager.list_services():
            logger.info("\nTesting gTTS (async)...")
            tts_g = TTSManager(tts_service_name="gtts", language="fr")
            if tts_g.is_initialized:
                out_g_path = await tts_g.synthesize(
                    "Bonjour le monde, de manière asynchrone.",
                    "client_gtts_async_test.mp3",
                )
                if out_g_path:
                    logger.info(f"gTTS async test audio saved to {out_g_path}")
                else:
                    logger.error("gTTS async test synthesis failed.")
            else:
                logger.warning(
                    "gTTS (async test) was not initialized, skipping synthesis test."
                )

        if "xttsv2" in TTSManager.list_services():
            logger.info("\nTesting XTTSv2 (async)...")
            tts_x = TTSManager(
                tts_service_name="xttsv2",
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                language="en",
            )
            if tts_x.is_initialized:
                out_x_path = await tts_x.synthesize(
                    "Hello from client Coqui XTTS, async default voice.",
                    "client_xtts_async_default.wav",
                )
                if out_x_path:
                    logger.info(
                        f"XTTSv2 async (default voice) test audio saved to {out_x_path}"
                    )
                else:
                    logger.error("XTTSv2 async (default voice) test synthesis failed.")

                # Example for cloned voice test (requires a reference audio)
                # Ensure you have a sample like 'dummy_speaker.wav' in CLIENT_TTS_REFERENCE_VOICES_PATH for this to run
                # dummy_speaker_path = os.path.join(CLIENT_TTS_REFERENCE_VOICES_PATH, "dummy_speaker.wav")
                # if os.path.exists(dummy_speaker_path):
                #    logger.info(f"Testing XTTSv2 cloned voice with {dummy_speaker_path}...")
                #    out_x_cloned_path = await tts_x.synthesize("This is a cloned voice, asynchronously.", "client_xtts_async_cloned.wav", speaker_wav_for_synthesis=dummy_speaker_path)
                #    if out_x_cloned_path:
                #        logger.info(f"XTTSv2 async (cloned voice) test audio saved to {out_x_cloned_path}")
                #    else:
                #        logger.error("XTTSv2 async (cloned voice) test synthesis failed.")
                # else:
                #    logger.info(f"Dummy speaker WAV for cloned voice test not found at {dummy_speaker_path}. Skipping cloned voice test.")
            else:
                logger.warning(
                    "XTTSv2 (async test) was not initialized, skipping synthesis test."
                )

        logger.info("\n--- Client TTSManager Async Test Complete ---")

    # Setup basic logging for the test runner if this script is run directly
    # This is redundant if logging_config.setup_client_logging() is comprehensive
    # but good for standalone testing of this file.
    if not logging.getLogger("dreamweaver_client").hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        )

    asyncio.run(test_async_tts_manager())
