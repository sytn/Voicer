# detect_voice_debug.py — drop-in instrumentation around your existing logic
import sys
import os
import time
import logging
import traceback
import faulthandler

# enable C-level crash tracebacks to stderr (helpful for native crashes)
faulthandler.enable(all_threads=True)

# --- logging setup ---
LOG_FILE = "debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

def log_exception(exc_type, exc, tb):
    logging.error("Uncaught exception", exc_info=(exc_type, exc, tb))
    # also write formatted traceback
    logging.error("".join(traceback.format_exception(exc_type, exc, tb)))

sys.excepthook = log_exception

# keep a safe wrapper so SystemExit is logged
_original_exit = sys.exit
def _safe_exit(code=0):
    logging.warning("Program requested exit with code %s", code)
    # don't exit — log and raise SystemExit to be caught by outer loop
    raise SystemExit(code)
sys.exit = _safe_exit

# --- import the rest of your app inside try so import-time errors are logged ---
try:
    import pyaudio
    import webrtcvad
    import collections
    import speech_recognition as sr
    from langdetect import detect
    from deepseek_api import get_deepseek_response
    import requests

    try:
        from tts_api import speak_text
        TTS_AVAILABLE = True
    except Exception as e:
        logging.warning("TTS import failed: %s", e)
        TTS_AVAILABLE = False

    # --- paste your working detect_voice.py code here, but replace main() with guarded loop ---
    RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
    BYTES_PER_FRAME = FRAME_SIZE * 2

    vad = webrtcvad.Vad(2)
    r = sr.Recognizer()
    HISTORY_KEEP = 6
    conversation_history = []
    import threading
    is_playing_lock = threading.Lock()

    def record_with_vad(timeout=15):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                         input=True, frames_per_buffer=FRAME_SIZE)
        ring_buffer = collections.deque(maxlen=10)
        triggered = False
        speech_frames = []
        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                if is_playing_lock.locked():
                    time.sleep(0.05)
                    try:
                        _ = stream.read(FRAME_SIZE, exception_on_overflow=False)
                    except Exception:
                        pass
                    continue
                try:
                    frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
                except Exception as e:
                    logging.warning("stream.read error: %s", e)
                    continue
                if len(frame) != BYTES_PER_FRAME:
                    logging.debug("skipping malformed frame len=%d", len(frame))
                    continue
                try:
                    is_speech = vad.is_speech(frame, RATE)
                except Exception as e:
                    logging.debug("vad error: %s", e)
                    continue
                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = sum(1 for f, v in ring_buffer if v)
                    if num_voiced > 0.6 * ring_buffer.maxlen:
                        triggered = True
                        for f, _ in ring_buffer:
                            speech_frames.append(f)
                        ring_buffer.clear()
                else:
                    speech_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = sum(1 for f, v in ring_buffer if not v)
                    if num_unvoiced > 0.6 * ring_buffer.maxlen:
                        break
            if not speech_frames:
                return None
            audio_bytes = b"".join(speech_frames)
            return sr.AudioData(audio_bytes, RATE, 2)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()

    def detect_language(text):
        try:
            lang_code = detect(text)
            return "tr-TR" if lang_code == "tr" else "en-US"
        except Exception:
            return "en-US"

    def append_history(role: str, content: str):
        conversation_history.append({"role": role, "content": content})
        if len(conversation_history) > HISTORY_KEEP:
            del conversation_history[0: len(conversation_history) - HISTORY_KEEP]

    def speak_and_block(response_text: str, language: str):
        if not TTS_AVAILABLE:
            return
        with is_playing_lock:
            try:
                logging.info("TTS start")
                speak_text(response_text, language)
                logging.info("TTS done")
            except Exception as e:
                logging.exception("TTS playback exception")

    def listen_once_and_respond():
        audio_data = record_with_vad(timeout=30)
        if audio_data is None:
            logging.debug("no speech detected")
            return True  # keep running
        try:
            text = r.recognize_google(audio_data, language="tr-TR")
        except sr.UnknownValueError:
            try:
                text = r.recognize_google(audio_data, language="en-US")
            except sr.UnknownValueError:
                logging.info("could not understand")
                return True
        except Exception as e:
            logging.exception("speech recognition error")
            return True
        language = detect_language(text)
        logging.info("User (%s): %s", language, text)
        append_history("user", text)
        try:
            response = get_deepseek_response(text, language, convo_history=conversation_history)
            logging.info("DeepSeek reply: %s", response)
            append_history("assistant", response)
        except Exception as e:
            logging.exception("DeepSeek error")
            return True
        if TTS_AVAILABLE:
            speak_and_block(response, language)
        return True

except Exception:
    logging.exception("Import-time exception")
    raise

# --- guarded main loop: auto-restart on unexpected exceptions ---
def main_loop():
    logging.info("Starting main loop (debug mode). Ctrl+C to stop.")
    while True:
        try:
            ok = listen_once_and_respond()
            if not ok:
                logging.warning("listen_once_and_respond requested stop")
                break
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received — exiting main loop.")
            break
        except SystemExit as e:
            logging.error("SystemExit raised: %s", e)
            # do not exit — log and continue (helps find callers of sys.exit)
            time.sleep(0.5)
            continue
        except BaseException as e:
            # catch everything, log full traceback and continue
            logging.exception("Unexpected error in main loop (will continue): %s", e)
            time.sleep(1)
            continue
    logging.info("Main loop ended.")


if __name__ == "__main__":
    main_loop()
