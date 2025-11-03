import time
import pyaudio
import webrtcvad
import collections
import speech_recognition as sr
from deepseek_api import get_deepseek_response
import requests
import traceback
import threading
import os
from dotenv import load_dotenv

load_dotenv()

# Optional TTS helper
try:
    from tts_api import speak_text, ELEVEN_API_KEY
    TTS_AVAILABLE = bool(ELEVEN_API_KEY)
    if not TTS_AVAILABLE:
        print("⚠️  ELEVEN_API_KEY not set - TTS will be disabled")
except Exception as e:
    print(f"⚠️  TTS module not available: {e}")
    TTS_AVAILABLE = False

# === CONFIG ===
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = FRAME_SIZE * 2

# Configurable language (default Turkish)
LANGUAGE = os.getenv("BOT_LANGUAGE", "tr-TR")

vad = webrtcvad.Vad(2)
r = sr.Recognizer()

# Conversation memory
HISTORY_KEEP = 6
conversation_history = []

# Prevent recording while TTS is playing
is_playing_lock = threading.Lock()


def record_with_vad(timeout=15):
    """
    Record audio using VAD (Voice Activity Detection).
    Returns AudioData when speech is detected, None on timeout/silence.
    """
    pa = pyaudio.PyAudio()
    stream = None
    
    try:
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=FRAME_SIZE)
    except Exception as e:
        print(f"❌ Could not open audio stream: {e}")
        pa.terminate()
        return None

    ring_buffer = collections.deque(maxlen=10)
    triggered = False
    speech_frames = []
    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            # Skip recording if TTS is playing
            if is_playing_lock.locked():
                time.sleep(0.05)
                try:
                    _ = stream.read(FRAME_SIZE, exception_on_overflow=False)
                except Exception:
                    pass
                continue

            try:
                frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            except (IOError, OSError):
                continue

            if len(frame) != BYTES_PER_FRAME:
                continue

            try:
                is_speech = vad.is_speech(frame, RATE)
            except Exception:
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
            if stream:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        pa.terminate()


def append_history(role: str, content: str):
    """Append message to conversation history with size limit."""
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > HISTORY_KEEP:
        del conversation_history[0: len(conversation_history) - HISTORY_KEEP]


def speak_and_block(response_text: str, language: str):
    """Speak text and block recording during playback."""
    if not TTS_AVAILABLE:
        return
    
    with is_playing_lock:
        try:
            speak_text(response_text, language)
        except Exception as e:
            print(f"🔊 TTS playback error (ignored): {e}")


def listen_loop():
    """Main listening loop."""
    lang_name = "Turkish" if LANGUAGE == "tr-TR" else "English"
    print(f"🎧 Mic ready. Speak freely ({lang_name}). Press Ctrl+C to stop.")
    
    if not TTS_AVAILABLE:
        print("⚠️  Running without TTS - responses will be text-only")
    
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while True:
        try:
            audio_data = record_with_vad(timeout=30)
            if audio_data is None:
                time.sleep(0.05)
                continue

            # Transcribe
            try:
                text = r.recognize_google(audio_data, language=LANGUAGE)
                consecutive_errors = 0  # Reset error counter on success
            except sr.UnknownValueError:
                print("❓ Could not understand audio.")
                continue
            except sr.RequestError as e:
                print(f"❌ Speech service error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}). Exiting.")
                    break
                time.sleep(1)
                continue

            print(f"\n🎙️  You said ({lang_name}): {text}")
            append_history("user", text)

            # Get AI response
            try:
                response = get_deepseek_response(
                    text, 
                    language=LANGUAGE,
                    convo_history=conversation_history
                )
                if not isinstance(response, str):
                    response = str(response)
                print(f"🤖 BerkBot: {response}\n")
                append_history("assistant", response)
                consecutive_errors = 0  # Reset on success
            except requests.exceptions.RequestException as e:
                print(f"❌ DeepSeek network error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}). Exiting.")
                    break
                continue
            except Exception as e:
                print(f"❌ DeepSeek error: {e}")
                traceback.print_exc()
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}). Exiting.")
                    break
                continue

            # Speak response
            if TTS_AVAILABLE:
                try:
                    speak_and_block(response, LANGUAGE)
                except Exception as e:
                    print(f"🔊 TTS error (ignored): {e}")

            time.sleep(0.15)

        except KeyboardInterrupt:
            print("\n🛑 Stopped listening by user.")
            break
        except Exception as e:
            print(f"⚠️  Unexpected error (ignored): {repr(e)}")
            traceback.print_exc()
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"❌ Too many consecutive errors ({consecutive_errors}). Exiting.")
                break
            time.sleep(0.1)


def main():
    """Entry point."""
    try:
        listen_loop()
    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        traceback.print_exc()
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()