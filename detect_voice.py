import time
import pyaudio
import webrtcvad
import collections
import speech_recognition as sr
from deepseek_api import get_deepseek_response
import requests
import traceback
import threading

# Optional TTS helper (won't crash if missing)
try:
    from tts_api import speak_text
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# === CONFIG ===
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = FRAME_SIZE * 2

vad = webrtcvad.Vad(2)
r = sr.Recognizer()

# Conversation memory
HISTORY_KEEP = 6
conversation_history = []

# Prevent recording while TTS is playing
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
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()

def append_history(role: str, content: str):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > HISTORY_KEEP:
        del conversation_history[0: len(conversation_history) - HISTORY_KEEP]

def speak_and_block(response_text: str):
    if not TTS_AVAILABLE:
        return
    with is_playing_lock:
        try:
            speak_text(response_text, "tr-TR")
        except Exception as e:
            print("TTS playback error (ignored):", e)

def listen_loop():
    print("🎧 Mic ready. Speak freely (Turkish only). Press Ctrl+C to stop.")
    while True:
        try:
            audio_data = record_with_vad(timeout=30)
            if audio_data is None:
                time.sleep(0.05)
                continue

            # Transcribe in Turkish only
            try:
                text = r.recognize_google(audio_data, language="tr-TR")
            except sr.UnknownValueError:
                print("Could not understand.")
                continue
            except sr.RequestError as e:
                print("Speech service error:", e)
                continue

            language = "tr-TR"
            print(f"\n🎙 You said (Turkish): {text}")
            append_history("user", text)

            try:
                response = get_deepseek_response(text, convo_history=conversation_history)
                if not isinstance(response, str):
                    response = str(response)
                print(f"🤖 DeepSeek: {response}\n")
                append_history("assistant", response)
            except requests.exceptions.RequestException as e:
                print("DeepSeek network error:", e)
                continue
            except Exception as e:
                print("DeepSeek error:", e)
                traceback.print_exc()
                continue

            if TTS_AVAILABLE:
                try:
                    speak_and_block(response)
                except Exception as e:
                    print("TTS error (ignored):", e)

            time.sleep(0.15)

        except KeyboardInterrupt:
            print("\n🛑 Stopped listening by user.")
            break
        except Exception as e:
            print("Unexpected error (ignored):", repr(e))
            traceback.print_exc()
            time.sleep(0.1)

def main():
    try:
        listen_loop()
    except Exception as e:
        print("Fatal error in main:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
