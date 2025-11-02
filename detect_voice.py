import threading
import keyboard
import speech_recognition as sr
from deepseek_api import get_deepseek_response

r = sr.Recognizer()
mic = sr.Microphone()
stop_listening = False
current_language = "tr-TR"  # Default Turkish

# === LIVE SPEECH LISTENING ===
def listen_loop():
    global stop_listening, current_language
    with mic as source:
        print("Calibrating mic (2 sec)...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("Mic ready. Speak freely.\n")

        while not stop_listening:
            print(f"🎙 Speak now... (Lang: {'TR' if current_language == 'tr-TR' else 'EN'})")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                continue

            try:
                text = r.recognize_google(audio, language=current_language)
                print(f"You said: {text}")
                response = get_deepseek_response(text, current_language)
                print(f"🤖 DeepSeek: {response}\n")
            except sr.UnknownValueError:
                print("Could not understand.")
            except sr.RequestError as e:
                print("Speech API error:", e)

def toggle_language():
    global current_language
    current_language = "en-US" if current_language == "tr-TR" else "tr-TR"
    print(f"\nLanguage switched to: {'Turkish' if current_language == 'tr-TR' else 'English'}")

def live_mode():
    global stop_listening
    stop_listening = False
    listener = threading.Thread(target=listen_loop, daemon=True)
    listener.start()
    keyboard.add_hotkey('space', toggle_language)
    print("🎧 Listening... (Space = toggle language, Enter = stop)\nCurrent language: Turkish")
    try:
        keyboard.wait("enter")
    except KeyboardInterrupt:
        pass
    stop_listening = True
    listener.join()
    print("Stopped listening.")

if __name__ == "__main__":
    live_mode()
