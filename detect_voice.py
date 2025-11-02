import speech_recognition as sr
import threading, keyboard

r = sr.Recognizer()
mic = sr.Microphone()
stop_listening = False
current_language = "tr-TR"  # Default to Turkish

def listen_loop():
    global stop_listening, current_language
    with mic as source:
        r.adjust_for_ambient_noise(source)
        while not stop_listening:
            print(f"Speak now... (Current language: {'Turkish' if current_language == 'tr-TR' else 'English'})")
            audio = r.listen(source, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language=current_language)
                lang_name = "Turkish" if current_language == "tr-TR" else "English"
                print(f"You said ({lang_name}): {text}")
            except sr.UnknownValueError:
                print("Could not understand")
            except sr.RequestError as e:
                print("Service error:", e)

def toggle_language():
    global current_language
    current_language = "en-US" if current_language == "tr-TR" else "tr-TR"
    lang_name = "Turkish" if current_language == "tr-TR" else "English"
    print(f"\nLanguage switched to: {lang_name}")

# Start listening thread
listener = threading.Thread(target=listen_loop)
listener.start()

# Register spacebar to toggle language
keyboard.add_hotkey('space', toggle_language)

print("Listening... (press Space to toggle language, Enter to stop)")
print("Current language: Turkish")
keyboard.wait("enter")
stop_listening = True
listener.join()
print("Stopped listening.")