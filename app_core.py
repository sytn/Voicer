# app_core.py
from typing import Optional, List, Dict, Any
import io
import threading
import speech_recognition as sr
import requests

# Import your own modules directly
from deepseek_api import get_deepseek_response
from tts_api import speak_text, get_tts_bytes, ELEVEN_API_KEY, VOICE_MAP


class AppCore:
    def __init__(self, language: str = "tr-TR", history_keep: int = 6):
        self.language = language
        self.history_keep = history_keep
        self.conversation_history: List[Dict[str, str]] = []

        self.recognizer = sr.Recognizer()
        self._session = requests.Session()
        self._play_lock = threading.Lock()

    # ---------- Conversation History ----------
    def append_history(self, role: str, content: str) -> None:
        """Append a message to conversation history and maintain max length."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.history_keep:
            self.conversation_history = self.conversation_history[-self.history_keep:]

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history = []

    # ---------- ASR ----------
    def transcribe_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe given audio bytes to text."""
        try:
            with sr.AudioFile(io.BytesIO(audio_bytes)) as src:
                audio = self.recognizer.record(src)
                return self.recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            return None
        except Exception:
            try:
                audio_data = sr.AudioData(audio_bytes, 16000, 2)
                return self.recognizer.recognize_google(audio_data, language=self.language)
            except Exception:
                return None

    # ---------- DeepSeek ----------
    def get_response(self, text: str, max_tokens: int = 150) -> str:
        """Get AI response via DeepSeek."""
        try:
            response = get_deepseek_response(
                text,
                language=self.language,
                convo_history=self.conversation_history,
                max_tokens=max_tokens
            )
            self.append_history("user", text)
            self.append_history("assistant", response)
            return response
        except Exception as e:
            print("❌ DeepSeek error:", e)
            # Return error message in appropriate language
            if self.language == "tr-TR":
                return "Bir hata oluştu."
            else:
                return "An error occurred."

    # ---------- TTS ----------
    def speak(self, text: str) -> bool:
        """Speak using ElevenLabs' speak_text() function."""
        try:
            return speak_text(text, language=self.language)
        except Exception as e:
            print("🔊 TTS error:", e)
            return False

    def get_tts_bytes(self, text: str) -> Optional[bytes]:
        """Get TTS audio as WAV bytes without playing."""
        try:
            return get_tts_bytes(text, language=self.language)
        except Exception as e:
            print("🔊 TTS bytes error:", e)
            return None

    # ---------- End-to-End ----------
    def handle_audio_bytes(self, audio_bytes: bytes, return_audio: bool = True) -> Dict[str, Any]:
        """
        Transcribe → DeepSeek → TTS (optional)
        
        Args:
            audio_bytes: Input audio bytes to transcribe
            return_audio: If True, generate TTS audio bytes
            
        Returns:
            Dictionary with keys: text, response, audio_bytes
        """
        result = {"text": None, "response": None, "audio_bytes": None}

        # Transcribe
        text = self.transcribe_bytes(audio_bytes)
        if not text:
            if self.language == "tr-TR":
                result["response"] = "Sizi anlayamadım."
            else:
                result["response"] = "I couldn't understand you."
            return result

        result["text"] = text
        
        # Get AI response
        response = self.get_response(text)
        result["response"] = response

        # Generate TTS audio if requested
        if return_audio:
            audio_bytes = self.get_tts_bytes(response)
            result["audio_bytes"] = audio_bytes

        return result

    def process_text(self, text: str, return_audio: bool = True) -> Dict[str, Any]:
        """
        Process text input → DeepSeek → TTS (optional)
        
        Args:
            text: Input text
            return_audio: If True, generate TTS audio bytes
            
        Returns:
            Dictionary with keys: response, audio_bytes
        """
        result = {"response": None, "audio_bytes": None}
        
        # Get AI response
        response = self.get_response(text)
        result["response"] = response
        
        # Generate TTS audio if requested
        if return_audio:
            audio_bytes = self.get_tts_bytes(response)
            result["audio_bytes"] = audio_bytes
        
        return result

    # ---------- Cleanup ----------
    def close(self):
        """Clean up resources."""
        try:
            self._session.close()
        except Exception:
            pass