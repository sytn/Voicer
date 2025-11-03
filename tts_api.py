# tts_api.py
import io
import os
import requests
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_API_BASE = "https://api.elevenlabs.io/v1"

# same Turkish voice for both languages (set your id)
VOICE_MAP = {
    "en-US": "fg8pljYEn5ahwjyOQaro",
    "tr-TR": "fg8pljYEn5ahwjyOQaro",
}

HEADERS_BASE = {
    "xi-api-key": ELEVEN_API_KEY,
    "Accept": "audio/wav",
    "Content-Type": "application/json",
}

_session = requests.Session()
_session.headers.update({"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/wav", "Content-Type": "application/json"})
_DEFAULT_TIMEOUT = (3, 30)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# create a single PyAudio instance for reuse
_GLOBAL_PYAUDIO = None


def _ensure_pyaudio():
    global _GLOBAL_PYAUDIO
    if _GLOBAL_PYAUDIO is None:
        import pyaudio
        _GLOBAL_PYAUDIO = pyaudio.PyAudio()
    return _GLOBAL_PYAUDIO


def _close_pyaudio():
    global _GLOBAL_PYAUDIO
    try:
        if _GLOBAL_PYAUDIO is not None:
            _GLOBAL_PYAUDIO.terminate()
    except Exception:
        pass
    _GLOBAL_PYAUDIO = None


def _play_wav_bytes(wav_bytes: bytes) -> bool:
    """
    Play WAV bytes via PyAudio (no temp files).
    """
    import wave
    pa = _ensure_pyaudio()

    buf = io.BytesIO(wav_bytes)
    try:
        wf = wave.open(buf, 'rb')
    except Exception as e:
        print("🔊 Could not open WAV from bytes:", e)
        return False

    try:
        stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                         channels=wf.getnchannels(),
                         rate=wf.getframerate(),
                         output=True)
        chunk = 1024
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print("🔊 Playback error (PyAudio WAV):", e)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        return False
    finally:
        wf.close()
    return True


def _mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes | None:
    """
    Convert MP3 bytes -> WAV bytes using ffmpeg (in-memory).
    Requires ffmpeg in PATH.
    """
    if not _has_ffmpeg():
        print("🔊 ffmpeg not found for MP3 conversion.")
        return None

    try:
        p = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        wav_bytes, _ = p.communicate(mp3_bytes)
        if p.returncode != 0:
            return None
        return wav_bytes
    except Exception as e:
        print("🔊 ffmpeg conversion failed:", e)
        return None


def _apply_speed_change(wav_bytes: bytes, speed: float) -> bytes:
    """
    Apply speed change to WAV bytes using ffmpeg if available.
    Returns modified WAV bytes, or original if conversion fails.
    """
    if not _has_ffmpeg() or speed == 1.0:
        return wav_bytes
    
    try:
        # Build atempo filters (supports 0.5-2.0 per filter)
        atempo_filters = []
        v = float(speed)
        
        # Chain multiple atempo filters if needed
        while v > 2.0:
            atempo_filters.append("atempo=2.0")
            v /= 2.0
        while v < 0.5:
            atempo_filters.append("atempo=0.5")
            v *= 2.0
        atempo_filters.append(f"atempo={v:.3f}")
        filter_str = ",".join(atempo_filters)

        p = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-af", filter_str, "-f", "wav", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        modified_wav, _ = p.communicate(wav_bytes)
        
        if p.returncode == 0:
            return modified_wav
        else:
            print("🔊 Speed adjustment failed, using original audio")
            return wav_bytes
    except Exception as e:
        print("🔊 Speed adjustment error:", e)
        return wav_bytes


def get_tts_bytes(text: str, language: str = "en-US", voice_id: str | None = None, speed: float = 1.0) -> bytes | None:
    """
    Request ElevenLabs TTS and return WAV bytes (does not play audio).
    Returns WAV bytes on success, None on failure.
    """
    if not ELEVEN_API_KEY:
        print("🔊 ElevenLabs API key not set (ELEVEN_API_KEY).")
        return None

    vid = voice_id or VOICE_MAP.get(language) or VOICE_MAP.get("en-US")
    if not vid:
        print("🔊 ElevenLabs voice ID missing for language:", language)
        return None

    url = f"{ELEVEN_API_BASE}/text-to-speech/{vid}"
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}

    try:
        resp = _session.post(url, json=payload, timeout=_DEFAULT_TIMEOUT, stream=True)
    except requests.RequestException as e:
        print("🔊 ElevenLabs network/error:", e)
        return None

    if resp.status_code >= 400:
        print(f"🔊 ElevenLabs error: {resp.status_code} {resp.reason}")
        try:
            print("Response body:", resp.text)
        except Exception:
            pass
        return None

    content_type = (resp.headers.get("Content-Type") or "").lower()
    
    # Accumulate bytes
    try:
        audio_bytes = resp.content
    except Exception:
        b = bytearray()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                b.extend(chunk)
        audio_bytes = bytes(b)

    # Convert to WAV if necessary
    if "wav" in content_type or audio_bytes[:4] == b'RIFF':
        wav_bytes = audio_bytes
    else:
        # Try mp3 -> wav conversion in-memory
        converted = _mp3_to_wav_bytes(audio_bytes)
        if converted:
            wav_bytes = converted
        else:
            print("🔊 Received non-wav audio and conversion failed.")
            return None

    # Apply speed change if requested
    if speed != 2.0:
        wav_bytes = _apply_speed_change(wav_bytes, speed)

    return wav_bytes


def speak_text(text: str, language: str = "en-US", voice_id: str | None = None, speed: float = 1.0) -> bool:
    """
    Request ElevenLabs TTS and play the result from memory (no temp file).
    Returns True on successful playback, False otherwise.
    """
    wav_bytes = get_tts_bytes(text, language, voice_id, speed)
    
    if wav_bytes is None:
        return False
    
    # Play wav_bytes via PyAudio
    ok = _play_wav_bytes(wav_bytes)
    return ok


def close():
    """Call when program exits to terminate the shared PyAudio instance."""
    _close_pyaudio()