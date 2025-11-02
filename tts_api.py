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
    "Accept": "audio/wav",  # prefer WAV to avoid conversion
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


def speak_text(text: str, language: str = "en-US", voice_id: str | None = None, speed: float = 1.0) -> bool:
    """
    Request ElevenLabs TTS and play the result from memory (no temp file).
    Returns True on successful playback, False otherwise.
    """
    if not ELEVEN_API_KEY:
        print("🔊 ElevenLabs API key not set (ELEVEN_API_KEY).")
        return False

    vid = voice_id or VOICE_MAP.get(language) or VOICE_MAP.get("en-US")
    if not vid:
        print("🔊 ElevenLabs voice ID missing for language:", language)
        return False

    url = f"{ELEVEN_API_BASE}/text-to-speech/{vid}"
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}

    try:
        resp = _session.post(url, json=payload, timeout=_DEFAULT_TIMEOUT, stream=True)
    except requests.RequestException as e:
        print("🔊 ElevenLabs network/error:", e)
        return False

    if resp.status_code >= 400:
        print(f"🔊 ElevenLabs error: {resp.status_code} {resp.reason}")
        try:
            print("Response body:", resp.text)
        except Exception:
            pass
        return False

    content_type = (resp.headers.get("Content-Type") or "").lower()
    # accumulate bytes (small)
    try:
        audio_bytes = resp.content
    except Exception as e:
        # fallback to streamed accumulation
        b = bytearray()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                b.extend(chunk)
        audio_bytes = bytes(b)

    # Try WAV path first
    if "wav" in content_type or audio_bytes[:4] == b'RIFF':
        wav_bytes = audio_bytes
    else:
        # try mp3 -> wav conversion in-memory
        converted = _mp3_to_wav_bytes(audio_bytes)
        if converted:
            wav_bytes = converted
        else:
            print("🔊 Received non-wav audio and conversion failed.")
            return False

    # optional speedup: re-encode faster locally (frame rate trick) - simple and fast
    if speed != 1.0:
        try:
            # use ffmpeg for in-memory speed change if available
            if _has_ffmpeg():
                factor = float(speed)
                # ffmpeg at runtime: change tempo without pitch shift using atempo (supports 0.5-2.0 chained)
                # build atempo filters (supports 0.5->2.0 per filter)
                # For generality, we will use single atempo if in range, otherwise chain
                atempo_filters = []
                v = factor
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
                wav_bytes, _ = p.communicate(wav_bytes)
                if p.returncode != 0:
                    # if conversion fails, fallback to original wav_bytes
                    pass
            else:
                # fallback: change frame rate quick trick (will change pitch)
                import wave, audioop
                buf = io.BytesIO(wav_bytes)
                wf = wave.open(buf, 'rb')
                params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes(), wf.getcomptype(), wf.getcompname())
                raw = wf.readframes(params[3])
                wf.close()
                new_rate = int(params[2] * speed)
                # repackage to wav bytes with new frame rate via ffmpeg not available -> skip speed
                # safe fallback: skip speed change if we can't do it properly
                pass
        except Exception:
            pass

    # play wav_bytes via PyAudio
    ok = _play_wav_bytes(wav_bytes)
    return ok


def close():
    """Call when program exits to terminate the shared PyAudio instance."""
    _close_pyaudio()
