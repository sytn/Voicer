import os
import re
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

DEFAULT_MAX_TOKENS = 200
TEMPERATURE = 0.2
TOP_P = 0.9
HISTORY_KEEP = 6

# Language-specific system prompts
SYSTEM_PROMPTS = {
    "tr-TR": (
        "You are BerkBot — a concise, friendly, and slightly witty bilingual assistant. "
        "Always reply in Turkish. Keep answers short and clear. "
        "IMPORTANT: Format responses for speech synthesis. Produce plain, pronounceable Turkish text only — "
        "do NOT use emojis, emoticons, markdown (**, `, ```), code fences, angle brackets, or stage directions like [laughs]. "
        "Avoid excessive punctuation, unusual symbols, or parentheses. If you give steps, use short numbered sentences. "
        "If unsure, say you don't know and offer how to find out."
    ),
    "en-US": (
        "You are BerkBot — a concise, friendly, and slightly witty bilingual assistant. "
        "Always reply in English. Keep answers short and clear. "
        "IMPORTANT: Format responses for speech synthesis. Produce plain, pronounceable English text only — "
        "do NOT use emojis, emoticons, markdown (**, `, ```), code fences, angle brackets, or stage directions like [laughs]. "
        "Avoid excessive punctuation, unusual symbols, or parentheses. If you give steps, use short numbered sentences. "
        "If unsure, say you don't know and offer how to find out."
    )
}

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
})
_DEFAULT_TIMEOUT = (3, 15)

# Cleaning regex patterns
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
_SPECIAL_CHARS_RE = re.compile(r'[*#~<>•…»«©®∆`]')
_CODE_FENCE_RE = re.compile(r'```.*?```', flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`([^`]*)`')
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r'(\*\*|\*|__|_)(.*?)\1', flags=re.DOTALL)
_BRACKETED_RE = re.compile(r'\[.*?\]|\(.*?\)')
_MULTI_PUNCT_RE = re.compile(r'[\.\?!]{2,}')
_WHITESPACE_RE = re.compile(r'\s+')


def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS by removing markdown, emojis, and special characters."""
    if not text:
        return text
    text = _CODE_FENCE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _MARKDOWN_BOLD_ITALIC_RE.sub(r"\2", text)
    text = _BRACKETED_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _SPECIAL_CHARS_RE.sub("", text)
    text = _MULTI_PUNCT_RE.sub(".", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if text and text[-1] not in ".!?":
        text = text + "."
    return text


def _build_messages(user_text: str, language: str, convo_history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build messages list with appropriate system prompt and conversation history."""
    # Get system prompt for language, default to Turkish
    system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["tr-TR"])
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]
    
    if convo_history:
        tail = convo_history[-HISTORY_KEEP:]
        for m in tail:
            if "role" in m and "content" in m:
                messages.append({"role": m["role"], "content": m["content"]})
    
    # Language-specific hints
    language_hints = {
        "tr-TR": "Cevap bu dilde olmalı: Türkçe.\n\n",
        "en-US": "Reply in this language: English.\n\n"
    }
    hint = language_hints.get(language, language_hints["tr-TR"])
    
    messages.append({"role": "user", "content": f"{hint}{user_text}"})
    return messages


def get_deepseek_response(
    text: str,
    language: str = "tr-TR",
    convo_history: Optional[List[Dict]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Get response from DeepSeek API.
    
    Args:
        text: User input text
        language: Language code ("tr-TR" or "en-US")
        convo_history: Conversation history
        max_tokens: Maximum tokens in response
        
    Returns:
        Cleaned response text suitable for TTS
    """
    messages = _build_messages(text, language, convo_history)
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "web_search": True
    }

    try:
        res = _session.post(DEEPSEEK_URL, json=payload, timeout=_DEFAULT_TIMEOUT)
        if res.status_code != 200:
            print("❌ DeepSeek error (status):", res.status_code, "body:", res.text)
        res.raise_for_status()

        data = res.json()
        content = data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print("❌ DeepSeek network error:", e)
        raise
    except (KeyError, IndexError) as e:
        print("❌ DeepSeek response parsing error:", e)
        content = str(data) if 'data' in locals() else "Error parsing response"

    cleaned = clean_text_for_tts(content.strip())
    return cleaned