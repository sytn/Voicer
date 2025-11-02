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

SYSTEM_PROMPT = (
    "You are BerkBot — a concise, friendly, and slightly witty bilingual assistant. "
    "Always reply in Turkish for this deployment. Keep answers short and clear. "
    "IMPORTANT: Format responses for speech synthesis. Produce plain, pronounceable Turkish text only — "
    "do NOT use emojis, emoticons, markdown (**, `, ```), code fences, angle brackets, or stage directions like [laughs]. "
    "Avoid excessive punctuation, unusual symbols, or parentheses. If you give steps, use short numbered sentences. "
    "If unsure, say you don't know and offer how to find out."
)

_session = requests.Session()
_session.headers.update({
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
})
_DEFAULT_TIMEOUT = (3, 15)

def _build_messages(user_text: str, convo_history: Optional[List[Dict]] = None) -> List[Dict]:
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if convo_history:
        tail = convo_history[-HISTORY_KEEP:]
        for m in tail:
            if "role" in m and "content" in m:
                messages.append({"role": m["role"], "content": m["content"]})
    # Turkish-only hint
    messages.append({"role": "user", "content": f"Cevap bu dilde olmalı: Türkçe.\n\n{user_text}"})
    return messages

# reuse cleaning helpers (same as before)
import re
_EMOJI_RE = re.compile( "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF" "\U00002700-\U000027BF" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
_SPECIAL_CHARS_RE = re.compile(r'[*#~<>•…»«©®∆`]')
_CODE_FENCE_RE = re.compile(r'```.*?```', flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r'`([^`]*)`')
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r'(\*\*|\*|__|_)(.*?)\1', flags=re.DOTALL)
_BRACKETED_RE = re.compile(r'\[.*?\]|\(.*?\)')
_MULTI_PUNCT_RE = re.compile(r'[\.\?!]{2,}')
_WHITESPACE_RE = re.compile(r'\s+')

def clean_text_for_tts(text: str) -> str:
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

def get_deepseek_response(
    text: str,
    convo_history: Optional[List[Dict]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    messages = _build_messages(text, convo_history)
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "web_search": True
    }

    res = _session.post(DEEPSEEK_URL, json=payload, timeout=_DEFAULT_TIMEOUT)
    if res.status_code != 200:
        print("❌ DeepSeek error (status):", res.status_code, "body:", res.text)
    res.raise_for_status()

    data = res.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = str(data)

    cleaned = clean_text_for_tts(content.strip())
    return cleaned
