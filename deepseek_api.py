import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def get_deepseek_response(text, language="en-US"):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful bilingual assistant (Turkish & English)."
            },
            {"role": "user", "content": text}
        ],
        "stream": False
    }

    res = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    if res.status_code != 200:
        print("❌ DeepSeek error:", res.text)  # helpful debug print
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]
