# server.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io

from app_core import AppCore

app = FastAPI(title="Voice-to-Response API", version="1.0")

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = AppCore(language="tr-TR")


@app.post("/chat")
async def chat(request: Request):
    """
    Text input -> DeepSeek reply (JSON response, no audio).
    
    Request body: {"text": "your message"}
    Response: {"reply": "AI response"}
    """
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
    
    user_text = data.get("text", "").strip()

    if not user_text:
        return JSONResponse(content={"error": "Eksik metin"}, status_code=400)

    try:
        response = bot.get_response(user_text)
        return JSONResponse(content={"reply": response})
    except Exception as e:
        print("❌ /chat error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/process_text")
async def process_text(text: str = Form(...)):
    """
    Plain text -> DeepSeek -> ElevenLabs audio.
    
    Returns: Audio WAV stream with X-Reply-Text header containing the text response
    """
    if not text or not text.strip():
        return JSONResponse({"error": "Eksik metin"}, status_code=400)
    
    try:
        result = bot.process_text(text.strip(), return_audio=True)
        reply_text = result["response"]
        audio_bytes = result["audio_bytes"]

        if not audio_bytes:
            return JSONResponse({
                "text": reply_text,
                "warning": "Ses oluşturulamadı."
            })

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "X-Reply-Text": reply_text,
                "Content-Disposition": "inline; filename=response.wav"
            },
        )
    except Exception as e:
        print("❌ /process_text error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Audio upload -> transcribe -> DeepSeek -> ElevenLabs reply.
    
    Returns: Audio WAV stream with headers:
      - X-Reply-Text: AI response text
      - X-Transcribed-Text: What you said
    """
    try:
        audio_bytes = await file.read()
        
        if not audio_bytes:
            return JSONResponse({"error": "Boş ses dosyası"}, status_code=400)
        
        result = bot.handle_audio_bytes(audio_bytes, return_audio=True)
        
        transcribed_text = result["text"]
        reply_text = result["response"]
        response_audio = result["audio_bytes"]

        # If transcription failed
        if not transcribed_text:
            return JSONResponse({
                "error": reply_text,  # Contains "Sizi anlayamadım"
                "transcribed": None
            }, status_code=400)

        # If audio generation failed but we have text
        if not response_audio:
            return JSONResponse({
                "transcribed": transcribed_text,
                "reply": reply_text,
                "warning": "Ses oluşturulamadı."
            })

        return StreamingResponse(
            io.BytesIO(response_audio),
            media_type="audio/wav",
            headers={
                "X-Transcribed-Text": transcribed_text,
                "X-Reply-Text": reply_text,
                "Content-Disposition": "inline; filename=response.wav"
            },
        )
    except Exception as e:
        print("❌ /process_audio error:", e)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/clear_history")
async def clear_history():
    """Clear conversation history."""
    try:
        bot.clear_history()
        return JSONResponse({"message": "Geçmiş temizlendi"})
    except Exception as e:
        print("❌ /clear_history error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "message": "Voice-to-Response API çalışıyor ✅",
        "endpoints": {
            "/chat": "POST - Text to text response (JSON)",
            "/process_text": "POST - Text to audio response",
            "/process_audio": "POST - Audio to audio response",
            "/clear_history": "POST - Clear conversation history"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "language": bot.language}