import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.templating import Jinja2Templates # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles # pyright: ignore[reportMissingImports]
import shutil
import tempfile
import asyncio
from pydantic import BaseModel

from speech.asr import KikuyuASR

# Import your existing functions
from app import get_naive_query_engine_async


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Kikuyu ↔ Swahili AI Translator",
    version="1.0.0",
    description="RAG-based translation chatbot using LlamaIndex + Chroma"
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# ASR Model
# -----------------------------------------------------------------------------
asr_model = KikuyuASR()

# -----------------------------------------------------------------------------
# Query Engine Cache
# -----------------------------------------------------------------------------
_query_engine_cache = {}

async def get_query_engine(collection_name: str, top_k: int):
    cache_key = f"{collection_name}_{top_k}"
    if cache_key not in _query_engine_cache:
        logger.info("Initializing query engine...")
        engine = await get_naive_query_engine_async(
            collection_name=collection_name,
            similarity_top_k=top_k
        )
        _query_engine_cache[cache_key] = engine
    return _query_engine_cache[cache_key]

# -----------------------------------------------------------------------------
# Request / Response Models
# -----------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    source_language: str  # "kikuyu" or "swahili"
    target_language: str  # "kikuyu" or "swahili"
    top_k: Optional[int] = 6

class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

# -----------------------------------------------------------------------------
# Root HTML Endpoint
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "running"}

# -----------------------------------------------------------------------------
# Translation Endpoint
# -----------------------------------------------------------------------------
@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    source = request.source_language.lower()
    target = request.target_language.lower()

    if source == target:
        raise HTTPException(status_code=400, detail="Source and target languages must be different.")
    if source not in ["kikuyu", "swahili"] or target not in ["kikuyu", "swahili"]:
        raise HTTPException(status_code=400, detail="Unsupported language.")

    try:
        engine = await get_query_engine("kikuyu_swahili_translation", request.top_k)

        prompt = f"""
        You are a professional translator.
        Translate the following text from {source} to {target}.

        Text:
        {request.text}

        Provide only the translated output.
        """

        response = await asyncio.to_thread(engine.query, prompt)

        return TranslateResponse(
            original_text=request.text,
            translated_text=str(response),
            source_language=source,
            target_language=target
        )

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed.")

# -----------------------------------------------------------------------------
# Transcription Endpoint
# -----------------------------------------------------------------------------
@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .wav, .mp3, or .flac file.")

    temp_file_path = None
    try:
        # Use tempfile for safe temporary storage
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file_path = tmp.name
            tmp.write(await file.read())

        # Transcribe asynchronously in a thread
        transcription = await asyncio.to_thread(asr_model.transcribe, temp_file_path)

        return templates.TemplateResponse("index.html", {"request": request, "transcription": transcription})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed.")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)