
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File # type: ignore
from pydantic import BaseModel # type: ignore
import asyncio
from speech.asr import KikuyuASR

# Import your existing functions
from app import (
    get_naive_query_engine_async,
)

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

    if request.source_language.lower() == request.target_language.lower():
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be different."
        )

    if request.source_language.lower() not in ["kikuyu", "swahili"]:
        raise HTTPException(status_code=400, detail="Unsupported source language.")

    if request.target_language.lower() not in ["kikuyu", "swahili"]:
        raise HTTPException(status_code=400, detail="Unsupported target language.")

    try:
        engine = await get_query_engine(
            collection_name="kikuyu_swahili_translation",
            top_k=request.top_k
        )

        # Construct structured translation prompt
        prompt = f"""
        You are a professional translator.
        Translate the following text from {request.source_language}
        to {request.target_language}.

        Text:
        {request.text}

        Provide only the translated output.
        """

        response = await asyncio.to_thread(engine.query, prompt)

        return TranslateResponse(
            original_text=request.text,
            translated_text=str(response),
            source_language=request.source_language,
            target_language=request.target_language
        )

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail="Translation failed.")


asr_model = KikuyuASR()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .wav, .mp3, or .flac file.")

    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())  # Read and write the uploaded file content

        transcription = asr_model.transcribe(temp_file_path)
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed.")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)