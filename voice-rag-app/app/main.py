# app/main.py
import pysqlite3                     # swap in newer SQLite
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import time
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import soundfile as sf
import tempfile
from pydantic import BaseModel
from typing import List, Optional

# Import our modules
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.speech import SpeechProcessor, VoiceActivityDetector
from app.rag_engine import RAGEngine
from app.summarizer import Summarizer

# Create FastAPI app
app = FastAPI(title="Voice-Interactive RAG System")

# Initialize components
pdf_processor = PDFProcessor()
vector_store = VectorStore(persist_directory="./chroma_db")
speech_processor = SpeechProcessor()
rag_engine = RAGEngine()
summarizer = Summarizer()
vad = VoiceActivityDetector()

# Create directories for uploads and temp files
os.makedirs("uploads", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="ui"), name="static")

# Data models
class QueryRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = 5

class SummaryRequest(BaseModel):
    collection_name: str
    use_full_text: bool = True
    top_k: int = 10

class TranscriptionResponse(BaseModel):
    text: str

# Routes
@app.get("/")
async def read_root():
    return FileResponse("ui/index.html")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Generate unique ID for the collection
    collection_id = f"pdf_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Save uploaded file
    file_path = f"uploads/{collection_id}.pdf"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Process PDF
        pdf_data = pdf_processor.process_pdf(file_path)
        
        # Store in vector DB
        vector_store.add_documents(
            collection_name=collection_id,
            chunks=pdf_data["chunks"],
            embeddings=pdf_data["embeddings"]
        )
        
        # Generate a summary of the full document
        full_summary = summarizer.summarize(pdf_data["text"])
        
        return {
            "status": "success",
            "collection_name": collection_id,
            "num_chunks": len(pdf_data["chunks"]),
            "summary": full_summary
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/query", response_model=dict)
async def process_query(request: QueryRequest):
    try:
        # Process query through RAG pipeline
        result = rag_engine.process_query(
            query=request.query,
            vector_store=vector_store,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
        
        # Generate audio response
        audio_path = f"temp/response_{uuid.uuid4().hex[:8]}.wav"
        speech_processor.text_to_speech(result["answer"], output_path=audio_path)
        
        result["audio_path"] = audio_path
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/summarize", response_model=dict)
async def generate_summary(request: SummaryRequest):
    try:
        # Get collection
        collection = vector_store.client.get_collection(request.collection_name)
        
        if request.use_full_text:
            # Get original PDF path
            pdf_path = f"uploads/{request.collection_name}.pdf"
            
            # Extract text
            full_text = pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Generate summary
            summary = summarizer.summarize(full_text)
        else:
            # Get all chunks
            all_documents = collection.get()["documents"]
            
            # Sort by relevance if needed
            # For simplicity, we'll just take the top k chunks
            chunks_to_summarize = all_documents[:request.top_k]
            
            # Generate summary
            summary = summarizer.summarize_chunks(chunks_to_summarize)
        
        # Generate audio for summary
        audio_path = f"temp/summary_{uuid.uuid4().hex[:8]}.wav"
        speech_processor.text_to_speech(summary, output_path=audio_path)
        
        return {
            "summary": summary,
            "audio_path": audio_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/transcribe-audio", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded audio file
    temp_audio_path = f"temp/audio_{uuid.uuid4().hex[:8]}.wav"
    with open(temp_audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Transcribe audio
        transcription = speech_processor.transcribe_audio(audio_file_path=temp_audio_path)
        
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Buffer to store audio chunks
        audio_buffer = []
        sample_rate = 16000
        
        # Process incoming audio stream
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array (assuming 16-bit PCM format)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            audio_buffer.append(audio_chunk)
            
            # Check if we have enough audio for transcription
            if len(audio_buffer) > 5:  # Process every ~5 chunks
                # Combine chunks
                audio_data = np.concatenate(audio_buffer)
                
                # Detect voice activity
                speech_segments = vad.detect_voice(audio_data)
                
                if speech_segments:
                    # Transcribe the audio
                    transcription = speech_processor.transcribe_audio(
                        audio_array=audio_data,
                        sample_rate=sample_rate
                    )
                    
                    # Send transcription back to client
                    await websocket.send_json({"transcription": transcription})
                    
                    # Clear buffer
                    audio_buffer = []
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in WebSocket: {str(e)}")
        await websocket.close()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)