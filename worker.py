import os
import gc
import logging
import torch
import whisper
import numpy as np
from dotenv import load_dotenv
from celery import Celery

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

celery = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

# Lazy loading for models to avoid loading in main process
_whisper_model = None
_diarize_func = None


def get_whisper_model():
    """Lazy load whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model...")
        WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}, model: {WHISPER_MODEL}")
        _whisper_model = whisper.load_model(WHISPER_MODEL).to(device)
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def get_diarize():
    """Lazy load diarization function."""
    global _diarize_func
    if _diarize_func is None:
        logger.info("Loading diarization pipeline...")
        from diarization import diarize
        _diarize_func = diarize
        logger.info("Diarization pipeline loaded successfully")
    return _diarize_func


def sec_to_sample(t: float, sr: int) -> int:
    """Convert seconds to sample index."""
    return int(t * sr)


@celery.task
def transcribe_and_diarize(audio_path: str, index: int, offset: float = 0.0):
    """
    Process audio segment: run diarization then transcribe each speaker turn.
    
    Args:
        audio_path: Path to the audio segment file
        index: Segment index for ordering results
        offset: Time offset in seconds (for chunked audio)
    """
    logger.info(f"Processing segment {index}: {audio_path}")
    
    # Get models (lazy loaded)
    model = get_whisper_model()
    diarize = get_diarize()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Run speaker diarization (returns segments, audio data, and sample rate)
    logger.info(f"Running diarization on segment {index}...")
    speaker_segments, audio, sr = diarize(audio_path)
    logger.info(f"Diarization complete: {len(speaker_segments)} speaker turns found")

    # 2. Transcribe each speaker turn directly
    aligned = []
    
    for i, seg in enumerate(speaker_segments):
        # Extract audio for this speaker turn
        start_sample = sec_to_sample(seg["start"], sr)
        end_sample = sec_to_sample(seg["end"], sr)
        segment_audio = audio[start_sample:end_sample]
        
        # Skip very short segments (less than 0.5 seconds)
        if len(segment_audio) < sr * 0.5:
            continue
        
        # Transcribe the segment
        logger.info(f"Transcribing turn {i+1}/{len(speaker_segments)} ({seg['speaker']})")
        result = model.transcribe(
            segment_audio.astype(np.float32),
            fp16=(device == "cuda"),
            language=os.getenv("WHISPER_LANGUAGE", None),  # None = auto-detect
        )
        
        text = result["text"].strip()
        
        # Skip empty transcriptions
        if not text:
            continue
        
        aligned.append({
            "start": round(seg["start"] + offset, 2),
            "end": round(seg["end"] + offset, 2),
            "speaker": seg["speaker"],
            "text": text
        })
    
    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Segment {index} complete: {len(aligned)} transcribed segments")
    return {
        "index": index,
        "segments": aligned
    }
