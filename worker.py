import os
import gc
import torch
import whisper
import numpy as np
from celery import Celery
from diarization import diarize

celery = Celery(
    "worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

# Load Whisper model once (GPU if available)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(WHISPER_MODEL).to(device)


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
    # 1. Run speaker diarization (returns segments, audio data, and sample rate)
    speaker_segments, audio, sr = diarize(audio_path)

    # 2. Transcribe each speaker turn directly
    aligned = []
    
    for seg in speaker_segments:
        # Extract audio for this speaker turn
        start_sample = sec_to_sample(seg["start"], sr)
        end_sample = sec_to_sample(seg["end"], sr)
        segment_audio = audio[start_sample:end_sample]
        
        # Skip very short segments (less than 0.5 seconds)
        if len(segment_audio) < sr * 0.5:
            continue
        
        # Transcribe the segment
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

    return {
        "index": index,
        "segments": aligned
    }
