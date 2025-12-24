import os
import logging
import torch
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

# Lazy loading for pipeline
_pipeline = None


def get_pipeline():
    """Lazy load the diarization pipeline."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading pyannote diarization pipeline...")
        
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        
        # Fix for PyTorch 2.6+ weights_only=True default
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
        
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            logger.info("Moving pipeline to GPU...")
            _pipeline.to(torch.device("cuda"))
        
        logger.info("Diarization pipeline loaded successfully")
    return _pipeline


def diarize(audio_path: str):
    """Run speaker diarization on audio file using waveform input."""
    pipeline = get_pipeline()
    
    # Load audio manually
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Create waveform tensor (channels, samples)
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    
    # Run diarization with waveform input
    diarization = pipeline({
        "waveform": waveform,
        "sample_rate": sr,
    })

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return segments, audio, sr
