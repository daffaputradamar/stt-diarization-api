import os
import torch
import soundfile as sf
from pyannote.audio import Pipeline

HF_TOKEN = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Move to GPU if available
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))


def diarize(audio_path: str):
    """Run speaker diarization on audio file using waveform input."""
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
