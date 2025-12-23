import os
from pyannote.audio import Pipeline

HF_TOKEN = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

def diarize(audio_path: str):
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return segments
