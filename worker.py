import whisper
from celery import Celery
from diarization import diarize

celery = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

# Load model sekali (GPU)
model = whisper.load_model("medium").to("cuda")

@celery.task
def transcribe_and_diarize(audio_path: str, index: int):
    # 1. Whisper transcription (with timestamps)
    result = model.transcribe(
        audio_path,
        language="id"
    )

    whisper_segments = result["segments"]

    # 2. Speaker diarization
    speaker_segments = diarize(audio_path)

    # 3. Align transcript ↔ speaker
    aligned = []

    for seg in whisper_segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        speaker = "UNKNOWN"

        for sp in speaker_segments:
            if sp["start"] <= midpoint <= sp["end"]:
                speaker = sp["speaker"]
                break

        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"].strip()
        })

    return {
        "index": index,
        "segments": aligned
    }
