import os
import uuid
from fastapi import FastAPI, UploadFile, File, Depends
from celery import group
from celery.result import GroupResult

from auth import verify_api_key
from audio_utils import split_audio
from worker import transcribe_and_diarize

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(title="Speech-to-Text + Speaker Diarization API")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key)
):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir)

    audio_path = os.path.join(job_dir, file.filename)

    with open(audio_path, "wb") as f:
        f.write(await file.read())

    segments = split_audio(audio_path, os.path.join(job_dir, "segments"))

    task_group = group(
        transcribe_and_diarize.s(seg, i)
        for i, seg in enumerate(segments)
    )()

    return {
        "job_id": job_id,
        "task_id": task_group.id,
        "segments": len(segments),
        "status": "processing"
    }


@app.get("/result/{task_id}")
def get_result(task_id: str, _: str = Depends(verify_api_key)):
    result = GroupResult.restore(task_id)

    if not result:
        return {"status": "not_found"}

    if not result.ready():
        return {"status": "processing"}

    outputs = result.get()
    outputs.sort(key=lambda x: x["index"])

    merged = []
    for o in outputs:
        merged.extend(o["segments"])

    # Normalize speaker labels (global)
    speaker_map = {}
    counter = 1

    for seg in merged:
        sp = seg["speaker"]
        if sp not in speaker_map:
            speaker_map[sp] = f"SPEAKER_{counter}"
            counter += 1
        seg["speaker"] = speaker_map[sp]

    return {
        "status": "done",
        "segments": merged
    }
