import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
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
    """
    Upload an audio file for transcription with speaker diarization.
    Returns a job_id and task_id to check the status.
    """
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir)

    audio_path = os.path.join(job_dir, file.filename)

    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Split audio into segments (returns list of (path, offset) tuples)
    segments = split_audio(audio_path, os.path.join(job_dir, "segments"))

    # Create task group with offset information for proper timestamp alignment
    task_group = group(
        transcribe_and_diarize.s(seg_path, i, offset)
        for i, (seg_path, offset) in enumerate(segments)
    )()

    # Save the group result for later retrieval
    task_group.save()

    return {
        "job_id": job_id,
        "task_id": task_group.id,
        "segments": len(segments),
        "status": "processing"
    }


@app.get("/result/{task_id}")
def get_result(task_id: str, _: str = Depends(verify_api_key)):
    """
    Get the result of a transcription job by task_id.
    """
    result = GroupResult.restore(task_id)

    if not result:
        return {"status": "not_found"}

    if not result.ready():
        # Calculate progress
        completed = sum(1 for r in result.results if r.ready())
        total = len(result.results)
        return {
            "status": "processing",
            "progress": f"{completed}/{total}"
        }

    try:
        outputs = result.get()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    outputs.sort(key=lambda x: x["index"])

    # Merge all segments
    merged = []
    for o in outputs:
        merged.extend(o["segments"])

    # Normalize speaker labels globally
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
        "total_speakers": len(speaker_map),
        "segments": merged
    }


@app.delete("/job/{job_id}")
def cleanup_job(job_id: str, _: str = Depends(verify_api_key)):
    """
    Clean up temporary files for a completed job.
    """
    job_dir = os.path.join(TEMP_DIR, job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        return {"status": "cleaned"}
    return {"status": "not_found"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
