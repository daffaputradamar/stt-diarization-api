import os
import subprocess

SEGMENT_SECONDS = 300  # 5 minutes per chunk


def get_audio_duration(input_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def convert_to_wav_16k(input_path: str, output_path: str) -> str:
    """Convert audio to WAV format with 16kHz sample rate (required for models)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def split_audio(input_path: str, output_dir: str):
    """
    Split audio into segments and return list of (segment_path, offset) tuples.
    Each segment is converted to 16kHz WAV format.
    """
    os.makedirs(output_dir, exist_ok=True)

    # First convert to 16kHz WAV
    converted_path = os.path.join(output_dir, "converted.wav")
    convert_to_wav_16k(input_path, converted_path)

    # Get total duration
    total_duration = get_audio_duration(converted_path)

    # If audio is shorter than segment length, return single file
    if total_duration <= SEGMENT_SECONDS:
        return [(converted_path, 0.0)]

    # Split into segments
    output_pattern = os.path.join(output_dir, "segment_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", converted_path,
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_pattern
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # Collect segments with their time offsets
    segments = []
    segment_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("segment_") and f.endswith(".wav")
    )

    for i, filename in enumerate(segment_files):
        segment_path = os.path.join(output_dir, filename)
        offset = i * SEGMENT_SECONDS
        segments.append((segment_path, offset))

    # Clean up converted file if we split it
    if os.path.exists(converted_path) and len(segments) > 0:
        os.remove(converted_path)

    return segments
