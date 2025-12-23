import os
import subprocess
import soundfile as sf

SEGMENT_SECONDS = 300  # 5 minutes per chunk


def get_audio_duration(input_path: str) -> float:
    """Get audio duration in seconds using soundfile."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    
    try:
        info = sf.info(input_path)
        return info.duration
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {str(e)}") from e


def convert_to_wav_16k(input_path: str, output_path: str) -> str:
    """Convert audio to WAV format with 16kHz sample rate (required for models)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg conversion failed: output file not created")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}") from e


def split_audio(input_path: str, output_dir: str):
    """
    Split audio into segments and return list of (segment_path, offset) tuples.
    Each segment is converted to 16kHz WAV format.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)

    # First convert to 16kHz WAV
    converted_path = os.path.join(output_dir, "converted.wav")
    try:
        convert_to_wav_16k(input_path, converted_path)
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio to 16kHz WAV: {str(e)}") from e

    # Get total duration
    try:
        total_duration = get_audio_duration(converted_path)
    except Exception as e:
        # Clean up converted file if duration check fails
        if os.path.exists(converted_path):
            os.remove(converted_path)
        raise RuntimeError(f"Failed to get audio duration: {str(e)}") from e

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

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg segmentation failed: {e.stderr}") from e

    # Collect segments with their time offsets
    segments = []
    segment_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("segment_") and f.endswith(".wav")
    )

    if not segment_files:
        raise RuntimeError("No segments were created after splitting")

    for i, filename in enumerate(segment_files):
        segment_path = os.path.join(output_dir, filename)
        offset = i * SEGMENT_SECONDS
        segments.append((segment_path, offset))

    # Clean up converted file if we split it
    if os.path.exists(converted_path) and len(segments) > 0:
        os.remove(converted_path)

    return segments
