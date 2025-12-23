import os
import subprocess

SEGMENT_SECONDS = 300  # 5 menit (ideal untuk diarization)

def split_audio(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "segment_%03d.wav")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-c", "copy",
        output_pattern
    ]

    subprocess.run(cmd, check=True)

    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".wav")
    )
