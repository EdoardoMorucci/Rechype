import librosa
import soundfile
import typer
import json
import subprocess
import numpy as np

from pathlib import Path
from tqdm import tqdm


def merge_audio(directory_path: str):
    """Merge all audio files in a directory into a unique audio file"""

    # Get all audio files in the directory
    files = librosa.util.find_files(directory_path)

    # Find all the unique sample rates
    srs = set([librosa.get_samplerate(file) for file in files])

    # Get the max of the sample rates
    sr = max(srs)

    # Print info on screen
    print(f"{len(srs)} sample rate detected: {srs}. Defaulting to {sr}.")

    # Load all files, converting them to mono, and resampling them to the max sr
    # we found before
    all_audios = [librosa.load(file, mono=True, sr=sr)[0] for file in tqdm(files)]

    # Concatenate them to create audio file
    merged_audio = np.concatenate(all_audios)

    # Get the ending timestamp for each file
    ends = np.cumsum([len(audio) / sr for audio in all_audios])

    # Get the beginning time stamp
    starts = np.insert(ends[:-1], 0, 0.0)

    # Create json file with this info
    json_file = {
        Path(file).name: {"start": start, "end": end}
        for file, start, end in zip(files, starts, ends)
    }

    return merged_audio, json_file, sr


def main(
    directory_path: str,
    output_json: str = "timestamps.json",
    output_video: str = "merged.mp4",
    save_audio: bool = True,
):
    # Merge the audio files
    merged_audio, json_file, sr = merge_audio(directory_path)

    # Create output audio to apply FFmpeg
    output_audio = Path(output_video).with_suffix(".wav")

    # Write audio to file
    soundfile.write(output_audio, merged_audio, sr)

    # Write json file
    with open(output_json, "w") as file:
        json.dump(json_file, file)

    # Run the Bash script with arguments
    subprocess.run(["bash", "Scripts/waveform.sh", output_audio, output_video])

    if not save_audio:
        # Remove audio file if not required
        output_audio.unlink()


if __name__ == "__main__":
    typer.run(main)
