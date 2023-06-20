import json
import typer
import librosa
import soundfile as sf
import numpy as np
import polars as pl
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from pathlib import Path


def convert_label(label_directory: str, seconds: int = 3):
    """Convert the labels from the json files to a multi-label format"""

    # Find all label files
    files = glob(label_directory + "/*")

    # Store a list with all labels read from the json files
    labels = []

    # Loop over all the files
    for file in tqdm(files):
        # Open the file
        with open(file) as f:
            # Read the json
            data = json.load(f)

        # Get the name of the audio file
        name = Path(data["task"]["data"]["audio"]).name

        # Add the labels to the list
        labels.extend(
            [
                {
                    "file": name,
                    "start": label["value"]["start"],
                    "end": label["value"]["end"],
                    "class_name": label["value"]["labels"][0],
                }
                for label in data["result"]
            ]
        )

    # Turn labels into a Polars dataframe
    labels = pl.DataFrame(labels)

    # Get the number of unique labels
    n_labels = labels["class_name"].n_unique()

    # Create an encoding for the labels
    encoding = {label: i for i, label in enumerate(labels["class_name"].unique())}

    # Split into train, validation and test sets
    train_validation_samples, test_samples = train_test_split(
        labels["file"].unique(), test_size=0.2, random_state=42
    )
    train_samples, validation_samples = train_test_split(
        train_validation_samples, test_size=0.2, random_state=42
    )

    # Assign split
    labels = labels.with_columns(
        pl.col("file")
        .apply(
            lambda file: "Train"
            if file in train_samples
            else "Validation"
            if file in validation_samples
            else "Test"
        )
        .alias("split")
    )

    # Sort
    labels.sort(by=["file", "start", "end"])

    # Create intervals from timestamps
    labels = (
        labels.with_columns(
            pl.arange(pl.col("start"), pl.col("end") + 1).alias("range")
        )
        .explode("range")
        .sort(["file", "range"])
    )

    # Convert seconds to microseconds and then to datetime
    labels = labels.with_columns((pl.col("range") * 1e6).cast(pl.Datetime))

    # Group by 3 second intervals, keeping track of the class names and the split
    labels = labels.groupby_dynamic(
        "range", every=f"{seconds}s", by="file", include_boundaries=True
    ).agg(
        [
            pl.col("class_name").unique().sort(),
            pl.col("split").first(),
            pl.col("end").max(),
        ]
    )

    # Remove intervals that are not fully contained in the audio file
    labels = labels.filter(pl.col("_upper_boundary") < pl.col("end") * 1e6)

    # Convert to start and end columns in seconds
    labels = labels.with_columns(
        [
            (pl.col("_lower_boundary").cast(pl.Int32) / 1e6).alias("start"),
            (pl.col("_upper_boundary").cast(pl.Int32) / 1e6).alias("end"),
        ]
    ).select(["file", "start", "end", "class_name", "split"])

    # Create multi-label (convert to list at the end to avoid issues with object type being written to parquet)
    labels = labels.with_columns(
        pl.col("class_name")
        .apply(
            lambda x: np.eye(n_labels)[[encoding[label] for label in x]]
            .sum(axis=0)
            .astype(int)
            .tolist()
        )
        .alias("class")
    )

    # Create the names of the new files
    labels = labels.rename({"file": "original_file"}).with_columns(
        pl.concat_str(
            [
                labels.with_row_count()["row_nr"],
                pl.col("original_file"),
            ],
            separator="_",
        ).alias("file"),
    )

    return labels


def trim_single_audio(
    original_path: str, target_path: str, start: float, end: float, length: int = 3
):
    """Read audio file, trim it to desired lengths, and re-save it"""

    # Read from file
    waveform, sr = librosa.load(
        original_path, sr=None, mono=True, offset=start, duration=end - start
    )

    # Ensure the length is correct, by padding it with zeros on the right
    waveform = np.pad(waveform, (0, int(sr * length) - len(waveform)), mode="constant")

    # Write to file
    sf.write(target_path, waveform, sr)

    return


def trim_audios(
    labels: pl.DataFrame,
    original_directory: str,
    target_directory: str,
    length: int = 3,
):
    """Trim the audio files to the desired lengths"""

    # Create desired dataframe for the task
    df = labels.with_columns(
        [
            pl.concat_str(
                [pl.lit(original_directory), pl.col("original_file")], separator="/"
            ).alias("original_path"),
            pl.concat_str(
                [pl.lit(target_directory), pl.col("file")], separator="/"
            ).alias("target_path"),
            pl.lit(length).alias("length"),
        ]
    ).select(["original_path", "target_path", "start", "end", "length"])

    # Init pool
    with mp.Pool(mp.cpu_count()) as pool:
        # Trim the audio files
        pool.starmap(trim_single_audio, tqdm(df.to_numpy()))

    return


def main(
    label_directory: str,
    original_audio_directory: str,
    target_audio_directory: str,
    length: int = 3,
):
    # Create labels
    labels = convert_label(label_directory=label_directory, seconds=length)

    # Write labels to file
    label_path = Path(original_audio_directory).parent / "labels.parquet"
    labels.write_parquet(label_path)

    # Fix audio files
    trim_audios(
        labels=labels,
        original_directory=original_audio_directory,
        target_directory=target_audio_directory,
        length=length,
    )

    return


if __name__ == "__main__":
    typer.run(main)
