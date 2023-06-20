import torch
import random
import librosa
import torchaudio
import os
import typer
import pandas as pd
import numpy as np

from pathlib import Path
from copy import deepcopy
from itertools import chain, combinations


def augment(dataset_path: str, audiofile: str, augmentation_rate: int):
    # Get name
    filename = Path(audiofile).stem

    # Values to choose from
    EFFECT_VALUES = {
        "lowpass": ["4000", "5000", "6000"],
        "highpass": ["200", "300", "100"],
        "pitchup": ["100", "200", "150"],
        "pitchdown": ["-100", "-200", "-150"],
        "noise": [0.005, 0.0001, 0.0004],
    }

    # Load waveform and sample_rate
    waveform, sample_rate = librosa.load(audiofile, mono=True, sr=None)

    # Convert waveform to torch tensor
    waveform = torch.tensor(waveform)

    # List all possible effects
    effects = ["lowpass", "highpass", "reverb", "pitchup", "pitchdown", "noise"]

    # Generate all possible combinations of effects
    effect_combinations = list(
        chain.from_iterable(
            combinations(effects, r) for r in range(1, len(effects) + 1)
        )
    )

    if augmentation_rate <= len(effect_combinations):
        chosen_effects = random.choices(effect_combinations, k=augmentation_rate)

    else:
        chosen_effects = effect_combinations

    # Init
    augmented_names = []

    for i, combination in enumerate(chosen_effects):
        # Duplicate waveform
        modified_waveform = deepcopy(waveform)

        # Create stereo track to apply effects
        if len(modified_waveform.shape) == 1:
            modified_waveform = torch.vstack([modified_waveform, modified_waveform])

        # Turn tuple into list
        combination = list(combination)

        if "noise" in combination:
            # Apply noise
            value = random.choice(EFFECT_VALUES["noise"])
            white_noise = np.random.rand(2, len(waveform)) * value
            modified_waveform = modified_waveform + white_noise
            modified_waveform = modified_waveform.to(torch.float32)
            # Reove noise from list
            combination.remove("noise")

        if len(combination):
            # Init list of effects to apply
            effects_to_apply = []

            for j, effect in enumerate(combination):
                if effect in ["lowpass", "highpass"]:
                    # Get value
                    value = random.choice(EFFECT_VALUES[effect])
                    # Create command
                    command = [effect, "-1", value]
                elif effect in ["pitchup", "pitchdown"]:
                    # Get value
                    value = random.choice(EFFECT_VALUES[effect])
                    # Create command
                    command = ["pitch", value]
                elif effect == "reverb":
                    command = [effect, "-w"]

                effects_to_apply.append(command)

                if effect in ["pitchup", "pitchdown"]:
                    effects_to_apply.append(["rate", f"{sample_rate}"])

            # Apply effects
            modified_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                modified_waveform, sample_rate, effects_to_apply, channels_first=True
            )

        else:
            j = 0

        if modified_waveform.shape[1] > len(waveform):
            modified_waveform = modified_waveform[:, : len(waveform)]

        # Create saving path
        saving_path = os.path.join(dataset_path, "Audio", f"aug_{i}_{j}_{filename}.wav")
        torchaudio.save(saving_path, modified_waveform, sample_rate)

        # Add
        augmented_names.append(f"aug_{i}_{j}_{filename}.wav")

    return augmented_names


def augment_dataset(dataset_path: str):
    # Read df from labels file
    df = pd.read_csv(os.path.join(dataset_path, "labels.csv"))

    # Restrict to train dataset
    train_df = df[df["split"] == "Train"]

    # Init df with augmented files
    augmented_df = pd.DataFrame(columns=df.columns)

    # Count number of appearaces per class
    counting = df["class"].value_counts().sort_values(ascending=False)

    # Compute multiplier needed for dataset to be perfectly balanced
    multipliers = (counting.max() / counting).round().astype(int)

    for label, multiplier in zip(multipliers.index, multipliers.values):
        # Restrict to relevant class
        restricted_df = train_df[train_df["class"] == label]

        # Get class name
        class_name = list(restricted_df["class_name"].unique())

        if multiplier == 1:
            continue

        else:
            paths = restricted_df["file"].apply(
                lambda x: os.path.join(dataset_path, "Audio", x)
            )

            for path in paths:
                # Get augmented names
                augmented_names = augment(dataset_path, path, multiplier)

                # Number of augmented names
                n = len(augmented_names)

                # DF to be added
                add_df = pd.DataFrame(
                    {
                        "file": augmented_names,
                        "class": [label] * n,
                        "split": ["Train"] * n,
                        "class_name": class_name * n,
                    }
                )

                # Concatenate dfs
                augmented_df = pd.concat([augmented_df, add_df])

    # Concatenate everything
    augmented_df = pd.concat([df, augmented_df])

    # Save to file
    augmented_df.to_csv(os.path.join(dataset_path, "augmented_labels.csv"))


if __name__ == "__main__":
    typer.run(augment_dataset)
