import json
import typer
import warnings

import pandas as pd

from datetime import timedelta
from typing import List

# Disable the warning
warnings.simplefilter("ignore")


def excel_from_json(
    json_path: str,
    excel_path: str,
    exclude_tags: List[str],
    confidence_threshold: float = 0.25,
):
    """Converts json file to excel file"""
    # Read json file
    with open(json_path) as file:
        json_data = json.load(file)

    with pd.ExcelWriter(excel_path) as writer:
        # Iterate over json file
        for name, items in json_data.items():
            # Initialize dataframe
            df = pd.DataFrame(columns=["start", "end", "tag", "confidence"])

            # Loop over items for a given file
            for item in items:
                # Convert start and end times to datetime objects
                start = str(timedelta(seconds=item["start"]))
                end = str(timedelta(seconds=item["end"]))

                # Get tag name and confidence (only the first one)
                tag = item["tags"][0]["tag"]
                confidence = item["tags"][0]["confidence"]

                if tag in exclude_tags:
                    continue

                if confidence < confidence_threshold:
                    continue

                # Concatenate to dataframe
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "start": start,
                                "end": end,
                                "tag": tag,
                                "confidence": confidence,
                            },
                            index=range(len(df), len(df) + 1),
                        ),
                    ]
                )

            # Save dataframe to excel
            df.to_excel(writer, sheet_name=name, index=False)


if __name__ == "__main__":
    typer.run(excel_from_json)
