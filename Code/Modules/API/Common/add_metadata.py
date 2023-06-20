from typing import List, Dict
from copy import deepcopy


def add_metadata(ml_results: Dict, metadata_list=List[Dict]) -> Dict:
    """Add a list of metadata (each metadata should be a dictionary)
    to the Machine Learning results. Return a JSON file"""

    # Create copy of results
    results = deepcopy(ml_results)

    # Update the dictionary
    for metadata in metadata_list:
        results.update(metadata)

    return results
