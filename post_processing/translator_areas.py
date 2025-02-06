import json
import os
from pathlib import Path


def reference_dict(municipality: str, direction="id_to_name") -> int:
    """
    Args:
    municipality (str): The municipality name or ID
    direction (str): The direction of the translation. Default is "id_to_name". Other option is "name_to_id"

    Returns:
    int: The municipality ID or the municipality name, depending on the direction.
    """
    CUR_DIR = Path(os.getcwd())
    BASE_DIR = CUR_DIR.parent
    # Path to the JSON file
    file_path = os.path.join(CUR_DIR, 'data/init_data/municipalities_area_set.json')

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    municipalities = data["data"]["area_entities"]["display_name"]

    if direction == "name_to_id":
        municipality_indexes = []
        for idx, municipality_value in enumerate(municipalities):
            if municipality_value == municipality:
                municipality_indexes.append(idx)
        if len(municipality_indexes) > 0:
            if len(municipality_indexes) == 1:
                return municipality_indexes[0]
            return municipality_indexes
        elif len(municipality_indexes) == 0:
            raise ValueError(f"Municipality name '{municipality}' not found.")

    elif direction == "id_to_name":
        municipality_name = municipalities[int(municipality)]
        return municipality_name
    else:
        raise ValueError("Invalid direction. Please choose 'name_to_id' or 'id_to_name'")


if __name__ == "__main__":
    # print(reference_dict("Houten", "name_to_id"))
    print(reference_dict(56, direction = "id_to_name"))