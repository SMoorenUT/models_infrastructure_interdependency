import json
import os


def reference_dict(bridge_number: int, direction = "id_to_reference") -> int:
    """
    This function creates a dictionary with the bridge number as the key and the reference as the value

    Args:
    bridge_number (int): The bridge number
    direction (str): The direction of the translation. Default is "id_to_reference". Other option is "reference_to_id"

    Returns:
    int: The bridge reference or the bridge number, depending on the direction. 
    """
    BASE_DIR = os.getcwd()
    # Path to the JSON file
    file_path = os.path.join(BASE_DIR, 'data/init_data/bridges.json')

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    bridge_references = list(data["data"]["bridge_entities"]["reference"])

    if direction == "id_to_reference":
        bridge_id = bridge_references[bridge_number]
        return bridge_id
    elif direction == "reference_to_id":
        bridge_reference = bridge_references.index(str(bridge_number))
        return bridge_reference
    else:
        raise ValueError("Invalid direction. Please choose 'id_to_reference' or 'reference_to_id'")
     

if __name__ == "__main__":
    print(reference_dict(885, "id_to_reference"))