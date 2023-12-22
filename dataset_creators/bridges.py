import json
from pathlib import Path
from movici_simulation_core.preprocessing.dataset_creator import create_dataset

CURRENT_DIR = Path(__file__).parent
DATASET_NAME = "bridges"

config = {
    "__meta__": {"crs": "EPSG:28992"},
    "__sources__": {
        "bridges": {
            "source_type": "file",
            "path": str(CURRENT_DIR.parent.joinpath("B&V+rs_pivoted.shp")),
        }
    },
    "name": DATASET_NAME,  # lowercase en underscore
    "display_name": "Bridges",  # anything goes
    "type": "bridges",
    "version": 4,
    "general": {
        "special": {
            "bridge_entities.transport.capacity_utilization_upper": -1,
            "bridge_entities.transport.capacity_utilization_lower": -1,
            "bridge_entities.transport.automatic_incident_detection_upper": 3,
            "bridge_entities.transport.automatic_incident_detection_lower": 3,
            "bridge_entities.transport.lighting_upper": 3,
            "bridge_entities.transport.lighting_lower": 3,
        },
        "enum": {
            "kpi_status": ["red", "orange", "green", "n/a"],
        },
    },
    "data": {
        "bridge_entities": {
            "__meta__": {"source": "bridges", "geometry": "points"},
            "reference": {"property": "bo_id", "loaders": ["str"]},
            "connection.upper_references": {
                "property": "FID_upper",
                "loaders": [
                    "csv",
                ],
            },
            "connection.lower_references": {
                "property": "FID_lower",
                "loaders": [
                    "csv",
                ],
            },
        }
    },
}

Path(DATASET_NAME).with_suffix(".json").write_text(json.dumps(create_dataset(config), indent=2))
