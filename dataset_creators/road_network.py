from inspect import getdoc
import json
from pathlib import Path
from movici_simulation_core.preprocessing.dataset_creator import create_dataset, NumpyDataSource
import numpy as np

CURRENT_DIR = Path(__file__).parent
DATASET_NAME = "road_network"
GEOJSON_PATH = CURRENT_DIR.parent / "data/road_geojson_data"
OLD_ROAD_NETWORK_FILE = "road_network_old.json"


def get_od_matrix_source(virtual_node_data):
    fields = [
        "transport.passenger_demand",
        "transport.international_cargo_demand",
        "transport.domestic_cargo_demand",
    ]
    return {field: np.asarray(virtual_node_data[field]) for field in fields}


config = {
    "__meta__": {"crs": "EPSG:28992"},
    "__sources__": {
        "road_segments": str(GEOJSON_PATH.joinpath("road_segments.geojson")),
        "road_nodes": str(GEOJSON_PATH.joinpath("road_transport_nodes.geojson")),
        "virtual_links": str(GEOJSON_PATH.joinpath("road_virtual_links.geojson")),
        "virtual_nodes": str(GEOJSON_PATH.joinpath("road_virtual_nodes.geojson")),
    },
    "name": DATASET_NAME,  # lowercase en underscore
    "display_name": "Road Network",  # anything goes
    "type": "transport_network",
    "version": 4,
    "general": {
        "special": {
            "road_segment_entities.transport.passenger_flow": -1,
            "road_segment_entities.transport.cargo_flow": -1,
        },
        "enum": {
            "kpi_status": ["red", "orange", "green", "n/a"],
        },
    },
    "data": {
        "road_segment_entities": {
            "__meta__": {"source": "road_segments", "geometry": "lines"},
            "reference": {"property": "fid", "loaders": ["str"]},
            "display_name": {
                "property": "display_name",
            },
            "topology.from_node_id": {
                "property": "source",
                "id_link": {
                    "entity_group": "transport_node_entities",
                    "property": "node_id",
                },
            },
            "topology.to_node_id": {
                "property": "target",
                "id_link": {
                    "entity_group": "transport_node_entities",
                    "property": "node_id",
                },
            },
            "transport.layout": {
                "property": "layout",
                "loaders": ["csv", "int"],
            },
            "transport.capacity.hours": {
                "property": "capacity_per_lane",
                "loaders": ["float"],
            },
            "transport.max_speed": {"property": "max_speed"},
            "shape.length": {
                "property": "length",
                "loaders": ["float"],
            },
            "transport.automatic_incident_detection_presence": {
                "property": "pfAID_AID?",
                "loaders": ["bool"],
            },
            "transport.light_presence": {
                "property": "pfVerl_VER",
                "loaders": ["bool"],
            }
        },
        "transport_node_entities": {
            "__meta__": {"source": "road_nodes", "geometry": "points"},
            "reference": {
                "property": "fid",
                "loaders": ["str"],
            },
        },
        "virtual_node_entities": {
            "__meta__": {"source": "virtual_nodes", "geometry": "points"},
            "reference": {
                "property": "fid",
                "loaders": ["str"],
            },
            "display_name": {"property": "display_name"},
            "transport.passenger_demand": {
                "property": "transport.passenger_demand",
                "source": "od_matrix",
            },
            "transport.domestic_cargo_demand": {
                "property": "transport.domestic_cargo_demand",
                "source": "od_matrix",
            },
            "transport.international_cargo_demand": {
                "property": "transport.international_cargo_demand",
                "source": "od_matrix",
            },
        },
        "virtual_link_entities": {
            "__meta__": {"source": "virtual_links", "geometry": "lines"},
            "topology.from_node_id": {
                "property": "source_transport_node_id",
                "id_link": [
                    {
                        "entity_group": "transport_node_entities",
                        "property": "node_id",
                    },
                    {
                        "entity_group": "virtual_node_entities",
                        "property": "node_id",
                    },
                ],
            },
            "topology.to_node_id": {
                "property": "target",
                "id_link": [
                    {
                        "entity_group": "transport_node_entities",
                        "property": "node_id",
                    },
                    {
                        "entity_group": "virtual_node_entities",
                        "property": "node_id",
                    },
                ],
            },
        },
    },
}
road_network = json.loads((GEOJSON_PATH / OLD_ROAD_NETWORK_FILE).read_text())
additional_sources = {
    "od_matrix": NumpyDataSource(
        get_od_matrix_source(road_network["data"]["virtual_node_entities"])
    )
}
result = create_dataset(config, sources=additional_sources)
Path(DATASET_NAME).with_suffix(".json").write_text(json.dumps(result, indent=2))
