import re
import typing as t
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from movici_simulation_core.core.attributes import GlobalAttributes
from movici_simulation_core.core.schema import AttributeSpec, DataType, AttributeSchema
from movici_simulation_core.data_tracker.arrays import TrackedCSRArray
from movici_simulation_core.data_tracker.data_format import EntityInitDataFormat
from movici_simulation_core.data_tracker.property import AttributeObject
from movici_simulation_core.data_tracker.state import TrackedState

# Change this to point to the correct init data and updates directories
BASE_DIR = Path(__file__).parent

INIT_DATA_DIR = BASE_DIR / "data/init_data"

SCENARIOS = [
    "green",
]
UPDATES_DIR = BASE_DIR / "data/scenarios/" / SCENARIOS[0]

# Change this to point to the correct attribute to track
DATASET = "road_network"
ENTITY_GROUP = "virtual_node_entities"
# "virtual_node_entities" "road_segment_entities"
ATTRIBUTE = AttributeSpec("transport.passenger_demand", data_type=DataType(float, (), True))


def show_values(values):
    """After this script finishes reading the specific values from the updates, this function is
    called with the values as an argument. Change this function to show values in a different way,
    eg: create a plot
    """
    # print(values)
    fig = px.line(x=np.arange(len(values)),
                  y=values, title=ATTRIBUTE.name)
    # fig = go.Figure(
    #     data=go.Scatter(
    #         x=np.arange(len(values)),
    #         y=values,
    #         mode='lines+markers',
    #     ),
    #     title = {ATTRIBUTE.name}
    #
    # )
    fig.show()


def get_schema():
    """Modify this function to add additional attributes that may be required for reading the data"""
    schema = AttributeSchema([ATTRIBUTE])
    schema.use(GlobalAttributes)
    return schema


def get_value(prop: AttributeObject):
    arr = prop._data
    if isinstance(arr, TrackedCSRArray):
        arr = arr.data
    return np.sum(arr)


def sorted_updates(directory):
    pattern = re.compile(r"t(?P<timestamp>\d+)_(?P<iteration>\d+)_(?P<dataset>\w+)\.json")
    all_updates = directory.glob("*.json")
    updates_with_matches = ((path, match) for path in all_updates if (match := pattern.match(path.name)))
    updates_with_metadata = ((path, int(match.groupdict()['timestamp']), int(match.groupdict()['iteration'])) for
                             (path, match) in updates_with_matches)
    return sorted(updates_with_metadata, key=lambda u: (u[1], u[2]))


def main():
    check_dirs()
    state = TrackedState()
    schema = get_schema()
    prop = state.register_attribute(DATASET, ENTITY_GROUP, ATTRIBUTE)
    init_data = read_file(INIT_DATA_DIR.joinpath(DATASET).with_suffix(".json"), schema)
    state.receive_update(init_data, is_initial=True)

    values = [get_value(prop)]
    for file, _, _ in sorted_updates(UPDATES_DIR):
        if upd := read_file(file, schema, check_attribute=True):
            state.receive_update(upd)
            val = get_value(prop)
            values.append(val)
    show_values(values)


def check_dirs():
    print(INIT_DATA_DIR)
    if not INIT_DATA_DIR.is_dir():
        raise ValueError(f"{INIT_DATA_DIR} is not a valid directory")
    if not UPDATES_DIR.is_dir():
        raise ValueError(f"{UPDATES_DIR} is not a valid directory")


def read_file(
        file: Path, schema: AttributeSchema, check_attribute=False
) -> t.Optional[dict]:
    if DATASET not in file.stem:
        return None
    upd = EntityInitDataFormat(schema=schema).load_bytes(file.read_bytes())
    if check_attribute and not has_path(
            upd, (DATASET, ENTITY_GROUP, ATTRIBUTE.component, ATTRIBUTE.name)
    ):
        return None
    return upd


def has_path(d: dict, path: t.Sequence[str]):
    if not path:
        return True
    if path[0] is None:
        return has_path(d, path[1:])
    if path[0] in d:
        return has_path(d[path[0]], path[1:])
    return False


if __name__ == "__main__":
    main()
