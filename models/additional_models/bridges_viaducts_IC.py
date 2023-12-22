from additional_models.attributes import (
    AutomaticIncidentDetection,
    AutomaticIncidentDetectionLower,
    AutomaticIncidentDetectionUpper,
    IC_ratio_lower,
    IC_ratio_upper,
    Lighting,
    LightingLower,
    LightingUpper,
    LowerReferences,
    UpperReferences,
)
from movici_simulation_core.core.data_type import (
    UNDEFINED,
    DataType,
)
from movici_simulation_core.core.schema import (
    AttributeSpec,
)
from movici_simulation_core.data_tracker.arrays import TrackedCSRArray
from movici_simulation_core.data_tracker.csr_helpers import get_row, row_wise_max, row_wise_min
from movici_simulation_core.data_tracker.entity_group import (
    EntityGroup,
)
from movici_simulation_core.data_tracker.attribute import (
    OPT,
    field,
    INIT,
    PUB,
    SUB,
)
from movici_simulation_core.base_models.tracked_model import (
    TrackedModel,
)
from movici_simulation_core.data_tracker.state import (
    TrackedState,
)
from movici_simulation_core.core.attributes import (
    Reference,
)
from movici_simulation_core.models.common.attributes import (
    Transport_Capacity_Hours,
    Transport_VolumeToCapacityRatio,
)
from movici_simulation_core.utils.validate import validate_and_process
import numpy as np
import numba


class BridgeEntities(EntityGroup, name="bridge_entities"):
    IC_ratio = field(Transport_VolumeToCapacityRatio, flags=PUB)
    upper_references = field(UpperReferences, flags=OPT)
    lower_references = field(LowerReferences, flags=OPT)
    ic_ratio_upper = field(IC_ratio_upper, flags=PUB)
    ic_ratio_lower = field(IC_ratio_lower, flags=PUB)

    automatic_incident_detection_upper = field(AutomaticIncidentDetectionUpper, flags=PUB)
    automatic_incident_detection_lower = field(AutomaticIncidentDetectionLower, flags=PUB)
    automatic_incident_detection = field(AutomaticIncidentDetection, flags=PUB)

    lighting_upper = field(LightingUpper, flags=PUB)
    lighting_lower = field(LightingLower, flags=PUB)
    lighting = field(Lighting, flags=PUB)


class RoadSegmentEntities(EntityGroup, name="road_segment_entities"):
    reference = field(Reference, flags=INIT)
    IC_ratio = field(Transport_VolumeToCapacityRatio, flags=SUB)
    lighting = field(Lighting, flags=SUB)
    automatic_incident_detection = field(AutomaticIncidentDetection, flags=SUB)


config = {
    "name": "ic_model",
    "type": "intensity_capacity",
    "transport_dataset": "road_network",
    "bridges_dataset": "bridges",
    "capacity_threshold": 2000,
}


class ICModel(TrackedModel, name="intensity_capacity"):
    def __init__(self, model_config: dict):
        validate_and_process(model_config, self.model_config_schema())
        super().__init__(model_config)

    def setup(self, state: TrackedState, **_):
        self.bridges = state.register_entity_group(
            self.config["bridges_dataset"],
            BridgeEntities,
        )
        self.roads = state.register_entity_group(
            self.config["transport_dataset"],
            RoadSegmentEntities,
        )

    def model_config_schema(self):
        return {
            "type": "object",
            "required": ["transport_dataset", "bridges_dataset"],
            "properties": {
                "transport_dataset": {
                    "type": "string",
                    "movici.type": "dataset",
                    "movici.datasetType": "transport_network",
                },
                "bridges_dataset": {
                    "type": "string",
                    "movici.type": "dataset",
                    "movici.datasetType": "bridges",
                },
                "capacity_threshold": {"type": "number", "minimum": 0},
            },
        }

    def initialize(self, **_):
        # [["1", "2"], [None], ['2']]
        # [1,2,None, 2] [0, 2,3,4]

        # [1,2,2] [0,2,2,3]
        self.reference_index = {ref: idx for idx, ref in enumerate(self.roads.reference.array)}
        self.reference_index[UNDEFINED[str]] = -1  # Catch undefined
        upper_indices = np.array(
            [
                self.reference_index[ref] for ref in self.bridges.upper_references.csr.data
            ]  # [["1", "2"], None, []] -> [false, true, false] -> [1]
        )
        upper_undefineds = np.flatnonzero(self.bridges.upper_references.is_undefined())
        self.upper_indices = TrackedCSRArray(
            upper_indices, self.bridges.upper_references.csr.row_ptr.copy()
        )
        self.upper_indices.update(
            TrackedCSRArray([], np.zeros((len(upper_undefineds) + 1), dtype=int)), upper_undefineds
        )  # [[], [], []] -> data [], rowptr [0,0,0,0]

        lower_indices = np.array(
            [
                self.reference_index[ref] for ref in self.bridges.lower_references.csr.data
            ]  # [["1", "2"], None, []] -> [false, true, false] -> [1]
        )
        lower_undefineds = np.flatnonzero(self.bridges.lower_references.is_undefined())
        self.lower_indices = TrackedCSRArray(
            lower_indices, self.bridges.lower_references.csr.row_ptr.copy()
        )
        self.lower_indices.update(
            TrackedCSRArray([], np.zeros((len(lower_undefineds) + 1), dtype=int)), lower_undefineds
        )  # [[], [], []] -> data [], rowptr [0,0,0,0]

    def update(self, **_):
        self.bridges.ic_ratio_upper[:] = row_wise_max(
            self.roads.IC_ratio[
                self.upper_indices.data
            ],  # ic ratio  [1,2,3], upper_indicies = [[0],[],[1]] -> [[1], [1, 3], [2]]
            self.upper_indices.row_ptr,
            empty_row=self.bridges.ic_ratio_upper.options.special,
        )
        self.bridges.ic_ratio_lower[:] = row_wise_max(
            self.roads.IC_ratio[self.lower_indices.data],
            self.lower_indices.row_ptr,
            self.bridges.ic_ratio_lower.options.special,
        )

        # test code for updates with conditions
        # threshold = self.config.get("capacity_threshold", 2000)
        # self.bridges.ic_ratio_upper[:] = ic_roads_to_bridge(
        #     self.roads.IC_ratio[
        #         self.upper_indices.data
        #     ],  # ic ratio  [1,2,3], upper_indicies = [[0],[0,2],[1]] -> [[1], [1, 3], [2]]
        #     self.roads.capacity[self.upper_indices.data],
        #     self.upper_indices.row_ptr,
        #     threshold=threshold,
        #     special_value=self.bridges.ic_ratio_upper.options.special,
        # )
        # self.bridges.ic_ratio_lower[:] = ic_roads_to_bridge(
        #     self.roads.IC_ratio[self.lower_indices.data],
        #     self.roads.capacity[self.lower_indices.data],
        #     self.lower_indices.row_ptr,
        #     threshold=threshold,
        #     special_value=self.bridges.ic_ratio_lower.options.special,
        # )
        # use the fact that any valid value is greater than special value. We can now just
        # take the maximum of both upper and lower to calculate the combined IC-ratio
        self.bridges.IC_ratio[:] = np.maximum(
            self.bridges.ic_ratio_lower.array,
            self.bridges.ic_ratio_upper.array,
        )

        self.bridges.automatic_incident_detection_upper[:] = row_wise_min(
            self.roads.automatic_incident_detection[self.upper_indices.data],
            self.upper_indices.row_ptr,
            empty_row=self.bridges.automatic_incident_detection_upper.options.special,
        )
        self.bridges.automatic_incident_detection_lower[:] = row_wise_min(
            self.roads.automatic_incident_detection[self.lower_indices.data],
            self.lower_indices.row_ptr,
            empty_row=self.bridges.automatic_incident_detection_lower.options.special,
        )
        self.bridges.lighting_upper[:] = row_wise_min(
            self.roads.lighting[self.upper_indices.data],
            self.upper_indices.row_ptr,
            empty_row=self.bridges.lighting_upper.options.special,
        )
        self.bridges.lighting_lower[:] = row_wise_min(
            self.roads.lighting[self.lower_indices.data],
            self.lower_indices.row_ptr,
            empty_row=self.bridges.lighting_lower.options.special,
        )
        self.bridges.automatic_incident_detection[:] = np.minimum(
            self.bridges.automatic_incident_detection_upper.array,
            self.bridges.automatic_incident_detection_lower.array,
        )
        self.bridges.lighting[:] = np.minimum(
            self.bridges.lighting_upper.array,
            self.bridges.lighting_lower.array,
        )

    @classmethod
    def get_schema_attributes(cls):
        return [UpperReferences, LowerReferences, IC_ratio_lower, IC_ratio_upper]


# [[1,2], [1], []] -> [2, 1, -1]
# -> [1,1, -1]

# copy from csr_helpers.py in case of custom reduce logic
@numba.njit(cache=True)
def ic_roads_to_bridge(data, capacities, row_ptr, threshold, special_value):
    """
    data: ic ratio projected to csr upper indices
    capacities: capacity projected to csr upper indices
    """
    n_rows = row_ptr.size - 1
    rv = np.zeros((n_rows,), dtype=data.dtype)
    for i in range(n_rows):
        ic_ratio_row = get_row(
            data, row_ptr, i
        )  # [[1,2,], [3]] -> [1,2,3], [0,2,3] get_row([1,2,3], [0,2,3], 0) ->  -> [1,2]
        capacities_row = get_row(capacities, row_ptr, i)
        rv[i] = max_without_exits(ic_ratio_row, capacities_row, threshold, special_value)
    return rv


@numba.njit(cache=True)
def max_without_exits(ic_ratios, capacities, threshold, special_value):
    """
    ic ratio for all roads on a single bridge
    capacities for all roads on a single bridge
    """
    if len(ic_ratios) == 0:
        return special_value
    if len(ic_ratios) == 1:
        return ic_ratios[0]

    below_threshold_count = np.sum(capacities < threshold)
    should_filter = 0 < below_threshold_count < len(capacities)
    rv = 0
    for (icr, cap) in zip(ic_ratios, capacities):
        # if should filter: check threshold, else always use value for max calculation
        if cap >= threshold:
            rv = max(rv, icr)
    return rv
