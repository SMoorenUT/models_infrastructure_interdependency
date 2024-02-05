from additional_models.attributes import (
    AutomaticIncidentDetection,
    AutomaticIncidentDetectionPresence,
    LightPresence,
    Lighting,
)
from movici_simulation_core.core.data_type import (
    UNDEFINED,
    DataType,
)
from movici_simulation_core.core.schema import (
    AttributeSpec,
)
from movici_simulation_core.core.arrays import TrackedCSRArray
from movici_simulation_core.csr import get_row, row_wise_min
from movici_simulation_core.core.entity_group import (
    EntityGroup,
)
from movici_simulation_core.core.attribute import (
    OPT,
    field,
    INIT,
    PUB,
    SUB,
)
from movici_simulation_core.base_models.tracked_model import (
    TrackedModel,
)
from movici_simulation_core.core.state import (
    TrackedState,
)
from movici_simulation_core.attributes import (
    Reference,
)
from movici_simulation_core.models.common.attributes import (
    Transport_VolumeToCapacityRatio,
    Transport_PassengerCarUnit,
)
from movici_simulation_core.validate import validate_and_process
import numpy as np


class RoadSegmentEntities(EntityGroup, name="road_segment_entities"):
    IC_ratio = field(Transport_VolumeToCapacityRatio, flags=SUB)
    intensity = field(Transport_PassengerCarUnit, flags=SUB)
    automatic_incident_detection = field(AutomaticIncidentDetection, flags=PUB)
    automatic_incident_detection_presence = field(AutomaticIncidentDetectionPresence, flags=OPT)
    light_presence = field(LightPresence, flags=OPT)
    lighting = field(Lighting, flags=PUB)


config = {"name": "safety_roads", "type": "safety_model", "transport_dataset": "road_network"}


class SafetyModel(TrackedModel, name="safety_model"):
    def __init__(self, model_config: dict):
        validate_and_process(model_config, self.model_config_schema())
        super().__init__(model_config)

    def setup(self, state: TrackedState, **_):
        self.roads = state.register_entity_group(
            self.config["transport_dataset"],
            RoadSegmentEntities,
        )

    def model_config_schema(self):
        return {
            "type": "object",
            "required": ["transport_dataset"],
            "properties": {
                "transport_dataset": {
                    "type": "string",
                    "movici.type": "dataset",
                    "movici.datasetType": "transport_network",
                },
            },
        }

    def initialize(self, **_):
        pass

    def update(self, **_):
        # lighting "green" if lighting present. "red" if lighting absent AND intensity > 3000. "special" if lighting_presence is undefined
        defineds = ~self.roads.light_presence.is_undefined()
        self.roads.lighting[defineds] = self.roads.light_presence[defineds] | (
            self.roads.intensity[defineds] <= 3000
        )
        self.roads.lighting[~defineds] = 3  # n/a

        self.roads.lighting[self.roads.lighting.array == 1] = 2  # true -> green
        # AID "green" if AID present or IC <= 0.8. "red" if AID absent AND intensity IC > 0.8. "special" if AID presence is undefined
        defineds = ~self.roads.automatic_incident_detection_presence.is_undefined()
        self.roads.automatic_incident_detection[
            defineds
        ] = self.roads.automatic_incident_detection_presence[defineds] | (
            self.roads.IC_ratio[defineds] <= 0.8
        )
        self.roads.automatic_incident_detection[~defineds] = 3  # NA / special
        self.roads.automatic_incident_detection[
            self.roads.automatic_incident_detection.array == 1
        ] = 2  # true -> green
        self.roads.automatic_incident_detection[
            self.roads.automatic_incident_detection.array == 0
        ] = 1  # false -> orange

    @classmethod
    def get_schema_attributes(cls):
        return []
