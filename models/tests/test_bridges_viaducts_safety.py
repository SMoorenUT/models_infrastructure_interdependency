from movici_simulation_core.core.attributes import GlobalAttributes
from movici_simulation_core.core.schema import AttributeSchema
from movici_simulation_core.data_tracker.data_format import EntityInitDataFormat
from movici_simulation_core.data_tracker.serialization import UpdateDataFormat
from movici_simulation_core.testing.helpers import assert_dataset_dicts_equal
from movici_simulation_core.testing.model_tester import ModelTester
from movici_simulation_core.utils import strategies
import pytest
from additional_models.attributes import (
    LowerReferences,
    UpperReferences,
    LightPresence,
    Lighting,
    AutomaticIncidentDetectionPresence,
    AutomaticIncidentDetection,
)
from additional_models.bridges_viaducts_safety import SafetyModel
import numpy as np


# @pytest.fixture
# def global_schema():
#     schema = AttributeSchema([LowerReferences, UpperReferences, IC_ratio_lower, IC_ratio_upper])

#     schema.use(GlobalAttributes)
#     return schema


# @pytest.fixture(autouse=True)
# def clean_strategies(global_schema):
#     strategies.set(EntityInitDataFormat(schema=global_schema))
#     strategies.set(UpdateDataFormat)
#     yield
#     strategies.reset()


@pytest.fixture
def road_network_name():
    return "road_network"


@pytest.fixture
def model(road_network_name):
    return SafetyModel({"transport_dataset": road_network_name})


@pytest.fixture
def model_tester(model, road_network_name):
    tester = ModelTester(model)
    tester.add_init_data(
        road_network_name,
        {
            "name": road_network_name,
            "general": {
                "enum": {
                    "kpi_status": [
                        "red",
                        "orange",
                        "green",
                        "n/a"
                    ]
                }
            },
            "data": {
                "road_segment_entities": {
                    "id": [1, 2, 3],
                    "reference": ["a", "b", "c"],
                }
            },
        },
    )
    return tester


def test_reference_index(model, model_tester):
    model_tester.initialize()

def test_update_model(model, model_tester, road_network_name):
    model_tester.initialize()
    data, next_time = model_tester.update(
        0,
        {
            road_network_name: {
                "road_segment_entities": {
                    "id": [1, 2, 3],
                    "transport.light_presence": [True, False, False],
                    "transport.passenger_car_unit": [1000, 4000, 1500],
                    "transport.automatic_incident_detection_presence": [True, False, False],
                    "transport.volume_to_capacity_ratio": [0.5, 0.9, 0.2]
                }
            }
        },
    )
    adapter = model_tester.model.model
    assert adapter.model_ready_for_update, adapter.format_uninitialized_attributes()

    assert_dataset_dicts_equal(
        data,
        {
            road_network_name: {
                "road_segment_entities": {
                    "id": [1, 2, 3],
                    "transport.lighting": [2, 0, 2],
                    "transport.automatic_incident_detection": [2, 1, 2]
                }
            }
        },
    )
