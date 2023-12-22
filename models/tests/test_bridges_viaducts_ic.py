from movici_simulation_core.core.attributes import GlobalAttributes
from movici_simulation_core.core.schema import AttributeSchema
from movici_simulation_core.data_tracker.data_format import EntityInitDataFormat
from movici_simulation_core.data_tracker.serialization import UpdateDataFormat
from movici_simulation_core.testing.helpers import assert_dataset_dicts_equal
from movici_simulation_core.testing.model_tester import ModelTester
from movici_simulation_core.utils import strategies
import pytest
from additional_models.bridges_viaducts_IC import IC_ratio_lower, IC_ratio_upper, ICModel, LowerReferences, UpperReferences
import numpy as np

@pytest.fixture
def global_schema():
    schema = AttributeSchema(
        [LowerReferences, UpperReferences, IC_ratio_lower, IC_ratio_upper]
    )
    
    schema.use(GlobalAttributes)
    return schema


@pytest.fixture(autouse=True)
def clean_strategies(global_schema):
    strategies.set(EntityInitDataFormat(schema=global_schema))
    strategies.set(UpdateDataFormat)
    yield
    strategies.reset()

@pytest.fixture
def road_network_name():
    return "road_network"
@pytest.fixture
def model(road_network_name):
    return ICModel({"transport_dataset": road_network_name, "bridges_dataset": "bridges"})

@pytest.fixture
def model_tester(model,road_network_name):
    tester = ModelTester(model)
    tester.add_init_data(
        road_network_name,
        {
            "name": road_network_name,
            "data": {
                "road_segment_entities": {
                    "id": [1, 2],
                    "reference": ["a", "b"],
                }
            },
        },
        
    )
    tester.add_init_data(
        "bridges",
        {
            "name": "bridges",
            "general": {
                "special": {
                    "bridge_entities.transport.capacity_utilization_upper": -1,
                    "bridge_entities.transport.capacity_utilization_lower": -1,
                    "bridge_entities.transport.automatic_incident_detection_upper": 3,
                    "bridge_entities.transport.automatic_incident_detection_lower": 3,
                    "bridge_entities.transport.lighting_upper": 3,
                    "bridge_entities.transport.lighting_lower": 3                   
                }
            },
            "data": {
                "bridge_entities": {
                    "id": [10, 20, 30],
                    "connection.upper_references": [["a"], None, ["b"]],
                    "connection.lower_references": [None, ["a", "b"], ["b"]]
                }
            },
        },
    )  
    
    return tester

def test_reference_index(model, model_tester):
    model_tester.initialize()
    np.testing.assert_array_equal(model.upper_indices.data, [0,1])
    np.testing.assert_array_equal(model.upper_indices.row_ptr, [0,1,1,2])
    np.testing.assert_array_equal(model.lower_indices.data, [0,1,1])
    np.testing.assert_array_equal(model.lower_indices.row_ptr, [0,0,2,3])


def test_update_model(model, model_tester, road_network_name):
    model_tester.initialize()
    data, next_time = model_tester.update(0, {
      road_network_name: {
        "road_segment_entities": {
            "id": [1, 2],
            "transport.volume_to_capacity_ratio": [0.5, 0.4],
            # "transport.capacity.hours": [1000, 4000]
            "transport.automatic_incident_detection": [1, 2],
            "transport.lighting": [0, 3]
        }
      }  
    })
    adapter = model_tester.model.model
    assert adapter.model_ready_for_update, adapter.format_uninitialized_attributes()
    
    assert_dataset_dicts_equal(data, {
        "bridges": {
            "bridge_entities": {
                "id": [10, 20, 30],
                'transport.capacity_utilization_lower': [-1.0, 0.5, 0.4], 
                'transport.capacity_utilization_upper': [0.5, -1.0, 0.4],
                'transport.volume_to_capacity_ratio': [0.5, 0.5, 0.4],
                "transport.automatic_incident_detection_upper": [1, 3, 2],
                "transport.automatic_incident_detection_lower": [3, 1, 2],
                "transport.automatic_incident_detection": [1, 1, 2],
                "transport.lighting_upper": [0, 3, 2],
                "transport.lighting_lower": [3, 0, 2],
                "transport.lighting": [0, 0, 2]
            }
        }
    })