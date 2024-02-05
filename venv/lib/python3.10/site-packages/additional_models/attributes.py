

from movici_simulation_core.core.attribute_spec import AttributeSpec
from movici_simulation_core.core.data_type import DataType
from movici_simulation_core.core.schema import attribute_plugin_from_dict


UpperReferences = AttributeSpec(
    "connection.upper_references",
    DataType(str, csr=True),
)
LowerReferences = AttributeSpec(
    "connection.lower_references",
    DataType(str, csr=True),
)
# IC_ratio = AttributeSpec("transport.capacity_utilization", data_type=float)
IC_ratio_upper = AttributeSpec("transport.capacity_utilization_upper", data_type=float)
IC_ratio_lower = AttributeSpec("transport.capacity_utilization_lower", data_type=float)
AutomaticIncidentDetectionUpper = AttributeSpec(
    "transport.automatic_incident_detection_upper", data_type=int, enum_name="kpi_status"
)
AutomaticIncidentDetectionLower = AttributeSpec(
    "transport.automatic_incident_detection_lower", data_type=int, enum_name="kpi_status"
)
LightingUpper = AttributeSpec("transport.lighting_upper", data_type=int, enum_name="kpi_status")
LightingLower = AttributeSpec("transport.lighting_lower", data_type=int, enum_name="kpi_status")
AutomaticIncidentDetectionPresence = AttributeSpec(
    "transport.automatic_incident_detection_presence", data_type=bool
)
AutomaticIncidentDetection = AttributeSpec(
    "transport.automatic_incident_detection", data_type=int, enum_name="kpi_status"
)  # [red, orange, green, n/a]

LightPresence = AttributeSpec("transport.light_presence", data_type=bool)
Lighting = AttributeSpec("transport.lighting", data_type=int, enum_name="kpi_status")
NoiseLevel = AttributeSpec("noise.level", data_type=float)

BridgeViaductAttributes = attribute_plugin_from_dict(globals())