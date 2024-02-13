#! /usr/bin/env python3

import dataclasses
import functools
import json
import pathlib
import sys
import typing as t
import datetime as dt

NAME = "missed"
DISPLAY_NAME = NAME
SCENARIO_PARAMETERS = f"{NAME}_interpolated"
RELAXATION_FACTOR = 0.7
MAX_ITERATIONS = 50
PASSENGER_PEAK_MULTIPLIER = 4 * 280
CARGO_PEAK_MULTIPLIER = 9 * 280
NOW_STRING = dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

prefixes = {
    "roads": "road",
    "waterways": "waterway",
    "railways": "railway",
}
transport_segments = {
    "roads": "road_segment_entities",
    "waterways": "waterway_segment_entities",
    "railways": "track_segment_entities",
}

# if NAME == "green":
#     UNIT_CONVERSION_COEFFICIENTS = {
#         "roads": "traffic_kpi_coefficients_diesel_rapid_development",
#         "waterways": "waterway_kpi_coefficients_diesel_rapid_development"
#     }
# elif NAME in ("safety", "missed"):
#     UNIT_CONVERSION_COEFFICIENTS = {
#         "roads": "traffic_kpi_coefficients_diesel_mild_development",
#         "waterways": "waterway_kpi_coefficients_diesel_mild_development"
#     }
# else:
#     UNIT_CONVERSION_COEFFICIENTS = {
#         "roads": "traffic_kpi_coefficients_diesel",
#         "waterways": "waterway_kpi_coefficients_diesel"
#     }


def make_config(
    name,
    display_name,
    models: t.Optional[t.Sequence[dict]] = None,
    datasets: t.Optional[t.Sequence[dict]] = None,
):
    return {
        "name": name,
        "display_name": display_name,
        "version": 4,
        "epsg_code": 28992,
        "bounding_box": [1, 2, 3, 4],
        "simulation_info": {
            "mode": "time_oriented",
            "start_time": 0,
            "time_scale": 1,
            "reference_time": 1546333200,
            "duration": 1009152000,
        },
        "created_on": NOW_STRING,
        "description": (
            f"Ritri {DISPLAY_NAME} scenario. Calculations reference: Asgarpour, S.,"
            " Konstantinos, K., Hartmann, A., and Neef, R. (2021). Modeling interdependent"
            " infrastructures under future scenarios. Work in Progress."
        ),
        "models": models or [],
        "datasets": datasets or [],
    }


def data_collector(aggregate_updates=True, convergence_only=False):
    sub_mask = (
        {
            "road_network": {
                "virtual_node_entities": [
                    "transport.passenger_demand",
                    "transport.cargo_demand",
                ],
                transport_segments["roads"]: [
                    "transport.average_time",
                ],
            },
            "waterway_network": {
                "virtual_node_entities": [
                    "transport.cargo_demand",
                ],
                transport_segments["waterways"]: [
                    "transport.average_time",
                ],
            },
            "railway_network": {
                "virtual_node_entities": [
                    "transport.passenger_demand",
                    "transport.cargo_demand",
                    "transport.generalized_journey_time",
                ]
            },
        }
        if convergence_only
        else "*"
    )
    return {
        "type": "data_collector",
        "name": "data_collector",
        "aggregate_updates": aggregate_updates,
        "gather_filter": sub_mask,
    }


def tape_player(config: dict, dataset_name: str):
    add_dataset(config, dataset_name, "tabular")
    return {
        "type": "tape_player",
        "name": f"{dataset_name}_tape_player",
        "tabular": dataset_name,
    }


@dataclasses.dataclass
class Aggregation:
    dataset: str
    entity_group: str
    attribute: str
    geometry: str
    function: str
    target: str


def area_aggregation(
    name,
    aggregations: t.Sequence[Aggregation],
    target_dataset,
    target_entity_group="area_entities",
    output_interval=None,
):
    rv = {
        "type": "area_aggregation",
        "name": name,
        "target_entity_group": [target_dataset, target_entity_group],
        "aggregations": [
            {
                "source_entity_group": [
                    agg.dataset,
                    agg.entity_group,
                ],
                "source_attribute": agg.attribute,
                "target_attribute": agg.target,
                "function": agg.function,
                "source_geometry": agg.geometry,
            }
            for agg in aggregations
        ],
    }
    if output_interval is not None:
        rv["output_interval"] = output_interval

    return rv


def traffic_assignment(
    name: str,
    dataset: str,
    modality: str,
    vdf_alpha=None,
    vdf_beta=None,
    cargo_pcu=None,
):
    rv = {
        "type": "traffic_assignment_calculation",
        "name": name,
        "modality": modality,
        "dataset": dataset,
    }
    if vdf_alpha is not None:
        rv["vdf_alpha"] = vdf_alpha
    if vdf_beta is not None:
        rv["vdf_beta"] = vdf_beta
    if cargo_pcu is not None:
        rv["cargo_pcu"] = cargo_pcu
    return rv


@dataclasses.dataclass
class GlobalDemandParameter:
    name: str
    elasticity: float


@dataclasses.dataclass
class LocalDemandParameter:
    dataset: str
    entity_group: str
    attribute: str
    geometry: str
    mapping_type: str
    elasticity: float


@dataclasses.dataclass
class OutputDemandAttributes:
    dataset: str
    entity_group: str
    local_demand: str
    total_inward: str
    total_outward: str


def traffic_demand_calculation(
    name,
    scenario_parameters: str,
    output: OutputDemandAttributes,
    global_parameters: t.Sequence[GlobalDemandParameter],
    local_parameters: t.Sequence[LocalDemandParameter],
    scenario_multipliers: t.Optional[t.Sequence[str]] = None,
    investment_multipliers: t.Optional[t.List] = None,
    atol=1e-3,
    rtol=1e-6,
    max_iterations=MAX_ITERATIONS,
):
    rv = {
        "type": "traffic_demand_calculation",
        "name": name,
        "demand_path": [output.dataset, output.entity_group, output.local_demand],
        "global_parameters": [
            {"name": param.name, "elasticity": param.elasticity} for param in global_parameters
        ],
        "local_parameters": [
            {
                "attribute_path": [param.dataset, param.entity_group, param.attribute],
                "geometry": param.geometry,
                "elasticity": param.elasticity,
                "mapping_type": param.mapping_type,
            }
            for param in local_parameters
        ],
        "parameter_dataset": scenario_parameters,
        "total_outward_demand_attribute": output.total_outward,
        "total_inward_demand_attribute": output.total_inward,
        "atol": atol,
        "rtol": rtol,
        "max_iterations": max_iterations,
    }

    if scenario_multipliers:
        rv["scenario_multipliers"] = scenario_multipliers
    if investment_multipliers:
        rv["investment_multipliers"] = investment_multipliers
    return rv


def udf(name, dataset, entity_group, inputs, functions):
    return {
        "name": name,
        "type": "udf",
        "entity_group": [dataset, entity_group],
        "inputs": inputs,
        "functions": functions,
    }


def traffic_kpi(
    name: str,
    dataset: str,
    modality: str,
    fuel_type: str,
    scenario_parameters: str,
    config: dict,
    coefficents_csv: str = "{mod_prefix}_kpi_coefficients_{fuel_type}",
    cargo_parameters: str = "{fuel_type}_share_freight_{mod_suffix}",
    passenger_parameters: str = "{fuel_type}_share_passenger_{mod_suffix}",
    energy_attribute: str = "transport.energy_consumption_{fuel_type}.hours",
    co2_attribute: str = "transport.co2_emission_{fuel_type}.hours",
    nox_attribute: str = "transport.nox_emission_{fuel_type}.hours",
):
    if NAME == "green":
        coefficents_csv = "{mod_prefix}_kpi_coefficients_{fuel_type}_rapid_development"
    elif NAME != "infraconomy":
        coefficents_csv = "{mod_prefix}_kpi_coefficients_{fuel_type}_mild_development"

    prefixes = {
        "roads": "traffic",
        "waterways": "waterway",
        "railways": "railway",
    }
    suffixes = {
        "roads": "road",
        "waterways": "waterway",
        "railways": "rail",
    }
    modality_key_map = {
        "roads": "roads",
        "waterways": "waterways",
        "railways": "tracks",
    }
    sub_params = {
        "fuel_type": fuel_type,
        "mod_prefix": prefixes[modality],
        "mod_suffix": suffixes[modality],
    }
    csv_dataset = add_dataset(
        config,
        coefficents_csv.format(**sub_params),
        ds_type="parameters",
    )

    rv = {
        "type": "traffic_kpi",
        "name": name,
        modality_key_map[modality]: [dataset],
        "scenario_parameters": [scenario_parameters],
        "coefficients_csv": [csv_dataset],
        "energy_consumption_property": [None, energy_attribute.format(**sub_params)],
        "co2_emission_property": [None, co2_attribute.format(**sub_params)],
        "nox_emission_property": [None, nox_attribute.format(**sub_params)],
    }
    if cargo_parameters is not None:
        rv["cargo_scenario_parameters"] = [cargo_parameters.format(**sub_params)]
    if passenger_parameters is not None:
        rv["passenger_scenario_parameters"] = [passenger_parameters.format(**sub_params)]

    return rv


@dataclasses.dataclass
class UnitConversionEntityGroup:
    dataset: str
    entity_group: str
    type: t.Literal["flow", "od"]
    modality: t.Literal["roads", "waterways"]


def unit_conversion(name, coefficients: str, entity_groups: t.Sequence[UnitConversionEntityGroup]):
    rv = {
        "type": "unit_conversions",
        "name": name,
        "parameters": [coefficients],
        "flow_entities": [],
        "flow_types": [],
        "od_entities": [],
        "od_types": [],
    }
    for eg in entity_groups:
        path = (eg.dataset, eg.entity_group)
        rv[eg.type + "_entities"].append(path)
        rv[eg.type + "_types"].append(eg.modality)
    return rv


def gjt(name, dataset, entity_group, travel_time="transport.passenger_average_time"):
    return {
        "name": name,
        "type": "generalized_journey_time",
        "transport_segments": [dataset, entity_group],
        "travel_time": travel_time,
    }


@dataclasses.dataclass
class CSVPlayerParameter:
    source: str
    target: str


def csv_player(
    name,
    dataset,
    entity_group,
    csv_tape: str,
    parameters: t.Sequence[CSVPlayerParameter],
):
    rv = {
        "name": name,
        "type": "csv_player",
        "entity_group": [[dataset, entity_group]],
        "csv_tape": csv_tape,
        "csv_parameters": [
            {"parameter": param.source, "target_attribute": param.target} for param in parameters
        ],
    }

    return rv


def add_dataset(config, name, ds_type):
    existing = {ds["name"] for ds in config["datasets"]}
    if name not in existing:
        config["datasets"].append({"name": name, "type": ds_type})
    return name


def kpi_models(
    dataset: str,
    modality: str,
    scenario_parameters: str,
    config: dict,
    exclude_fuel_types=(),
    exclude_passenger_fuel_types=("diesel",),
    exclude_cargo_fuel_types=("petrol",),
):
    prefix = prefixes[modality]
    fuel_types = ["diesel", "petrol", "electricity", "h2"]
    for fuel in exclude_fuel_types:
        fuel_types.remove(fuel)
    models = []
    model_params = {
        key: {
            "passenger_parameters": None,
        }
        for key in exclude_passenger_fuel_types
    }

    for fuel in exclude_cargo_fuel_types:
        if fuel not in model_params:
            model_params[fuel] = {}
        model_params[fuel]["cargo_parameters"] = None

    udf_inputs = {}
    udf_input_keys = {"h2": "hh"}
    for fuel in fuel_types:
        models.append(
            traffic_kpi(
                f"{prefix}_kpi_{fuel}",
                dataset,
                modality=modality,
                fuel_type=fuel,
                scenario_parameters=scenario_parameters,
                config=config,
                **model_params.get(fuel, {}),
            )
        )
        udf_inputs.update(
            {
                f"coo_{udf_input_keys.get(fuel, fuel)}": [
                    None,
                    f"transport.co2_emission_{fuel}.hours",
                ],
                f"nox_{udf_input_keys.get(fuel, fuel)}": [
                    None,
                    f"transport.nox_emission_{fuel}.hours",
                ],
                f"energy_{udf_input_keys.get(fuel, fuel)}": [
                    None,
                    f"transport.energy_consumption_{fuel}.hours",
                ],
            }
        )

    return [
        *models,
        udf(
            f"{prefix}_total_traffic_kpi",
            dataset=dataset,
            entity_group=transport_segments[modality],
            inputs=udf_inputs,
            functions=[
                {
                    "expression": "+".join(
                        f"coo_{udf_input_keys.get(fuel, fuel)}" for fuel in fuel_types
                    ),
                    "output": [None, "transport.co2_emission.hours"],
                },
                {
                    "expression": "+".join(
                        f"nox_{udf_input_keys.get(fuel, fuel)}" for fuel in fuel_types
                    ),
                    "output": [None, "transport.nox_emission.hours"],
                },
                {
                    "expression": "+".join(
                        f"energy_{udf_input_keys.get(fuel, fuel)}" for fuel in fuel_types
                    ),
                    "output": [None, "transport.energy_consumption.hours"],
                },
            ],
        ),
    ]


def peak_demand_models(
    dataset,
    modality,
    aggregation_dataset,
    passengers=True,
    cargo_vehicles=True,
    passenger_vehicles=True,
    passenger_peak_multiplier=PASSENGER_PEAK_MULTIPLIER,
    cargo_peak_multiplier=CARGO_PEAK_MULTIPLIER,
):
    prefix = prefixes[modality]

    demand_inputs = {
        "domestic_cargo_demand": "transport.domestic_cargo_demand",
        "international_cargo_demand": "transport.international_cargo_demand",
        "cargo_demand": "transport.cargo_demand",
    }
    demand_functions = [
        {
            "expression": f"sum(domestic_cargo_demand*{cargo_peak_multiplier})",
            "output": "transport.domestic_cargo_demand.peak_yearly",
        },
        {
            "expression": f"sum(international_cargo_demand*{cargo_peak_multiplier})",
            "output": "transport.international_cargo_demand.peak_yearly",
        },
        {
            "expression": f"sum(cargo_demand*{cargo_peak_multiplier})",
            "output":  "transport.cargo_demand.peak_yearly",
        },
    ]
    if cargo_vehicles:
        cargo_flow_input = "transport.cargo_vehicle_flow"
        cargo_flow_output = "transport.cargo_vkm.peak_yearly"
    else:
        cargo_flow_input = "transport.cargo_flow"
        cargo_flow_output = "transport.cargo_tkm.peak_yearly"

    flow_inputs = {
        "cargo_flow": cargo_flow_input,
        "length": "shape.length",
    }

    flow_functions = [
        {
            "expression": f"cargo_flow*length*0.001*{cargo_peak_multiplier}",
            "output":cargo_flow_output,
        },
    ]
    aggregations = [
        Aggregation(
            dataset,
            transport_segments[modality],
            attribute=cargo_flow_output,
            geometry="line",
            function="sum",
            target=f"{cargo_flow_output}.{modality}",
        ),
        Aggregation(
            dataset,
            "virtual_node_entities",
            attribute="transport.domestic_cargo_demand.peak_yearly",
            geometry="point",
            function="sum",
            target=f"transport.domestic_cargo_demand.peak_yearly.{modality}",
        ),
        Aggregation(
            dataset,
            "virtual_node_entities",
            attribute="transport.international_cargo_demand.peak_yearly",
            geometry="point",
            function="sum",
            target=f"transport.international_cargo_demand.peak_yearly.{modality}",
        ),
        Aggregation(
            dataset,
            "virtual_node_entities",
            attribute="transport.cargo_demand.peak_yearly",
            geometry="point",
            function="sum",
            target=f"transport.cargo_demand.peak_yearly.{modality}",
        ),
    ]
    if passengers:
        if passenger_vehicles:
            passenger_flow_input = "transport.passenger_vehicle_flow"
            passenger_flow_output = "transport.passenger_vkm.peak_yearly"
        else:
            passenger_flow_input = "transport.passenger_flow"
            passenger_flow_output = "transport.passenger_km.peak_yearly"

        demand_inputs["passenger_demand"] = "transport.passenger_demand"
        demand_functions.append(
            {
                "expression": f"sum(passenger_demand*{passenger_peak_multiplier})",
                "output":"transport.passenger_demand.peak_yearly",
            }
        )

        flow_inputs["passenger_flow"] = passenger_flow_input
        flow_functions.append(
            {
                "expression": f"passenger_flow*length*0.001*{passenger_peak_multiplier}",
                "output": passenger_flow_output,
            },
        )
        aggregations.extend(
            [
                Aggregation(
                    dataset,
                    transport_segments[modality],
                    attribute=passenger_flow_output,
                    geometry="line",
                    function="sum",
                    target=f"{passenger_flow_output}.{modality}",
                ),
                Aggregation(
                    dataset,
                    "virtual_node_entities",
                    attribute="transport.passenger_demand.peak_yearly",
                    geometry="point",
                    function="sum",
                    target=f"transport.passenger_demand.peak_yearly.{modality}",
                ),
            ]
        )

    demand_udf = udf(
        f"{prefix}_yearly_peak_demand",
        dataset=dataset,
        entity_group="virtual_node_entities",
        inputs=demand_inputs,
        functions=demand_functions,
    )
    return [
        demand_udf,
        udf(
            f"{prefix}_yearly_peak_flow",
            dataset=dataset,
            entity_group=transport_segments[modality],
            inputs=flow_inputs,
            functions=flow_functions,
        ),
        area_aggregation(
            f"{prefix}_total_aggregation",
            aggregations=aggregations,
            target_dataset=aggregation_dataset,
            target_entity_group="area_entities",
        ),
    ]


def municipalities_aggregation(
    dataset, modality, target_dataset="municipalities_area_set", has_passenger=True
):
    entity_group = transport_segments[modality]
    part = functools.partial(
        Aggregation, dataset=dataset, entity_group=entity_group, geometry="line"
    )
    aggregations = [
        # part(
        #     attribute="transport.cargo_flow",
        #     target=f"transport.cargo_flow.{modality}",
        #     function="max",
        # ),
        # part(
        #     attribute="transport.energy_consumption.hours",
        #     target=f"transport.energy_consumption.{modality}",
        #     function="integral_hours",
        # ),
        # part(
        #     attribute="transport.co2_emission.hours",
        #     target=f"transport.co2_emission.hours.{modality}",
        #     function="sum",
        # ),
        # part(
        #     attribute="transport.nox_emission.hours",
        #     target=f"transport.nox_emission.hours.{modality}",
        #     function="sum",
        # ),
        # part(
        #     attribute="transport.cargo_flow",
        #     target=f"transport.cargo_flow.{modality}.total",
        #     function="sum",
        # ),
    ]
    # if has_passenger:
    #     aggregations.extend(
    #         [
    #             part(
    #                 attribute="transport.passenger_flow",
    #                 target=f"transport.passenger_flow.{modality}",
    #                 function="max",
    #             ),
    #             part(
    #                 attribute="transport.passenger_flow",
    #                 target=f"transport.passenger_flow.{modality}.total",
    #                 function="sum",
    #             ),
    #         ]
    #     )
    if modality != "railways":
        aggregations.extend(
            [
                part(
                    attribute="transport.cargo_vehicle_flow",
                    target=f"transport.cargo_vehicle_flow.{modality}",
                    function="max",
                ),
                part(
                    attribute="transport.volume_to_capacity_ratio",
                    target=f"transport.volume_to_capacity_ratio.{modality}.max",
                    function="max",
                ),
                part(
                    attribute="transport.delay_factor",
                    target=f"transport.delay_factor.{modality}.max",
                    function="max",
                ),
                part(
                    attribute="transport.cargo_vehicle_flow",
                    target=f"transport.cargo_vehicle_flow.{modality}.total",
                    function="sum",
                ),
                part(
                    attribute="transport.volume_to_capacity_ratio",
                    target=f"transport.volume_to_capacity_ratio.{modality}.average",
                    function="average",
                ),
                part(
                    attribute="transport.delay_factor",
                    target=f"transport.delay_factor.{modality}.average",
                    function="average",
                ),
            ]
        )
        if has_passenger:
            aggregations.extend(
                [
                    part(
                        attribute="transport.passenger_vehicle_flow",
                        target=f"transport.passenger_vehicle_flow.{modality}",
                        function="max",
                    ),
                    part(
                        attribute="transport.passenger_vehicle_flow",
                        target=f"transport.passenger_vehicle_flow.{modality}.total",
                        function="sum",
                    ),
                ]
            )
    return area_aggregation(
        name=f"{prefixes[modality]}_municipalities_aggregation",
        target_dataset=target_dataset,
        aggregations=aggregations,
        output_interval=2628000,
    )


def relaxation(
    dataset,
    modality,
    rf=0.6,
    name=None,
    attribute="transport.average_time",
    entity_group=None,
):
    return udf(
        name=name or f"{prefixes[modality]}_relaxation",
        dataset=dataset,
        entity_group=entity_group or transport_segments[modality],
        inputs={
            "outgoing":  f"{attribute}.star",
            "incoming": attribute,
        },
        functions=[
            {
                "expression": f"default(outgoing, incoming)*{1 - rf:.3}+incoming*{rf:.3}",
                "output":f"{attribute}.star",
            }
        ],
    )


def combined_cargo_demand(dataset, modality):
    return udf(
        f"{prefixes[modality]}_combined_cargo_demand",
        dataset=dataset,
        entity_group="virtual_node_entities",
        inputs={
            "domestic": "transport.domestic_cargo_demand",
            "intl": "transport.international_cargo_demand",
        },
        functions=[
            {"expression": "domestic+intl", "output":"transport.cargo_demand"},
        ],
    )


def total_cargo_demand(dataset, modality, suffix="_vehicles"):
    return udf(
        f"{prefixes[modality]}_total_cargo_demand",
        dataset=dataset,
        entity_group="virtual_node_entities",
        inputs={
            "total_inward_domestic":  f"transport.total_inward_domestic_cargo_demand{suffix}",
            "total_inward_intl":  f"transport.total_inward_international_cargo_demand{suffix}",
            "total_outward_domestic": f"transport.total_outward_domestic_cargo_demand{suffix}",
            "total_outward_intl":f"transport.total_outward_international_cargo_demand{suffix}",
        },
        functions=[
            {
                "expression": "total_inward_domestic+total_inward_intl",
                "output": f"transport.total_inward_cargo_demand{suffix}",
            },
            {
                "expression": "total_outward_domestic+total_outward_intl",
                "output": f"transport.total_outward_cargo_demand{suffix}",
            },
        ],
    )


def shortest_path(
    name,
    dataset,
    modality,
    cost_factor,
    calculation_type,
    calculation_input_attribute,
    calculation_output_attribute,
    no_update_shortest_paths=False,
):
    calculation = {
        "type": calculation_type,
        "input": calculation_input_attribute,
        "output": calculation_output_attribute,
    }
    return {
        "name": name,
        "type": "shortest_path",
        "transport_segments": [dataset, transport_segments[modality]],
        "cost_factor": cost_factor,
        "calculations": [calculation],
        "no_update_shortest_path": no_update_shortest_paths,
    }


def intensity_capacity(transport_dataset, bridges_dataset, capacity_threshold=2000):
    return {
        "type": "intensity_capacity",
        "name": "ic_" + bridges_dataset,
        "transport_dataset": transport_dataset,
        "bridges_dataset": bridges_dataset,
        "capacity_threshold": capacity_threshold
    }

def safety(transport_dataset):
    return {
        "type": "safety_model",
        "name": "safety_model",
        "transport_dataset": transport_dataset
    }

def generate():
    config = make_config(
        name=NAME,
        display_name=DISPLAY_NAME,
    )
    scenario_parameters = add_dataset(config, SCENARIO_PARAMETERS, "parameters")
    road_network = add_dataset(config, "road_network", "transport_network")
    bridges = add_dataset(config, "bridges", "bridges")

    municipalities_area_set = add_dataset(config, "municipalities_area_set", "area_set")
    total_area = add_dataset(config, "total_area", "area_set")
    config["models"] = [
        data_collector(aggregate_updates=True),
        tape_player(config=config, dataset_name=f"{NAME}_local_parameters_tape"),
        municipalities_aggregation(road_network, "roads"),
        traffic_assignment(
            "road_traffic_assignment",
            road_network,
            "roads",
            vdf_alpha=0.64,
            vdf_beta=4,
            cargo_pcu=2,
        ),
        traffic_demand_calculation(
            "road_passenger_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                road_network,
                "virtual_node_entities",
                local_demand="transport.passenger_demand",
                total_inward="transport.total_inward_passenger_demand_vehicles",
                total_outward="transport.total_outward_passenger_demand_vehicles",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", -0.09 / 2),
                GlobalDemandParameter("gdp", 0.38 / 2),
                GlobalDemandParameter("train_ticket_price", 0.02 / 2),
                GlobalDemandParameter("total_vehicles", 0.6 / 2),
                GlobalDemandParameter("commuting_jobs_share", 0.64),
            ],
            local_parameters=[
                LocalDemandParameter(
                    municipalities_area_set,
                    "area_entities",
                    "jobs.count.index",
                    "polygon",
                    "nearest",
                    0.64,
                ),
                LocalDemandParameter(
                    municipalities_area_set,
                    "area_entities",
                    "people.count.index",
                    "polygon",
                    "nearest",
                    0.33,
                ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    -1.1,
                ),
                LocalDemandParameter(
                    road_network,
                    "virtual_node_entities",
                    "transport.shortest_path_lane_length",
                    "point",
                    "nearest",
                    0.25,
                ),
            ],
        ),
        traffic_demand_calculation(
            "road_domestic_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                road_network,
                "virtual_node_entities",
                local_demand="transport.domestic_cargo_demand",
                total_inward="transport.total_inward_domestic_cargo_demand_vehicles",
                total_outward="transport.total_outward_domestic_cargo_demand_vehicles",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", -0.07 / 2),
                GlobalDemandParameter("gdp", 0.57 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
                GlobalDemandParameter("road_load_factor_index", -1 / 2),
            ],
            local_parameters=[
                # LocalDemandParameter(
                #     road_network,
                #     transport_segments["roads"],
                #     "transport.average_time.star",
                #     "line",
                #     "route",
                #     -0.1,
                # ),
            ],
            # investment_multipliers=[
            #     [
            #         94608000,
            #         8037,
            #         1.051
            #     ],
            #     [
            #         126144000,
            #         8037,
            #         1.051
            #     ],
            #     [
            #         157680000,
            #         8037,
            #         1.051
            #     ],
            #     [
            #         189216000,
            #         8037,
            #         1.054
            #     ],
            #     [
            #         220752000,
            #         8037,
            #         1.054
            #     ],
            #     [
            #         252288000,
            #         8037,
            #         1.054
            #     ]
            # ],
        ),
        traffic_demand_calculation(
            "road_international_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                road_network,
                "virtual_node_entities",
                local_demand="transport.international_cargo_demand",
                total_inward="transport.total_inward_international_cargo_demand_vehicles",
                total_outward="transport.total_outward_international_cargo_demand_vehicles",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", -0.07 / 2),
                GlobalDemandParameter("world_trade_volume", 0.41 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
                GlobalDemandParameter("road_load_factor_index", -1 / 2),
            ],
            local_parameters=[
                # LocalDemandParameter(
                #     road_network,
                #     transport_segments["roads"],
                #     "transport.average_time.star",
                #     "line",
                #     "route",
                #     -0.1,
                # ),
            ],
            investment_multipliers=[
                # [
                #     94608000,
                #     8037,
                #     1.051
                # ],
                # [
                #     126144000,
                #     8037,
                #     1.051
                # ],
                # [
                #     157680000,
                #     8037,
                #     1.051
                # ],
                # [
                #     189216000,
                #     8037,
                #     1.054
                # ],
                # [
                #     220752000,
                #     8037,
                #     1.054
                # ],
                # [
                #     252288000,
                #     8037,
                #     1.054
                # ]
            ],
        ),
        relaxation(road_network, "roads", rf=RELAXATION_FACTOR),
        combined_cargo_demand(road_network, "roads"),
        total_cargo_demand(road_network, "roads"),
        udf(
            name="road_lane_length",
            dataset=road_network,
            entity_group=transport_segments["roads"],
            inputs={
                "layout": "transport.layout",  # [ 1,2,3,4 ]
                "length":  "shape.length",
            },
            functions=[
                {
                    "expression": "sum(layout)*length",
                    "output": "transport.lane_length",
                },
            ],
        ),
        shortest_path(
            name="road_lane_length_shortest_path",
            dataset=road_network,
            modality="roads",
            cost_factor="transport.average_time",
            calculation_type="sum",
            calculation_input_attribute="transport.lane_length",
            calculation_output_attribute="transport.shortest_path_lane_length",
            no_update_shortest_paths=True,
        ),
        shortest_path(
            name="road_length_shortest_path",
            dataset=road_network,
            modality="roads",
            cost_factor="transport.average_time",
            calculation_type="sum",
            calculation_input_attribute="shape.length",
            calculation_output_attribute="transport.shortest_path_length",
        ),
        udf(
            name="road_vkm",
            dataset=road_network,
            entity_group="virtual_node_entities",
            inputs={
                "cargo_demand": "transport.cargo_demand",
                "passenger_demand": "transport.passenger_demand",
                "shortest_path_length": "transport.shortest_path_length",
            },
            functions=[
                {
                    "expression": "cargo_demand*shortest_path_length",
                    "output": "transport.cargo_demand_vkm",
                },
                {
                    "expression": f"cargo_demand*shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": "transport.cargo_demand_vkm.peak_yearly",
                },
                {
                    "expression": "passenger_demand*shortest_path_length",
                    "output": "transport.passenger_demand_vkm",
                },
                {
                    "expression": f"passenger_demand*shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": "transport.passenger_demand_vkm.peak_yearly",
                },
            ],
        ),
        tape_player(config=config, dataset_name="road_network_investment_tape"),
        tape_player(config=config, dataset_name=f"{NAME}_noise_tapefile"),
        # *kpi_models(
        #     dataset=road_network,
        #     modality="roads",
        #     scenario_parameters=scenario_parameters,
        #     config=config,
        # ),
        # unit_conversion(
        #     "road_unit_conversion",
        #     UNIT_CONVERSION_COEFFICIENTS["roads"],
        #     [
        #         UnitConversionEntityGroup(
        #             road_network,
        #             transport_segments["roads"],
        #             type="flow",
        #             modality="roads",
        #         ),
        #         UnitConversionEntityGroup(
        #             road_network, "virtual_node_entities", type="od", modality="roads"
        #         ),
        #     ],
        # ),
        *peak_demand_models(
            dataset=road_network, aggregation_dataset=total_area, modality="roads"
        ),
        intensity_capacity(transport_dataset=road_network, bridges_dataset=bridges),
        safety(transport_dataset=road_network)
    ]
    return config


def generate_and_output(path=None):
    result = json.dumps(generate(), indent=2)
    if path is not None:

        path.write_text(result)
    else:
        print(result)


def test_main():
    expected = json.loads(pathlib.Path(__file__).parent.joinpath("master_config.json").read_text())
    scenario_config = json.loads(json.dumps(generate()))
    assert scenario_config == expected


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_and_output(path)
