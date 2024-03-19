#! /usr/bin/env python3

import dataclasses
import functools
import json
import pathlib
import sys
import typing as t

NAME = "green"
DISPLAY_NAME = NAME
SCENARIO_PARAMETERS = f"{NAME}_interpolated"
RELAXATION_FACTOR = 0.5
MAX_ITERATIONS = 100
PASSENGER_PEAK_MULTIPLIER = 4 * 280
CARGO_PEAK_MULTIPLIER = 9 * 280
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

if NAME == "green":
    UNIT_CONVERSION_COEFFICIENTS = {
        "roads": "traffic_kpi_coefficients_diesel_rapid_development",
        "waterways": "waterway_kpi_coefficients_diesel_rapid_development",
    }
elif NAME != "infraconomy":
    UNIT_CONVERSION_COEFFICIENTS = {
        "roads": "traffic_kpi_coefficients_diesel_mild_development",
        "waterways": "waterway_kpi_coefficients_diesel_mild_development",
    }
else:
    UNIT_CONVERSION_COEFFICIENTS = {
        "roads": "traffic_kpi_coefficients_diesel",
        "waterways": "waterway_kpi_coefficients_diesel",
    }


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
        "description": (
            f"Ritri {DISPLAY_NAME} scenario. Calculations reference: Asgarpour, S.,"
            " Konstantinos, K., Hartmann, A., and Neef, R. (2021). Modeling interdependent"
            " infrastructures under future scenarios. Work in Progress."
        ),
        "models": models or [],
        "datasets": datasets or [],
    }


def data_collector(aggregate_updates=True, convergence_only=True):
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
        "tabular": [dataset_name],
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
        "target_entity_group": [(target_dataset, target_entity_group)],
        "source_entity_groups": [],
        "source_geometry_types": [],
        "source_properties": [],
        "aggregation_functions": [],
        "target_properties": [],
    }
    if output_interval is not None:
        rv["output_interval"] = output_interval

    for agg in aggregations:
        rv["source_entity_groups"].append((agg.dataset, agg.entity_group))
        rv["source_geometry_types"].append(agg.geometry)
        rv["source_properties"].append((None, agg.attribute))
        rv["aggregation_functions"].append(agg.function)
        rv["target_properties"].append((None, agg.target))
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
        "dataset": [dataset],
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
        "scenario_parameters": [scenario_parameters],
        "demand_entity": [[output.dataset, output.entity_group]],
        "demand_property": [None, output.local_demand],
        "total_outward_demand_property": [None, output.total_outward],
        "total_inward_demand_property": [None, output.total_inward],
        "atol": atol,
        "rtol": rtol,
        "max_iterations": max_iterations,
        "global_parameters": [],
        "global_elasticities": [],
        "local_entity_groups": [],
        "local_properties": [],
        "local_geometries": [],
        "local_mapping_type": [],
        "local_elasticities": [],
    }
    for param in global_parameters:
        rv["global_parameters"].append(param.name)
        rv["global_elasticities"].append(param.elasticity)

    for param in local_parameters:
        rv["local_entity_groups"].append((param.dataset, param.entity_group))
        rv["local_properties"].append((None, param.attribute))
        rv["local_geometries"].append(param.geometry)
        rv["local_mapping_type"].append(param.mapping_type)
        rv["local_elasticities"].append(param.elasticity)
    if scenario_multipliers:
        rv["scenario_multipliers"] = scenario_multipliers
    if investment_multipliers:
        rv["investment_multipliers"] = investment_multipliers
    return rv


def udf(name, dataset, entity_group, inputs, functions):
    return {
        "name": name,
        "type": "udf",
        "entity_group": [(dataset, entity_group)],
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
        rv["passenger_scenario_parameters"] = [
            passenger_parameters.format(**sub_params)
        ]

    return rv


@dataclasses.dataclass
class UnitConversionEntityGroup:
    dataset: str
    entity_group: str
    type: t.Literal["flow", "od"]
    modality: t.Literal["roads", "waterways"]


def unit_conversion(
    name, coefficients: str, entity_groups: t.Sequence[UnitConversionEntityGroup]
):
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
        "transport_segments": [[dataset, entity_group]],
        "travel_time": [None, travel_time],
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
        "csv_tape": [csv_tape],
        "csv_parameters": [],
        "target_attributes": [],
    }
    for param in parameters:
        rv["csv_parameters"].append(param.source)
        rv["target_attributes"].append([None, param.target])

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
                        f"energy_{udf_input_keys.get(fuel, fuel)}"
                        for fuel in fuel_types
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
        "domestic_cargo_demand": [None, "transport.domestic_cargo_demand"],
        "international_cargo_demand": [None, "transport.international_cargo_demand"],
        "cargo_demand": [None, "transport.cargo_demand"],
    }
    demand_functions = [
        {
            "expression": f"sum(domestic_cargo_demand*{cargo_peak_multiplier})",
            "output": [None, "transport.domestic_cargo_demand.peak_yearly"],
        },
        {
            "expression": f"sum(international_cargo_demand*{cargo_peak_multiplier})",
            "output": [None, "transport.international_cargo_demand.peak_yearly"],
        },
        {
            "expression": f"sum(cargo_demand*{cargo_peak_multiplier})",
            "output": [None, "transport.cargo_demand.peak_yearly"],
        },
    ]
    if cargo_vehicles:
        cargo_flow_input = "transport.cargo_vehicle_flow"
        cargo_flow_output = "transport.cargo_vkm.peak_yearly"
    else:
        cargo_flow_input = "transport.cargo_flow"
        cargo_flow_output = "transport.cargo_tkm.peak_yearly"

    flow_inputs = {
        "cargo_flow": [None, cargo_flow_input],
        "length": [None, "shape.length"],
    }

    flow_functions = [
        {
            "expression": f"cargo_flow*length*0.001*{cargo_peak_multiplier}",
            "output": [None, cargo_flow_output],
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

        demand_inputs["passenger_demand"] = [None, "transport.passenger_demand"]
        demand_functions.append(
            {
                "expression": f"sum(passenger_demand*{passenger_peak_multiplier})",
                "output": [None, "transport.passenger_demand.peak_yearly"],
            }
        )

        flow_inputs["passenger_flow"] = [None, passenger_flow_input]
        flow_functions.append(
            {
                "expression": f"passenger_flow*length*0.001*{passenger_peak_multiplier}",
                "output": [None, passenger_flow_output],
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
        part(
            attribute="transport.cargo_flow",
            target=f"transport.cargo_flow.{modality}",
            function="max",
        ),
        part(
            attribute="transport.energy_consumption.hours",
            target=f"transport.energy_consumption.{modality}",
            function="integral_hours",
        ),
        part(
            attribute="transport.co2_emission.hours",
            target=f"transport.co2_emission.{modality}",
            function="integral_hours",
        ),
        part(
            attribute="transport.nox_emission.hours",
            target=f"transport.nox_emission.{modality}",
            function="integral_hours",
        ),
        part(
            attribute="transport.energy_consumption.hours",
            target=f"transport.energy_consumption.hours.{modality}",
            function="sum",
        ),
        part(
            attribute="transport.co2_emission.hours",
            target=f"transport.co2_emission.hours.{modality}",
            function="sum",
        ),
        part(
            attribute="transport.nox_emission.hours",
            target=f"transport.nox_emission.hours.{modality}",
            function="sum",
        ),
        part(
            attribute="transport.cargo_flow",
            target=f"transport.cargo_flow.{modality}.total",
            function="sum",
        ),
    ]
    if has_passenger:
        aggregations.extend(
            [
                part(
                    attribute="transport.passenger_flow",
                    target=f"transport.passenger_flow.{modality}",
                    function="max",
                ),
                part(
                    attribute="transport.passenger_flow",
                    target=f"transport.passenger_flow.{modality}.total",
                    function="sum",
                ),
            ]
        )
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
            "outgoing": [None, f"{attribute}.star"],
            "incoming": [None, attribute],
        },
        functions=[
            {
                "expression": f"default(outgoing, incoming)*{1 - rf:.3}+incoming*{rf:.3}",
                "output": [None, f"{attribute}.star"],
            }
        ],
    )


def combined_cargo_demand(dataset, modality):
    return udf(
        f"{prefixes[modality]}_combined_cargo_demand",
        dataset=dataset,
        entity_group="virtual_node_entities",
        inputs={
            "domestic": [None, "transport.domestic_cargo_demand"],
            "intl": [None, "transport.international_cargo_demand"],
        },
        functions=[
            {"expression": "domestic+intl", "output": [None, "transport.cargo_demand"]},
        ],
    )


def total_cargo_demand(dataset, modality, suffix="_vehicles"):
    return udf(
        f"{prefixes[modality]}_total_cargo_demand",
        dataset=dataset,
        entity_group="virtual_node_entities",
        inputs={
            "total_inward_domestic": [
                None,
                f"transport.total_inward_domestic_cargo_demand{suffix}",
            ],
            "total_inward_intl": [
                None,
                f"transport.total_inward_international_cargo_demand{suffix}",
            ],
            "total_outward_domestic": [
                None,
                f"transport.total_outward_domestic_cargo_demand{suffix}",
            ],
            "total_outward_intl": [
                None,
                f"transport.total_outward_international_cargo_demand{suffix}",
            ],
        },
        functions=[
            {
                "expression": "total_inward_domestic+total_inward_intl",
                "output": [None, f"transport.total_inward_cargo_demand{suffix}"],
            },
            {
                "expression": "total_outward_domestic+total_outward_intl",
                "output": [None, f"transport.total_outward_cargo_demand{suffix}"],
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
        "input": [None, calculation_input_attribute],
        "output": [None, calculation_output_attribute],
    }
    return {
        "name": name,
        "type": "shortest_path",
        "transport_segments": [[dataset, transport_segments[modality]]],
        "cost_factor": [None, cost_factor],
        "calculations": [calculation],
        "no_update_shortest_paths": no_update_shortest_paths,
    }


def generate():
    config = make_config(
        name=NAME,
        display_name=DISPLAY_NAME,
    )
    scenario_parameters = add_dataset(config, SCENARIO_PARAMETERS, "parameters")
    road_network = add_dataset(config, "road_network", "transport_network")
    railway_network = add_dataset(config, "railway_network", "transport_network")
    waterway_network = add_dataset(config, "waterway_network", "transport_network")
    municipalities_area_set = add_dataset(config, "municipalities_area_set", "area_set")
    total_area = add_dataset(config, "total_area", "area_set")
    config["models"] = [
        data_collector(aggregate_updates=True, convergence_only=False),
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
                    "railway_network",
                    "virtual_node_entities",
                    "transport.generalized_journey_time.star",
                    "point",
                    "nearest",
                    0.14,
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
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    -0.1,
                ),
                LocalDemandParameter(
                    waterway_network,
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.086,
                ),
            ],
            investment_multipliers=[
                [94608000, 8037, 1.051],
                [126144000, 8037, 1.051],
                [157680000, 8037, 1.051],
                [189216000, 8037, 1.054],
                [220752000, 8037, 1.054],
                [252288000, 8037, 1.054],
            ],
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
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    -0.1,
                ),
                LocalDemandParameter(
                    "waterway_network",
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.086,
                ),
            ],
            investment_multipliers=[
                [94608000, 8037, 1.051],
                [126144000, 8037, 1.051],
                [157680000, 8037, 1.051],
                [189216000, 8037, 1.054],
                [220752000, 8037, 1.054],
                [252288000, 8037, 1.054],
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
                "layout": [None, "transport.layout"],
                "length": [None, "shape.length"],
            },
            functions=[
                {
                    "expression": "sum(layout)*length",
                    "output": [None, "transport.lane_length"],
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
                "cargo_demand": [None, "transport.cargo_demand"],
                "passenger_demand": [None, "transport.passenger_demand"],
                "shortest_path_length": [None, "transport.shortest_path_length"],
            },
            functions=[
                {
                    "expression": "cargo_demand*shortest_path_length",
                    "output": [None, "transport.cargo_demand_vkm"],
                },
                {
                    "expression": f"cargo_demand*shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": [None, "transport.cargo_demand_vkm.peak_yearly"],
                },
                {
                    "expression": "passenger_demand*shortest_path_length",
                    "output": [None, "transport.passenger_demand_vkm"],
                },
                {
                    "expression": f"passenger_demand*shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": [None, "transport.passenger_demand_vkm.peak_yearly"],
                },
            ],
        ),
        tape_player(config=config, dataset_name="road_network_investment_tape"),
        tape_player(config=config, dataset_name="lock_investment_tape"),
        tape_player(config=config, dataset_name="railway_network_investment_tape"),
        *kpi_models(
            dataset=road_network,
            modality="roads",
            scenario_parameters=scenario_parameters,
            config=config,
        ),
        unit_conversion(
            "road_unit_conversion",
            UNIT_CONVERSION_COEFFICIENTS["roads"],
            [
                UnitConversionEntityGroup(
                    road_network,
                    transport_segments["roads"],
                    type="flow",
                    modality="roads",
                ),
                UnitConversionEntityGroup(
                    road_network, "virtual_node_entities", type="od", modality="roads"
                ),
            ],
        ),
        *peak_demand_models(
            dataset=road_network, aggregation_dataset=total_area, modality="roads"
        ),
        municipalities_aggregation("railway_network", "railways"),
        traffic_assignment(
            "railway_cargo_traffic_assignment", "railway_network", "cargo_tracks"
        ),
        traffic_assignment(
            "railway_passenger_traffic_assignment",
            "railway_network",
            "passenger_tracks",
        ),
        traffic_demand_calculation(
            "railway_passenger_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                railway_network,
                "virtual_node_entities",
                local_demand="transport.passenger_demand",
                total_inward="transport.total_inward_passenger_demand",
                total_outward="transport.total_outward_passenger_demand",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", 0.11 / 2),
                GlobalDemandParameter("income", 0.27 / 2),
                GlobalDemandParameter("train_ticket_price", -0.63 / 2),
                GlobalDemandParameter("train_punctuality", 1.1 / 2),
                GlobalDemandParameter("total_vehicles", -0.02 / 2),
                GlobalDemandParameter("commuting_jobs_share", 0.29),
            ],
            local_parameters=[
                LocalDemandParameter(
                    municipalities_area_set,
                    "area_entities",
                    "jobs.count.index",
                    "polygon",
                    "nearest",
                    0.29,
                ),
                LocalDemandParameter(
                    municipalities_area_set,
                    "area_entities",
                    "people.count.index",
                    "polygon",
                    "nearest",
                    1,
                ),
                LocalDemandParameter(
                    railway_network,
                    "virtual_node_entities",
                    "transport.generalized_journey_time.star",
                    "point",
                    "nearest",
                    -1.51,
                ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.24,
                ),
            ],
        ),
        traffic_demand_calculation(
            "railway_domestic_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                railway_network,
                "virtual_node_entities",
                local_demand="transport.domestic_cargo_demand",
                total_inward="transport.total_inward_domestic_cargo_demand",
                total_outward="transport.total_outward_domestic_cargo_demand",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", 0.22 / 2),
                GlobalDemandParameter("gdp", 0.57 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
            ],
            local_parameters=[
                LocalDemandParameter(
                    railway_network,
                    transport_segments["railways"],
                    "transport.cargo_average_time.star",
                    "line",
                    "route",
                    -0.211,
                ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.223,
                ),
                LocalDemandParameter(
                    waterway_network,
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.6,
                ),
            ],
            investment_multipliers=[
                [94608000, 2074, 1.051],
                [126144000, 2074, 1.051],
                [157680000, 2074, 1.051],
                [189216000, 2074, 1.054],
                [220752000, 2074, 1.054],
                [252288000, 2074, 1.054],
            ],
        ),
        traffic_demand_calculation(
            "railway_international_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                railway_network,
                "virtual_node_entities",
                local_demand="transport.international_cargo_demand",
                total_inward="transport.total_inward_international_cargo_demand",
                total_outward="transport.total_outward_international_cargo_demand",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", 0.22 / 2),
                GlobalDemandParameter("world_trade_volume", 0.41 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
            ],
            local_parameters=[
                LocalDemandParameter(
                    railway_network,
                    transport_segments["railways"],
                    "transport.cargo_average_time.star",
                    "line",
                    "route",
                    -0.211,
                ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.223,
                ),
                LocalDemandParameter(
                    waterway_network,
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.6,
                ),
            ],
            investment_multipliers=[
                [94608000, 2074, 1.051],
                [126144000, 2074, 1.051],
                [157680000, 2074, 1.051],
                [189216000, 2074, 1.054],
                [220752000, 2074, 1.054],
                [252288000, 2074, 1.054],
            ],
        ),
        relaxation(
            railway_network,
            "railways",
            name="railway_cargo_relaxation",
            rf=RELAXATION_FACTOR,
            attribute="transport.cargo_average_time",
        ),
        relaxation(
            railway_network,
            "railways",
            name="railway_passenger_relaxation",
            rf=RELAXATION_FACTOR,
            attribute="transport.generalized_journey_time",
            entity_group="virtual_node_entities",
        ),
        combined_cargo_demand(railway_network, "railways"),
        total_cargo_demand(railway_network, "railways", suffix=""),
        udf(
            name="railway_vehicle_demand",
            dataset=railway_network,
            entity_group=transport_segments["railways"],
            inputs={
                "passenger_flow": [None, "transport.passenger_flow"],
                "cargo_flow": [None, "transport.cargo_flow"],
            },
            functions=[
                {
                    "expression": "passenger_flow",
                    "output": [None, "transport.passenger_vehicle_flow"],
                },
                {
                    "expression": "cargo_flow",
                    "output": [None, "transport.cargo_vehicle_flow"],
                },
            ],
        ),
        *kpi_models(
            railway_network,
            "railways",
            scenario_parameters,
            config=config,
            exclude_fuel_types=("petrol",),
        ),
        *peak_demand_models(
            dataset=railway_network,
            modality="railways",
            aggregation_dataset=total_area,
            cargo_vehicles=False,
            passenger_vehicles=False,
        ),
        gjt("railway_gjt", railway_network, transport_segments["railways"]),
        csv_player(
            "railway_gjt_parameters",
            railway_network,
            "virtual_node_entities",
            csv_tape=scenario_parameters,
            parameters=[
                # CSVPlayerParameter(  ## use this when we increase train frequency by scenario interpolated csvs (scenario input)
                #     "passenger_train_frequency_change",
                #     "transport.passenger_vehicle_frequency_index",
                # ),
                CSVPlayerParameter(
                    "passenger_train_capacity",
                    "transport.passenger_vehicle_capacity",
                ),
            ],
        ),
        # udf(  ## use this when we increase train frequency by scenario interpolated csvs (scenario input)
        #     "railway_frequency",
        #     dataset=railway_network,
        #     entity_group="virtual_node_entities",
        #     inputs={
        #         "base_frequency": [None, "transport.passenger_vehicle_base_frequency"],
        #         "frequency_index": [
        #             None,
        #             "transport.passenger_vehicle_frequency_index",
        #         ],
        #     },
        #     functions=[
        #         {
        #             "expression": "min(base_frequency * frequency_index / 100, 60/2.75)  + frequency_increase",
        #             "output": [None, "transport.passenger_vehicle_frequency"],
        #         }
        #     ],
        # ),
        shortest_path(
            name="railway_passenger_shortest_path",
            dataset=railway_network,
            modality="railways",
            cost_factor="transport.passenger_average_time",
            calculation_type="sum",
            calculation_input_attribute="shape.length",
            calculation_output_attribute="transport.passenger_shortest_path_length",
        ),
        shortest_path(
            name="railway_cargo_shortest_path",
            dataset=railway_network,
            modality="railways",
            cost_factor="transport.cargo_average_time",
            calculation_type="sum",
            calculation_input_attribute="shape.length",
            calculation_output_attribute="transport.cargo_shortest_path_length",
        ),
        udf(
            name="railway_tkm",
            dataset=railway_network,
            entity_group="virtual_node_entities",
            inputs={
                "cargo_demand": [None, "transport.cargo_demand"],
                "cargo_shortest_path_length": [
                    None,
                    "transport.cargo_shortest_path_length",
                ],
            },
            functions=[
                {
                    "expression": "cargo_demand*cargo_shortest_path_length",
                    "output": [None, "transport.cargo_demand_tkm"],
                },
                {
                    "expression": f"cargo_demand*cargo_shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": [None, "transport.cargo_demand_tkm.peak_yearly"],
                },
            ],
        ),
        municipalities_aggregation(waterway_network, "waterways", has_passenger=False),
        traffic_assignment(
            "waterway_traffic_assignment",
            waterway_network,
            "waterways",
            cargo_pcu=1,
            vdf_alpha=0.64,
            vdf_beta=4,
        ),
        traffic_demand_calculation(
            "waterway_domestic_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                waterway_network,
                "virtual_node_entities",
                local_demand="transport.domestic_cargo_demand",
                total_inward="transport.total_inward_domestic_cargo_demand_vehicles",
                total_outward="transport.total_outward_domestic_cargo_demand_vehicles",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", 0.06 / 2),
                GlobalDemandParameter("gdp", 0.57 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
                GlobalDemandParameter("waterway_load_factor_index", -1 / 2),
            ],
            local_parameters=[
                # LocalDemandParameter(  ## Low elasticity, we skip it
                #     railway_network,
                #     transport_segments["railways"],
                #     "transport.cargo_average_time.star",
                #     "line",
                #     "route",
                #     0.016,
                # ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.231,
                ),
                LocalDemandParameter(
                    waterway_network,
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    -0.284,
                ),
            ],
            investment_multipliers=[
                [94608000, 1829, 1.051],
                [126144000, 1829, 1.051],
                [157680000, 1829, 1.051],
                [189216000, 1829, 1.054],
                [220752000, 1829, 1.054],
                [252288000, 1829, 1.054],
            ],
            atol=1e-5,
            rtol=1e-8,
        ),
        traffic_demand_calculation(
            "waterway_international_cargo_demand_estimation",
            scenario_parameters,
            output=OutputDemandAttributes(
                waterway_network,
                "virtual_node_entities",
                local_demand="transport.international_cargo_demand",
                total_inward="transport.total_inward_international_cargo_demand_vehicles",
                total_outward="transport.total_outward_international_cargo_demand_vehicles",
            ),
            global_parameters=[
                GlobalDemandParameter("consumer_energy_price", 0.06 / 2),
                GlobalDemandParameter("world_trade_volume", 0.41 / 2),
                GlobalDemandParameter("share_service_sector_gdp", -0.8 / 2),
                GlobalDemandParameter("waterway_load_factor_index", -1 / 2),
            ],
            local_parameters=[
                # LocalDemandParameter(  ## Low elasticity, we skip it
                #     railway_network,
                #     transport_segments["railways"],
                #     "transport.cargo_average_time.star",
                #     "line",
                #     "route",
                #     0.016,
                # ),
                LocalDemandParameter(
                    road_network,
                    transport_segments["roads"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    0.231,
                ),
                LocalDemandParameter(
                    waterway_network,
                    transport_segments["waterways"],
                    "transport.average_time.star",
                    "line",
                    "route",
                    -0.284,
                ),
            ],
            investment_multipliers=[
                [94608000, 1829, 1.051],
                [126144000, 1829, 1.051],
                [157680000, 1829, 1.051],
                [189216000, 1829, 1.054],
                [220752000, 1829, 1.054],
                [252288000, 1829, 1.054],
            ],
            atol=1e-5,
            rtol=1e-8,
        ),
        relaxation(waterway_network, "waterways", RELAXATION_FACTOR * (0.65 / 0.5)),
        combined_cargo_demand(waterway_network, "waterways"),
        total_cargo_demand(waterway_network, "waterways"),
        *kpi_models(
            waterway_network,
            "waterways",
            scenario_parameters,
            config=config,
            exclude_fuel_types=("petrol",),
            exclude_passenger_fuel_types=(
                "diesel",
                "h2",
                "electricity",
                "petrol",
            ),
        ),
        shortest_path(
            name="waterway_shortest_path",
            dataset=waterway_network,
            modality="waterways",
            cost_factor="transport.average_time",
            calculation_type="sum",
            calculation_input_attribute="shape.length",
            calculation_output_attribute="transport.shortest_path_length",
        ),
        udf(
            name="waterway_vkm",
            dataset=waterway_network,
            entity_group="virtual_node_entities",
            inputs={
                "cargo_demand": [None, "transport.cargo_demand"],
                "shortest_path_length": [None, "transport.shortest_path_length"],
            },
            functions=[
                {
                    "expression": "cargo_demand*shortest_path_length",
                    "output": [None, "transport.cargo_demand_vkm"],
                },
                {
                    "expression": f"cargo_demand*shortest_path_length*{CARGO_PEAK_MULTIPLIER}",
                    "output": [None, "transport.cargo_demand_vkm.peak_yearly"],
                },
            ],
        ),
        udf(
            name="waterway_segment_volume_to_capacity",
            dataset=waterway_network,
            entity_group=transport_segments["waterways"],
            inputs={
                "capacity": [None, "transport.segment_capacity.hours"],
                "flow": [None, "transport.cargo_vehicle_flow"],
            },
            functions=[
                {
                    "expression": "flow/capacity",
                    "output": [None, "transport.waterway_segment_volume_to_capacity"],
                }
            ],
        ),
        unit_conversion(
            "waterway_unit_conversion",
            UNIT_CONVERSION_COEFFICIENTS["waterways"],
            entity_groups=[
                UnitConversionEntityGroup(
                    waterway_network,
                    transport_segments["waterways"],
                    "flow",
                    "waterways",
                ),
                UnitConversionEntityGroup(
                    waterway_network, "virtual_node_entities", "od", "waterways"
                ),
            ],
        ),
        *peak_demand_models(
            waterway_network,
            "waterways",
            aggregation_dataset=total_area,
            passengers=False,
        ),
    ]
    return config


def generate_and_output(path=None):
    result = json.dumps(generate(), indent=2)
    if path is not None:

        path.write_text(result)
    else:
        print(result)


def test_main():
    expected = json.loads(
        pathlib.Path(__file__).parent.joinpath("master_config.json").read_text()
    )
    scenario_config = json.loads(json.dumps(generate()))
    assert scenario_config == expected


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_and_output(path)
