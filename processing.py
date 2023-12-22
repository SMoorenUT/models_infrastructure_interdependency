from pathlib import Path

from PropertySpec import attributes
from movici_simulation_core.postprocessing.results import SimulationResults
from movici_simulation_core.utils.moment import TimelineInfo, string_to_datetime
from operators import save_as_pickle, SingleAttributeSliceOperator

# input path to init_data and output
BASE_DIR = Path(__file__).parent
INIT_DATA_DIR = BASE_DIR / "data/init_data/"
UPDATES_DIR = BASE_DIR / "new_core/out/rail_road_waterway/experiments/"
PROCESSED_DIR = BASE_DIR / "data/scenarios/"

scenarios = list()
[scenarios.append(scenario_folder.name) for scenario_folder in UPDATES_DIR.glob('*')]  # get scenario folders' name
scenarios = ["infraconomy"]

if __name__ == "__main__":
    timeline_info = TimelineInfo(
        reference=string_to_datetime("2019").timestamp(), time_scale=1, start_time=0
    )
    attributes = attributes
    for scenario in scenarios:
        SCENARIO_UPDATES_DIR = UPDATES_DIR / scenario
        SCENARIO_PROCESSED_DIR = PROCESSED_DIR / scenario

        results = SimulationResults(INIT_DATA_DIR, SCENARIO_UPDATES_DIR, attributes=attributes,
                                    timeline_info=timeline_info)
        dataset = results.get_dataset("road_network")

        '''
            Slicing a dataset over a specific timestamp
        '''
        timestamp = "2020"
        entity = "road_segment_entities"
        my_slice = dataset.slice(entity, timestamp)
        # my_slice_name = entity + "_" + timestamp + "_" + dataset.strategy_name

        # print(
        #     "Slicing a dataset over a specific timestamp",
        #     scenario,
        #     my_slice.keys(),
        #     sep="\n",
        # )
        # save_as_pickle(data= my_slice, processed_dir= SCENARIO_PROCESSED_DIR, file_name= my_slice_name )

        '''
            Slicing a dataset over a specific attribute
        '''

        print(
            "Slicing a dataset over a specific attribute",
            dataset.slice(
                "road_segment_entities", attribute="transport.cargo_flow"
            ),
            sep="\n",
        )

        # # {
        # #     "timestamps": [1,2,23]
        # #     "id": np.array([4,5,6])
        # #     "data": [
        # #         {"data": np.array([5,6,7])},
        # #         {"data": np.array([5,6,7])},
        # #     ]
        # # }
        #
        '''
            Slicing a dataset over a specific entity
        '''
        # print(
        #     "Slicing a dataset over a specific entity (entity ID 6)",
        #     dataset.slice("antenna_entities", entity_selector=6),
        #     sep="\n",
        # )

        dic_id_6 = \
            {
                2019: {
                    "i/c": 1,
                    "noise": 50
                },
                2020: {
                    "i/c": 1.5,
                    "noise": 51
                }
            }
