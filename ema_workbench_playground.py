# Description: This script is a playground for the EMA Workbench with the goal of multi-processing experiments.
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    Constant,
    ScalarOutcome,
    ema_logging,
    MultiprocessingEvaluator,
    perform_experiments,
)
import os

NUMBER_OF_EXPERIMENTS = 10


def some_model(scenario_file=None, dummy=None):
    """
    This is simple model that mimics the real model. It takes the same two parameters as the real model.
    Instead of the real model it just checks if the given directory and file exists and returns a dictionary with the results.
    """
    output_dir = scenario_file.replace(".json", "")

    scenario_file_exists = os.path.exists(scenario_file)
    output_dir_exists = os.path.exists(output_dir)
    return {
        f"{scenario_file}": scenario_file_exists,
        f"{output_dir}": output_dir_exists,
    }


def main():
    ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    model = Model("simpleModel", function=some_model)

    scenario_paths = {
        f"Scenario_{i}": f"data/scenarios_ema/ema_road_model_19_03_2024_scenario_{i}.json"
        for i in range(NUMBER_OF_EXPERIMENTS)
    }
    output_paths = {
        key: path.replace(".json", "") for key, path in scenario_paths.items()
    }

    # specify uncertainties
    model.uncertainties = [IntegerParameter("dummy", 0, 10000000)]

    # specify levers
    model.levers = [
        CategoricalParameter("scenario_file", list(scenario_paths.values()))
    ]

    # perform experiments sequentially
    results = perform_experiments(model, NUMBER_OF_EXPERIMENTS)

    pass

    # # perform experiments in parallel when sequential works
    # with MultiprocessingEvaluator(model) as evaluator:
    #     results = perform_experiments(model, NUMBER_OF_EXPERIMENTS, evaluator=evaluator)


if __name__ == "__main__":
    main()
