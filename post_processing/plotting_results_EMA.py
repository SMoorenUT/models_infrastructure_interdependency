import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

matplotlib.use('TkAgg')

SIM_NAME = "ema_road_model_08_05_2024"
TRAFFIC_TYPE = "passenger"  # "cargo", "passenger" or "combined"
# Load the data
BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
DATA_DIR = (
    BASE_DIR.parent / "output_simulations" / SIM_NAME
)  # Folder with the results of the simulations as CSV files
DATA_SUBDIR_BRIDGES = (
    "bridges_ICratio"  # Subdirectory of the data directory where the results are stored
)
DATA_SUBDIR_ROAD_NETWORK = (
    f"road_network"  # Subdirectory of the data directory where the results are stored
)


def get_scenario_list():
    # Check if DATA_DIR exists
    if DATA_DIR.exists():
        # List CSV files in DATA_DIR and sort them
        scenarios = [
            file.name for file in Path(DATA_DIR / DATA_SUBDIR_BRIDGES).glob("*.csv")
        ]
        scenarios.sort()
        print(
            f"Found {len(scenarios)} scenarios in {DATA_DIR}. First scenario is called {scenarios[0]}"
        )
    else:
        raise ValueError(f"DATA_DIR {DATA_DIR} does not exist.")
    return scenarios


def load_results_bridges(entity_number: int) -> pd.DataFrame:
    # Get the list of scenarios
    scenarios = get_scenario_list()
    # Initialize the results dictionary
    results = {}
    # Load the results from all csv files the data directory. Take every the results in row N for every file and return them as a dictionary.
    directory = DATA_DIR / DATA_SUBDIR_BRIDGES
    for scenario in scenarios:
        scenario_name = "Scenario" + scenario.split("scenario")[-1].replace(".csv", "")
        df = pd.read_csv(directory / scenario, header=0, index_col=0)
        results[scenario_name] = df.iloc[entity_number]
    # Transform results into pandas dataframe
    df_results = pd.DataFrame(results)
    return df_results


def load_results_single_df(filename) -> pd.DataFrame:
    filepath = DATA_DIR / DATA_SUBDIR_ROAD_NETWORK / filename
    if not filepath.exists():
        raise ValueError(f"File {filepath} does not exist.")
    df = pd.read_csv(filepath, header=0, index_col=0)
    return df


def plot_results_bridge(
    df_results: pd.DataFrame, entity_number: int, save_fig: bool = False
) -> None:
    # Check input types
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("df_results must be a pandas DataFrame")
    if not isinstance(entity_number, int):
        raise TypeError("entity_number must be an integer")
    if not isinstance(save_fig, bool):
        raise TypeError("save_fig must be a boolean")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [9, 1]}, sharey=True
    )

    ax1 = plt.subplot(1, 2, 1)
    for scenario in df_results.columns:
        ax1.plot(df_results.index, df_results[scenario], label=scenario)
    ax1.set_ylim(0, df_results.max().max())
    ax1.set_xticks(np.arange(1, 32, 5))
    ax1.minorticks_on()
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax1.set_xlim(df_results.index.min(), df_results.index.max())
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Volume to Capacity Ratio")
    ax1.set_title(
        f"Bridge {entity_number}: Scenario Ensemble for Volume to Capacity Ratio"
    )
    ax1.grid(True)
    ax1.axvline(
        x=df_results.index[0], color="gray", linestyle="--", alpha=0.5
    )  # Add vertical gridline at the first value

    # Add a second subplot for KDE
    ax2 = plt.subplot(1, 2, 2)
    values_2030 = df_results.iloc[11]
    values_2050 = df_results.iloc[-1]
    # ax2.hist(values_2050, bins=10, edgecolor='black', orientation='horizontal')
    sns.kdeplot(y=values_2030, fill=True)
    sns.kdeplot(y=values_2050, fill=True)
    ax2.legend(["2030", "2050"])
    ax2.set_xlabel("Frequency")
    ax2.set_title("Kernel density estimation")
    ax2.grid(True)

    if save_fig:
        plt.savefig(PLOT_DIR / f"bridge_{entity_number}_volume_to_capacity_ratio.png")
    plt.tight_layout()
    plt.show()


def plot_results_road_network(df_results: pd.DataFrame, save_fig: bool = False) -> None:
    # Check input types
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("df_results must be a pandas DataFrame")
    if not isinstance(save_fig, bool):
        raise TypeError("save_fig must be a boolean")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [9, 1]}, sharey=True
    )

    # Draw the first subplot
    ax1 = plt.subplot(1, 2, 1)
    for scenario in df_results.columns:
        ax1.plot(df_results.index, df_results[scenario], label=scenario)
    ax1.set_ylim(0, df_results.max().max())
    ax1.minorticks_on()
    ax1.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    ax1.set_xlim(df_results.index.min(), df_results.index.max())
    ax1.set_xlabel("Year")
    ax1.set_ylabel(f"{TRAFFIC_TYPE.capitalize()} vehicle kilometers (vkm)")
    ax1.set_title(
        f"Scenario Ensemble for {TRAFFIC_TYPE} vehicle kilometers (vkm). Simulation: {SIM_NAME}"
    )
    ax1.grid(True)
    ax1.axvline(
        x=df_results.index[0], color="gray", linestyle="--", alpha=0.5
    )  # Add vertical gridline at the first value

    # Add a second subplot for KDE
    ax2 = plt.subplot(1, 2, 2)
    values_2030 = df_results.iloc[11]
    values_2050 = df_results.iloc[-1]
    sns.kdeplot(y=values_2030, fill=True)
    sns.kdeplot(y=values_2050, fill=True)
    ax2.legend(["2030", "2050"])
    ax2.set_xlabel("Frequency")
    ax2.set_title("Kernel density estimation")
    ax2.grid(True)

    if save_fig:
        plt.savefig(PLOT_DIR / f"{TRAFFIC_TYPE}_VKM.png")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots
    plt.show()


def process_bridge_results():
    attribute = "transport.volume_to_capacity_ratio"
    entity_number = 2
    df_results = load_results_bridges(entity_number)
    plot_results_bridge(df_results, entity_number, save_fig=False)


def process_road_network_results(filename, save_fig=False):
    df_results = load_results_single_df(filename)
    plot_results_road_network(df_results, save_fig=save_fig)

def main():
    filename = f"{TRAFFIC_TYPE}_vkm.csv"
    # process_bridge_results()
    process_road_network_results(filename, save_fig=False)

if __name__ == "__main__":
    main()