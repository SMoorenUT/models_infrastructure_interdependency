import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
from scipy.stats import median_abs_deviation
from tabulate import tabulate
from translator_id_reference import reference_dict

# Below is a workaround to import sdf from the parent directory. This is necessary to run the script from the command line.
import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)
from analysis.scenario_discovery_functions import scenario_discovery_functions as sdf

matplotlib.use("TkAgg")

entity_number = 671  # Number of the bridge to plot
SIM_NAME = "ema_road_model_17_07_2025"  # Name of the simulation the results are from
TRAFFIC_TYPE = "cargo"  # "cargo", "passenger" or "combined"
YEARS_KDE = [
    2030,
    2040,
    2050,
]  # Years for which to plot the kernel density estimation. TODO: Implement None
color_palette = "Spectral"  # Colors for the scenarios (was Spectral 100)
colors2 = sns.color_palette(
    "Dark2", len(YEARS_KDE)
)  # Colors for the years of the KDE plot
BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots" / SIM_NAME
DATA_DIR = (
    BASE_DIR.parent / "output_simulations" / SIM_NAME
)  # Folder with the results of the simulations as CSV files
DATA_SUBDIR_BRIDGES = Path(
    "bridges/individual"  # Subdirectory of the data directory where the results are stored
)
DATA_SUBDIR_ROAD_NETWORK = (
    f"road_network"  # Subdirectory of the data directory where the results are stored
)
# Declare variables for the complex binarization of the IC ratio of two years
IC_DF = pd.read_csv(
    DATA_DIR / f"road_network/{TRAFFIC_TYPE}_vkt.csv",
    index_col=0,
)
YEAR_1 = 2035
CONDITION_1 = "<"
THRESHOLD_1 = 0.25
YEAR_2 = 2050
CONDITION_2 = "<"
THRESHOLD_2 = 0.25

PLOT_MODE = "policy"  # "all", "pop", "policy"
FONTSIZE = 20  # Font size for the plot
SAVE_FIG = False  # Boolean to determine whether to save the figure or not


def get_cluster_name(condition, threshold):
    if condition == ">" and threshold == 0.75:
        name = "high"
    elif condition == "<" and threshold == 0.25:
        name = "low"
    else:
        raise ValueError("Condition and threshold combination not recognized.")
    return name


analysis_name = (
    str(YEAR_1)
    + "_"
    + get_cluster_name(CONDITION_1, THRESHOLD_1)
    + "_"
    + str(YEAR_2)
    + "_"
    + get_cluster_name(CONDITION_2, THRESHOLD_2)
)
analysis_name_human_readable = (
    str(YEAR_1)
    + "-"
    + get_cluster_name(CONDITION_1, THRESHOLD_1)
    + " & "
    + str(YEAR_2)
    + "-"
    + get_cluster_name(CONDITION_2, THRESHOLD_2)
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


def load_results_single_df(
    filename: str, subdir: str = "road_network", transpose: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Load results from a single CSV file.

    Args:
        filename (str): Name of the file to load.
        subdir (str): "road_network" or "bridges".
        **kwargs: Additional keyword arguments to pass to pd.read_csv.

    Returns:
        pd.DataFrame: DataFrame with the results.
    """
    if subdir == "road_network":
        filepath = DATA_DIR / DATA_SUBDIR_ROAD_NETWORK / filename
    elif subdir == "bridges":
        filepath = DATA_DIR / DATA_SUBDIR_BRIDGES / filename
    else:
        raise ValueError("subdir must be either 'road_network' or 'bridges'")
    if not filepath.exists():
        raise ValueError(f"File {filepath} does not exist.")
    df = pd.read_csv(filepath, header=0, **kwargs)
    df = df.T if transpose else df
    return df


def plot_results_bridge(
    df_results: pd.DataFrame,
    entity_number: int,
    years_kde: list[int],
    plot_mode: str = "all",  # "all", "pop", "policy"
    subset_pop_out: np.ndarray = None,
    save_fig: bool = False,
) -> None:
    # Validate plot_mode
    if plot_mode not in ("all", "pop", "policy"):
        raise ValueError("plot_mode must be one of: 'all', 'pop', 'policy'")
    # For "all" mode, ignore any subset provided
    if plot_mode == "all":
        subset_pop_out = None
    save_fig = False
    # Check input types
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("df_results must be a pandas DataFrame")
    if not isinstance(entity_number, int):
        raise TypeError("entity_number must be an integer")
    if not isinstance(years_kde, list) or not all(
        isinstance(year, int) for year in years_kde
    ):
        raise TypeError("years_kde must be a list of integers")
    if not isinstance(save_fig, bool):
        raise TypeError("save_fig must be a boolean")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [9, 1]}, sharey=True
    )
    # Set the figure size
    fig.set_size_inches(18, 9)

    # Draw the first subplot
    ax1 = plt.subplot(1, 2, 1)

    # Add vertical lines and year annotations for each year in YEARS_KDE
    for idx, year in enumerate(YEARS_KDE):
        year_position = df_results.index.get_loc(year)
        ax1.axvline(x=year, color=colors2[idx], linestyle="--", alpha=0.8)
        max_value = df_results.iloc[year_position].max()
        if year == df_results.index.max():
            ax1.text(
                year,
                max_value,
                str(year),
                color=colors2[idx],
                fontsize=FONTSIZE * 0.8,
                ha="center",
                va="bottom",
            )
        else:
            ax1.text(
                year,
                max_value + 0.1,
                str(year),
                color=colors2[idx],
                fontsize=FONTSIZE * 0.8,
                ha="left",
                va="bottom",
            )

    def plot_result_lines(dataframe, color_cycle):
        # Plot each scenario as a line
        if isinstance(color_cycle, str):
            color = color_cycle
            for scenario in dataframe.columns:
                ax1.plot(
                    dataframe.index,
                    dataframe[scenario],
                    label=scenario,
                    linestyle=":",
                    linewidth=0.3,
                    color=color,
                    alpha=0.5,
                    zorder=np.random.rand(),
                )
        else:
            for scenario in dataframe.columns:
                color = next(color_cycle)
                ax1.plot(
                    dataframe.index,
                    dataframe[scenario],
                    label=scenario,
                    linewidth=0.3,
                    color=color,
                    alpha=1.0,
                    zorder=np.random.rand(),
                )

    if plot_mode == "all":
        # Cycle through colors for the scenarios
        color_cycle = itertools.cycle(
            sns.color_palette(color_palette, min(len(df_results), 100))
        )
        plot_result_lines(df_results, color_cycle=color_cycle)
        ax1.set_title(
            f"Scenario ensemble bridge {entity_number}",
            fontsize=FONTSIZE * 1.5,
        )
    elif plot_mode == "pop":
        df_results_popped_out = df_results.drop(
            columns=df_results.columns[~subset_pop_out]
        )
        df_results_background = df_results.drop(
            columns=df_results.columns[subset_pop_out]
        )

        color_cycle = itertools.cycle(
            sns.color_palette(color_palette, min(len(df_results_popped_out), 100))
        )
        plot_result_lines(df_results_background, color_cycle="gray")
        plot_result_lines(df_results_popped_out, color_cycle=color_cycle)

        number_highlighted_scenarios = subset_pop_out.sum()
        ax1.text(
            0.01,
            0.99,
            f"{number_highlighted_scenarios} scenarios in cluster",
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        ax1.set_title(
            f"Scenario ensemble bridge {entity_number}. {analysis_name_human_readable}.\nHighlighted scenarios in cluster",
            fontsize=FONTSIZE * 1.5,
        )
    elif plot_mode == "policy":
        # create mask: 500 zeros followed by 499 ones, convert to boolean and adapt to number of scenarios
        mask = [0] * 500 + [1] * 499
        if len(mask) != df_results.shape[1]:
            if len(mask) < df_results.shape[1]:
                mask += [0] * (df_results.shape[1] - len(mask))
            else:
                mask = mask[: df_results.shape[1]]
        subset_pop_out = np.array(mask, dtype=bool)
        # Split the dataframe into two based on the mask
        df_results_policy_0 = df_results.drop(
            columns=df_results.columns[~subset_pop_out]
        )
        df_results_policy_1 = df_results.drop(
            columns=df_results.columns[subset_pop_out]
        )

        # Create color cycles for both subsets
        color_cycle_0 = itertools.cycle(
            sns.color_palette("autumn", min(len(df_results_policy_0), 100))
        )
        color_cycle_1 = itertools.cycle(
            sns.color_palette("winter", min(len(df_results_policy_1), 100))
        )

        # Plot background scenarios in ocean color
        plot_result_lines(df_results_policy_0, color_cycle=color_cycle_0)

        # Plot popped out scenarios in red
        plot_result_lines(df_results_policy_1, color_cycle=color_cycle_1)

        number_highlighted_scenarios = subset_pop_out.sum()
        ax1.text(0.01, 0.99, f"{number_highlighted_scenarios} scenarios in cluster"),
        # Set title
        ax1.set_title(
            f"Scenario ensemble bridge {entity_number}.",
            fontsize=FONTSIZE * 1.5,
        )
    # Set the y-axis limit
    ax1.set_ylim(0, df_results.max().max())

    # Enable minor ticks on the x-axis
    ax1.minorticks_on()
    ax1.tick_params(axis="x", which="minor", bottom=True, top=False)
    ax1.tick_params(
        axis="both", which="major", labelsize=FONTSIZE * 0.8
    )  # Change font size of tick labels

    # Add grid lines
    ax1.grid(which="major", linestyle=":", linewidth="0.5", color="gray")

    # Set the x-axis limit
    ax1.set_xlim(df_results.index.min(), df_results.index.max() + 0.1)

    # Set the x and y labels with larger font sizes
    ax1.set_xlabel("Year", fontsize=FONTSIZE)
    ax1.set_ylabel("Volume to Capacity Ratio", fontsize=FONTSIZE)

    # Enable the grid
    ax1.grid(True)

    # Customize the spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add a second subplot for KDE
    ax2 = plt.subplot(1, 2, 2)
    for idx, year in enumerate(years_kde):
        if plot_mode != "pop":
            values = df_results.loc[year]
        else:
            # Use the popped out dataframe for KDE
            values = df_results_popped_out.loc[year]
        sns.kdeplot(y=values, fill=True, label=str(year), color=colors2[idx])
    ax2.legend()
    ax2.set_xlabel("Frequency", fontsize=FONTSIZE)
    ax2.set_title("KDE", fontsize=FONTSIZE)
    ax2.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots
    if save_fig:
        if subset_pop_out is None:
            plt.savefig(
                PLOT_DIR / f"bridge_{entity_number}_volume_to_capacity_ratio.tiff",
                dpi=1200,
                bbox_inches="tight",
            )
            plt.savefig(
                PLOT_DIR / f"bridge_{entity_number}_volume_to_capacity_ratio.jpeg",
                dpi=1200,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                PLOT_DIR / f"bridge_{entity_number}_clusters_{analysis_name}.tiff",
                dpi=1200,
                bbox_inches="tight",
            )
            plt.savefig(
                PLOT_DIR / f"bridge_{entity_number}_clusters_{analysis_name}.jpeg",
                dpi=1200,
                bbox_inches="tight",
            )
    print(f"Bridge ID: {entity_number}")
    print(f"Bridge reference: {reference_dict(entity_number)}")
    plt.show()


def plot_results_road_network(
    df_results: pd.DataFrame, years_kde: list[int], save_fig: bool = False
) -> None:
    # Check input types
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("df_results must be a pandas DataFrame")
    if not isinstance(years_kde, list) or not all(
        isinstance(year, int) for year in years_kde
    ):
        raise TypeError("years_kde must be a list of integers")
    if not isinstance(save_fig, bool):
        raise TypeError("save_fig must be a boolean")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [9, 1]}, sharey=True
    )
    # Set the figure size
    fig.set_size_inches(18, 9)

    # Draw the first subplot
    ax1 = plt.subplot(1, 2, 1)
    # Cycle through colors for the scenarios
    color_cycle = itertools.cycle(
        sns.color_palette(color_palette, min(len(df_results), 100))
    )

    # Add vertical lines and year annotations for each year in YEARS_KDE
    for idx, year in enumerate(YEARS_KDE):
        year_position = df_results.index.get_loc(year)
        ax1.axvline(x=year, color=colors2[idx], linestyle="--", alpha=0.8)
        year_position = df_results.index.get_loc(year)
        max_value = df_results.iloc[year_position].max()
        if year == df_results.index.max():
            ax1.text(
                year,
                max_value,
                str(year),
                color=colors2[idx],
                fontsize=FONTSIZE * 0.8,
                ha="center",
                va="bottom",
            )
        else:
            ax1.text(
                year,
                max_value + 1e8,
                str(year),
                color=colors2[idx],
                fontsize=FONTSIZE * 0.8,
                ha="left",
                va="bottom",
            )

    # Plot each scenario with a different color
    for scenario in df_results.columns:
        color = next(color_cycle)
        ax1.plot(
            df_results.index,
            df_results[scenario],
            label=scenario,
            linewidth=0.5,
            color=color,
        )

    # Set the y-axis limit
    ax1.set_ylim(0, df_results.max().max())

    # Enable minor ticks on the x-axis
    ax1.minorticks_on()
    ax1.tick_params(axis="x", which="minor", bottom=True, top=False)
    ax1.tick_params(
        axis="both", which="major", labelsize=FONTSIZE * 0.8
    )  # Change font size of tick labels

    # Add grid lines
    ax1.grid(which="major", linestyle=":", linewidth="0.5", color="gray")

    # Set the x-axis limit
    ax1.set_xlim(df_results.index.min(), df_results.index.max() + 0.01)

    # Set the x and y labels with larger font sizes
    ax1.set_xlabel("Year", fontsize=FONTSIZE)
    ax1.set_ylabel(
        f"{TRAFFIC_TYPE.capitalize()} vehicle kilometers travelled (VKT)",
        fontsize=FONTSIZE,
    )

    # Set the title of the plot
    ax1.set_title(
        f"Scenario Ensemble for {TRAFFIC_TYPE} vehicle kilometers travelled (VKT)",
        fontsize=FONTSIZE * 1.5,
    )

    # Enable the grid
    ax1.grid(True)

    # Customize the spines
    ax1.spines["top"].set_visible(False)
    if df_results.index.max() in YEARS_KDE:
        ax1.spines["right"].set_visible(False)

    # Use ScalarFormatter to display y-axis values in scientific notation
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add a second subplot for KDE
    ax2 = plt.subplot(1, 2, 2)
    for idx, year in enumerate(years_kde):
        values = df_results.loc[year]
        sns.kdeplot(y=values, fill=True, label=str(year), color=colors2[idx])
    ax2.legend()
    ax2.set_xlabel("Frequency", fontsize=FONTSIZE)
    ax2.set_title("KDE", fontsize=FONTSIZE)
    ax2.grid(which="major", linestyle=":", linewidth="0.5", color="gray")
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Finalize the plot and show it. Also save the figure if save_fig is True
    plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots
    plt.tight_layout()
    if save_fig:
        for fig_format in ["tiff", "jpeg"]:
            plt.savefig(
                PLOT_DIR / f"{TRAFFIC_TYPE}_VKT.{fig_format}",
                dpi=1200,
                bbox_inches="tight",
            )
        print(f"Figure saved as {TRAFFIC_TYPE}_VKT.{fig_format} at {PLOT_DIR}/")
    plt.show()


def process_bridge_results():
    attribute = "transport.volume_to_capacity_ratio"
    entity_number = 2
    df_results = load_results_bridges(entity_number)
    plot_results_bridge(df_results, entity_number, YEARS_KDE, save_fig=False)


def process_road_network_results(filename, years_kde, save_fig=False):
    df_results = load_results_single_df(filename, index_col=0)
    sdf.get_spread_stats(df_results=df_results, years_kde=years_kde)
    plot_results_road_network(df_results, years_kde, save_fig=save_fig)


def process_bridge_IC_ratio(entity_number, years_kde, save_fig=False):
    df_results = load_results_single_df(
        filename=f"{SIM_NAME}_Bridge_{entity_number}_ICratio.csv",
        subdir="bridges",
        transpose=True,
        index_col=0,
    )
    df_results.index = df_results.index.astype(int)
    # binary_array_coi = sdf.binarize_array_complex(
    #     ic_dataframe=IC_DF,
    #     year_1=YEAR_1,
    #     threshold_1=THRESHOLD_1,
    #     condition_1=CONDITION_1,
    #     year_2=YEAR_2,
    #     threshold_2=THRESHOLD_2,
    #     condition_2=CONDITION_2,
    # )
    plot_results_bridge(
        df_results, entity_number, years_kde, plot_mode=PLOT_MODE, save_fig=save_fig
    )


def main():
    # process_road_network_results(
    #     filename=f"{TRAFFIC_TYPE}_vkt.csv", years_kde=YEARS_KDE, save_fig=SAVE_FIG
    # )

    process_bridge_IC_ratio(entity_number, years_kde=YEARS_KDE, save_fig=SAVE_FIG)


if __name__ == "__main__":
    main()
