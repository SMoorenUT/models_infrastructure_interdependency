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

matplotlib.use("TkAgg")

entity_number = 594
SIM_NAME = "ema_road_model_08_05_2024"
TRAFFIC_TYPE = "combined"  # "cargo", "passenger" or "combined"
YEARS_KDE = [
    2030,
    2040,
    2050,
]  # Years for which to plot the kernel density estimation. TODO: Implement None
SAVE_FIG = False  # Boolean to determine whether to save the figure or not
colors1 = sns.color_palette("Spectral", 100)  # Colors for the scenarios
colors2 = sns.color_palette(
    "Dark2", len(YEARS_KDE)
)  # Colors for the years of the KDE plot
BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
DATA_DIR = (
    BASE_DIR.parent / "output_simulations" / SIM_NAME
)  # Folder with the results of the simulations as CSV files
DATA_SUBDIR_BRIDGES = Path(
    "bridges/individual"  # Subdirectory of the data directory where the results are stored
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
    save_fig: bool = False,
) -> None:
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

    # Draw the first subplot
    ax1 = plt.subplot(1, 2, 1)
    # Cycle through colors for the scenarios
    color_cycle = itertools.cycle(colors1)

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
                fontsize=8,
                ha="center",
                va="bottom",
            )
        else:
            ax1.text(
                year,
                max_value + 0.1,
                str(year),
                color=colors2[idx],
                fontsize=8,
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
            alpha=1,
        )

    # Set the y-axis limit
    ax1.set_ylim(0, df_results.max().max())

    # Enable minor ticks on the x-axis
    ax1.minorticks_on()
    ax1.tick_params(axis="x", which="minor", bottom=True, top=False)

    # Add grid lines
    ax1.grid(which="major", linestyle=":", linewidth="0.5", color="gray")

    # Set the x-axis limit
    ax1.set_xlim(df_results.index.min(), df_results.index.max() + 0.1)

    # Set the x and y labels
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Volume to Capacity Ratio")

    # Set the title of the plot
    ax1.set_title(
        f"Bridge {entity_number}: Scenario Ensemble for Volume to Capacity Ratio"
    )

    # Enable the grid
    ax1.grid(True)

    # Customize the spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add a second subplot for KDE
    ax2 = plt.subplot(1, 2, 2)
    for idx, year in enumerate(years_kde):
        values = df_results.loc[year]
        sns.kdeplot(y=values, fill=True, label=str(year), color=colors2[idx])
    ax2.legend()
    ax2.set_xlabel("Frequency")
    ax2.set_title("Kernel density estimation")
    ax2.grid(True)

    if save_fig:
        plt.savefig(PLOT_DIR / f"bridge_{entity_number}_volume_to_capacity_ratio.png")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots
    plt.show()
    print(f"Bridge ID: {entity_number}")
    print()


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
    color_cycle = itertools.cycle(colors1)

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
                fontsize=8,
                ha="center",
                va="bottom",
            )
        else:
            ax1.text(
                year,
                max_value + 1e8,
                str(year),
                color=colors2[idx],
                fontsize=8,
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

    # Add grid lines
    ax1.grid(which="major", linestyle=":", linewidth="0.5", color="gray")

    # Set the x-axis limit
    ax1.set_xlim(df_results.index.min(), df_results.index.max() + 0.01)

    # Set the x and y labels
    ax1.set_xlabel("Year")
    ax1.set_ylabel(f"{TRAFFIC_TYPE.capitalize()} vehicle kilometers travelled (VKT)")

    # Set the title of the plot
    ax1.set_title(
        f"Scenario Ensemble for {TRAFFIC_TYPE} vehicle kilometers travelled (VKT)"
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
    ax2.set_xlabel("Frequency")
    ax2.set_title("Kernel density estimation", fontsize=9)
    ax2.grid(which="major", linestyle=":", linewidth="0.5", color="gray")
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Finalize the plot and show it. Also save the figure if save_fig is True
    plt.subplots_adjust(wspace=0.05)  # Adjust the spacing between subplots
    plt.tight_layout()
    if save_fig:
        fig_format = "svg"
        plt.savefig(
            PLOT_DIR / f"{TRAFFIC_TYPE}_VKT.{fig_format}", dpi=600, bbox_inches="tight"
        )
        print(f"Figure saved as {TRAFFIC_TYPE}_VKT.{fig_format} at {PLOT_DIR}/")
    plt.show()


def get_spread_stats(df_results: pd.DataFrame, years_kde: list[int]) -> pd.DataFrame:
    # Check input types
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("df_results must be a pandas DataFrame")
    if not isinstance(years_kde, list) or not all(
        isinstance(year, int) for year in years_kde
    ):
        raise TypeError("years_kde must be a list of integers")

    # Initialize the DataFrame to store the spread statistics
    spread_stats = pd.DataFrame(
        columns=["Year", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "IQR", "MAD"]
    )
    spread_stats["Year"] = years_kde
    spread_stats.set_index("Year", inplace=True)

    # Calculate the spread statistics for each year in YEARS_KDE
    for year in years_kde:
        values = df_results.loc[year]
        spread_stats.loc[year, "Mean"] = values.mean()
        spread_stats.loc[year, "Std"] = values.std()
        spread_stats.loc[year, "Min"] = values.min()
        spread_stats.loc[year, "25%"] = values.quantile(0.25)
        spread_stats.loc[year, "50%"] = values.median()
        spread_stats.loc[year, "75%"] = values.quantile(0.75)
        spread_stats.loc[year, "Max"] = values.max()
        spread_stats.loc[year, "IQR"] = values.quantile(0.75) - values.quantile(0.25)
        spread_stats.loc[year, "MAD"] = median_abs_deviation(values)

    print(tabulate(spread_stats, headers="keys", tablefmt="pretty"))
    return spread_stats


def process_bridge_results():
    attribute = "transport.volume_to_capacity_ratio"
    entity_number = 2
    df_results = load_results_bridges(entity_number)
    plot_results_bridge(df_results, entity_number, YEARS_KDE, save_fig=False)


def process_road_network_results(filename, years_kde, save_fig=False):
    df_results = load_results_single_df(filename, index_col=0)
    get_spread_stats(df_results=df_results, years_kde=years_kde)
    plot_results_road_network(df_results, years_kde, save_fig=save_fig)


def process_bridge_IC_ratio(entity_number, years_kde, save_fig=False):
    df_results = load_results_single_df(
        filename=f"ema_road_model_08_05_2024_Bridge_{entity_number}_ICratio.csv",
        subdir="bridges",
        transpose=True,
        index_col=0,
    )
    df_results.index = df_results.index.astype(int)
    plot_results_bridge(df_results, entity_number, years_kde, save_fig)


def main():
    # process_road_network_results(filename= f"{TRAFFIC_TYPE}_vkm.csv", years_kde=YEARS_KDE, save_fig=SAVE_FIG)
    # process_bridge_results()
    process_bridge_IC_ratio(entity_number, years_kde=YEARS_KDE, save_fig=SAVE_FIG)


if __name__ == "__main__":
    main()
