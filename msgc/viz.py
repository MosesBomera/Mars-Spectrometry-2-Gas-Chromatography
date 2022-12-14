import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def grid_plot(
    data,
    plot_func,
    iter_items,
    cols: Optional[int] = 4
):
    """
    Plot a grid of plots specified by the plot_func.
    Parameters
    ----------
    plot_func
        A function that defines the kind of chart to plot, it takes axes argument
        as the first parameter. This function should take strictly ax, data and 
        iter_item arguments. These arguments should be sufficient to define the 
        ploting logic.
    iter_items
        An iterable object that defines difference between the data plotted for each grid
        section. Could be a branch id, decile category.
    cols
        The number of columns to include in the grid.
    """
    # Check that plot_func required arguments exist.
    required_args = plot_func.__code__.co_varnames[:plot_func.__code__.co_argcount]
    if not required_args == ('ax', 'data', 'iter_item'):
        raise ValueError(
            f"Unknown parameter(s):\
            {set(required_args).difference(['ax', 'data', 'iter_item'])}.")

    # Rows.
    no_items = len(iter_items)
    rows, x = no_items // cols, no_items % cols
    rows = rows if x == 0 else rows + 1

    # Figsize.
    fig = plt.figure(figsize=(18, 5 * rows))

    # The loop.
    for i, iter_item in enumerate(iter_items):
        ax = fig.add_subplot(rows, cols, i + 1)

        # plot_func defines the plotting logic, any matplot chart.
        plot_func(ax=ax, data=data, iter_item=iter_item)

    plt.tight_layout()

def plot_mass_spectrum(
    ax, 
    data,
    iter_item
):
    """
    Given sample data, plot a spectogram at a given timestamp.

    Paramters
    ---------
    ax
        Matplotlib axes for plotting.
    data
        A dataframe containing relevant information for the plot.
    iter_item
        The descriptor for the data plotted in a given chart.
    """
    # Get the timestamp data.
    timestamp = data[data['time']==iter_item]
    ax.bar(timestamp['mass'], timestamp['intensity'], width=5)
    ax.set_title(f"{iter_item}")


def plot_spectrogram(
    ax, 
    data, 
    iter_item
):
    """
    Paramters
    ---------
    ax
        Matplotlib axes for plotting.
    data
        A dataframe containing relevant information for the plot.
    iter_item
        The descriptor for the data plotted in a given chart.
    """
    # To avoid over-engineering, this function expects the data to be stored 
    # under a folder named 'data/'.
    sample = pd.read_csv(f"data/{data.loc[iter_item].features_path}")

    # For visual clarity, we will round these intensity values to 
    # the nearest whole number and average the intensity.
    sample["mass"] = sample["mass"].round()
    sample = (
        sample.groupby(
            ["time", "mass"]
        )["intensity"].aggregate("mean").reset_index()
    )

    for m in sample["mass"].unique():
        plt.plot(
            sample[sample["mass"] == m]["time"],
            sample[sample["mass"] == m]["intensity"],
        )

    plt.title(iter_item)