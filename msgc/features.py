import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

# PRE-PROCESSING

def get_time_mass_stats(fpath):
    """Summary statistics for time and m/z for training set."""
    data = pd.read_csv(f"data/{fpath}")
    # Time
    time_min = data["time"].min()
    time_max = data["time"].max()
    time_range = time_max - time_min
    # m/z
    mass_min = data["mass"].min()
    mass_max = data["mass"].max()
    mass_range = mass_max - mass_min
    # Statistics
    return time_min, time_max, time_range, mass_min, mass_max, mass_range


def drop_frac_and_He(
    data, 
    mass_cutoff = None
):
    """
    Rounds fractional m/z values, drops m/z values > mass_cutoff, and drops carrier gas m/z

    Args:
        data: a dataframe representing a single sample, containing m/z values.
        mass_cutoff: integer specifying the rounded mass cutoff.

    Returns:
        The dataframe without fractional m/z and carrier gas m/z.
    """
    data = data.copy(deep=True)

    # rounds m/z fractional values
    data["rounded_mass"] = data["mass"].transform(round)

    # aggregates across rounded values
    data = data.groupby(["time", "rounded_mass"])["intensity"].aggregate("mean").reset_index()

    # drop m/z values greater than mass_cutoff
    if mass_cutoff:
        data = data[data["rounded_mass"] <= mass_cutoff].reset_index(drop=True)

    # drop carrier gas.
    data = data[data["rounded_mass"] != 4]

    return data


def remove_background_intensity(data):
    """
    Subtracts minimum abundance value

    Args:
        data: dataframe with 'mass' and 'intensity' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    data["intensity_minsub"] = data.groupby(["rounded_mass"])["intensity"].transform(
        lambda x: (x - x.min())
    )

    return data


def scale_intensity(data):
    """
    Scale abundance from 0-1 according to the min and max values across entire sample

    Args:
        data: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    data["int_minsub_scaled"] = minmax_scale(data["intensity_minsub"].astype(float))

    return data


def preprocess_sample(
    data,
    mass_cutoff = None
):
    # Preprocess function
    data = drop_frac_and_He(data, mass_cutoff=mass_cutoff)
    data = remove_background_intensity(data)
    data = scale_intensity(data)
    return data

# FEATURE ENGINEERING

def int_per_timebin(data):

    """
    Transforms dataset to take the preprocessed max abundance for each
    time range for each m/z value

    Args:
        data: dataframe to transform

    Returns:
        transformed dataframe
    """
    # Create a series of time bins
    timerange = pd.interval_range(start=0, end=25, freq=0.5)

    # Make dataframe with rows that are combinations of all time bins and all m/z values
    allcombs = list(itertools.product(timerange, [*range(0, 350)]))

    allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "rounded_mass"])

    # Bin times
    data["time_bin"] = pd.cut(data["time"], bins=timerange)

    # Combine with a list of all time bin-m/z value combinations
    data = pd.merge(allcombs_df, data, on=["time_bin", "rounded_mass"], how="left")

    # Aggregate to time bin level to find max
    data = data.groupby(["time_bin", "rounded_mass"]).max("int_minsub_scaled").reset_index()

    # Fill in 0 for intensity values without information
    data = data.replace(np.nan, 0)

    # Reshape so each row is a single sample
    data = data.pivot_table(
        columns=["rounded_mass", "time_bin"], values=["int_minsub_scaled"]
    )

    return data