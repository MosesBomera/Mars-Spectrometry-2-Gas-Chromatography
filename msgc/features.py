import pandas as pd
from sklearn.preprocessing import minmax_scale

# PRE-PROCESSING

def get_time_mass_stats(fpath):
    """Summary statistics for time and m/z for training set."""
    df = pd.read_csv(f"data/{fpath}")
    # Time
    time_min = df["time"].min()
    time_max = df["time"].max()
    time_range = time_max - time_min
    # m/z
    mass_min = df["mass"].min()
    mass_max = df["mass"].max()
    mass_range = mass_max - mass_min
    # Statistics
    return time_min, time_max, time_range, mass_min, mass_max, mass_range


def drop_frac_and_He(df):
    """
    Rounds fractional m/z values, drops m/z values > 350, and drops carrier gas m/z

    Args:
        df: a dataframe representing a single sample, containing m/z values

    Returns:
        The dataframe without fractional m/z and carrier gas m/z
    """
    df = df.copy(deep=True)

    # rounds m/z fractional values
    df["rounded_mass"] = df["mass"].transform(round)

    # aggregates across rounded values
    df = df.groupby(["time", "rounded_mass"])["intensity"].aggregate("mean").reset_index()

    # drop m/z values greater than 350
    df = df[df["rounded_mass"] <= 350]

    # drop carrier gas
    df = df[df["rounded_mass"] != 4]

    return df


def remove_background_intensity(df):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'mass' and 'intensity' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """

    df["intensity_minsub"] = df.groupby(["rounded_mass"])["intensity"].transform(
        lambda x: (x - x.min())
    )

    return df


def scale_intensity(df):
    """
    Scale abundance from 0-1 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["int_minsub_scaled"] = minmax_scale(df["intensity_minsub"].astype(float))

    return df


def preprocess_sample(df):
    # Preprocess function
    df = drop_frac_and_He(df)
    df = remove_background_intensity(df)
    df = scale_intensity(df)
    return df

# FEATURE ENGINEERING

def int_per_timebin(df):

    """
    Transforms dataset to take the preprocessed max abundance for each
    time range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """
    # Create a series of time bins
    timerange = pd.interval_range(start=0, end=25, freq=0.5)

    # Make dataframe with rows that are combinations of all temperature bins and all m/z values
    allcombs = list(itertools.product(timerange, [*range(0, 350)]))

    allcombs_df = pd.DataFrame(allcombs, columns=["time_bin", "rounded_mass"])

    # Bin times
    df["time_bin"] = pd.cut(df["time"], bins=timerange)

    # Combine with a list of all time bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["time_bin", "rounded_mass"], how="left")

    # Aggregate to time bin level to find max
    df = df.groupby(["time_bin", "rounded_mass"]).max("int_minsub_scaled").reset_index()

    # Fill in 0 for intensity values without information
    df = df.replace(np.nan, 0)

    # Reshape so each row is a single sample
    df = df.pivot_table(
        columns=["rounded_mass", "time_bin"], values=["int_minsub_scaled"]
    )

    return df