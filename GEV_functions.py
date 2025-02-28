"""

GEV_functions.py

This file contains functions for fitting the Generalized Extreme Value (GEV) distribution to data.


Ben Hutchins, 2025

"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.optimize import curve_fit
from scipy.stats import genextreme as gev

# Define functions
# Define a function to quantify the block minima for obs
def block_min_max_obs(
    obs_df: pd.DataFrame,
    month_ints: list[int],
    var_name: str,
    min_max: str = "min",
    time_name: str = "effective_dec_years"
) -> pd.DataFrame:
    """
    Quantifies the block minima for the observed data.

    Parameters
    ----------

    obs_df : pd.DataFrame
        The observed data.

    month_ints : list[int]
        The month integers.

    var_name : str
        The name of the variable to quantify in the dataframe.

    min_max : str
        The min or max value to quantify.
        Options: "min", "max"

    time_name : str
        The name of the effective december years column.

    Returns
    -------

    pd.DataFrame
        The block minima for the observed data.
    """

    # assert that the index is a datetime
    assert obs_df.index.dtype == "datetime64[ns]", "The index must be a datetime."

    # assert that the month_ints are integers
    assert all(
        isinstance(month_int, int) for month_int in month_ints
    ), "The month integers must be integers."

    # Check that the unique months are equal to the month_ints
    assert (
        obs_df.index.month.unique() == month_ints
    ).all(), "The unique months must be equal to the month integers."

    # assert that the df has the effective_dec_years column
    assert time_name in obs_df.columns, "The effective_dec_years column must be present."

    # assert that the var_name is in the columns
    assert var_name in obs_df.columns, "The variable name must be present in the columns."

    # Set up an empty dataframe
    block_df = pd.DataFrame()

    # Loop over the time name
    for time in obs_df[time_name].unique():
        # Get the data for the time
        time_df = obs_df[obs_df[time_name] == time]

        # Get the min or max
        if min_max == "min":
            worst_day = time_df[var_name].idxmin()
        elif min_max == "max":
            worst_day = time_df[var_name].idxmax()
        else:
            raise ValueError("The min_max must be min or max.")

        # Create a new dataframe
        df_new = pd.DataFrame(
            {
                "effective_dec_year": [time],
                "block_min_max_time": [worst_day],
                "data_value": [time_df.loc[worst_day, var_name]],
            }
        )

        # Append the new dataframe to the block_df
        block_df = pd.concat([block_df, df_new])

    # Return the block_df
    return block_df



# Set up if name is main
if __name__ == "__main__":
    pass