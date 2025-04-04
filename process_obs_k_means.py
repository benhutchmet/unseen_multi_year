#!/usr/bin/env python
"""
process_obs_k_means.py
=======================

This script processes the observations from the input file and performs k-means clustering on the data.

"""
# %%
# Local imports
import os
import sys
import glob
import time
import argparse
import warnings

# Third-party imports
import numpy as np
import dask.array as da
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import iris
import cftime

# Specific imports
from tqdm import tqdm
from matplotlib import gridspec
from datetime import datetime, timedelta

from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde, pearsonr
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from iris.util import equalise_attributes

# local imports
from gev_functions import pivot_detrend_obs

# Set up the function for calculating the pattern correlation
def pattern_correlation(field1, field2):
    return pearsonr(field1.flatten(), field2.flatten())[0]


# Set up a function for applying latitude weighting
def apply_latitude_weights(data, lats):
    cos_lat_weights = np.cos(np.deg2rad(lats))
    return data * cos_lat_weights[np.newaxis, :, np.newaxis]


# Define a function for normalising the data
def normalise_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Define a function for loading all of the model psl data
def load_model_data(
    var_name: str,
    region: str,
    season: str,
    years_list: list,
    temp_res: str = "day",  # temporal resolution
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/",
    model: str = "HadGEM3-GC31-MM",
    arr_shape: int = 3750,
) -> np.ndarray:
    """
    Loads the model data for the given variable and region.

    Parameters:
    -----------

    var_name: str
        The name of the variable to load.
    region: str
        The region to load the data for.
    season: str
        The season to load the data for.
    years_list: list
        The list of years to load the data for.
    temp_res: str
        The temporal resolution of the data.
    save_dir: str
        The directory to save the data to.
    model: str
        The model to load the data for.
    arr_shape: int
        The shape of the arrays to load.

    Returns:
    --------

    np.ndarray
        The model data for the given variable and region.
    
    """

    # First load a test array for the first year of years list
    test_arr_path = os.path.join(
        save_dir,
        f"{model}_{var_name}_{region}_{years_list[0]}_{season}_{temp_res}.npy",
    )

    # if the file does not exist then raise an error
    if not os.path.exists(test_arr_path):
        raise ValueError(f"File does not exist: {test_arr_path}")
    
    # Load the test array
    test_arr = np.load(test_arr_path)

    # Set up the shape of the full model array
    model_arr_full = np.zeros((len(years_list), test_arr.shape[1], test_arr.shape[2], test_arr.shape[3], test_arr.shape[4]))

    # Loop over the years list
    for i, year in tqdm(enumerate(years_list), desc="Looping over years"):
        # Form the file path
        model_arr_path = os.path.join(
            save_dir,
            f"{model}_{var_name}_{region}_{year}_{season}_{temp_res}.npy",
        )

        # if the file does not exist then raise an error
        if not os.path.exists(model_arr_path):
            raise ValueError(f"File does not exist: {model_arr_path}")
        
        # Load the data for this year
        # arr_this_year = np.load(model_arr_path)
        arr_this_year = np.load(model_arr_path)

        # if the shape of the 2th dimension is not equal to the test array then raise an error
        if arr_this_year.shape[2] != arr_shape:
            print(f"Shape of array for year {year} is {arr_this_year.shape}")
            arr_this_year = arr_this_year[:, :, 0:arr_shape, :, :]

        # add the data to the model array
        model_arr_full[i] = arr_this_year

    return model_arr_full

def assign_regimes(obs_data, cluster_centroids, threshold=1.0):
    """
    Assigns each day to the most representative regime based on projection scores.
    If the maximum projection score is below a threshold (default = 1 std), 
    the day is classified as a neutral regime (-1).
    
    Parameters:
    -----------
    obs_data : np.ndarray
        Daily MSLP anomaly fields with shape (n_days, lats, lons).
    cluster_centroids : np.ndarray
        Cluster centroids with shape (K, lats, lons).
    threshold : float, optional
        Threshold for projection standard deviation to assign a regime (default=1.0).
    
    Returns:
    --------
    np.ndarray
        Array of assigned regimes for each day (size = n_days).
    """

    # Reshape the data to (n_days, features)
    X_obs = obs_data.reshape(obs_data.shape[0], -1)  # (n_days, features)
    X_centroids = cluster_centroids.reshape(cluster_centroids.shape[0], -1)  # (K, features)

    # Compute projection of each day onto each cluster centroid
    projection_scores = X_obs @ X_centroids.T  # Shape (n_days, K)

    # Normalize projection scores
    mean_proj = np.mean(projection_scores, axis=0)
    std_proj = np.std(projection_scores, axis=0)
    projection_scores_normalized = (projection_scores - mean_proj) / std_proj  # Shape (n_days, K)

    # Find the highest projection index
    highest_regime = np.argmax(projection_scores_normalized, axis=1)  # (n_days,)

    # Assign regime only if projection is above the threshold
    assigned_labels = np.where(
        np.max(projection_scores_normalized, axis=1) >= threshold,  # Condition
        highest_regime,  # Assign regime
        -1  # Neutral regime
    )

    return assigned_labels


# Define a function for calculating the pattern correlation ratio
def calculate_pcr(
    data: np.ndarray,
    K_range: range,
    nboot: int = 10,  # for testing purposes
) -> tuple:
    """
    Calculates the pattern correlation ratio for the data.

    Parameters:
    -----------

    data: np.ndarray
        The data to calculate the pattern correlation ratio for.
    K_range: range
        The range of K values to use for the k-means clustering.
    nboot: int
        The number of bootstrap samples to use for the calculation.

    Returns:
    --------

    tuple containing:
        - The pattern correlation ratio for each nboot sample
        - The intra cluster pattern correlation for each nboot sample
        - The inter cluster pattern correlation for each nboot sample

    """

    # Set up the arrays to append to
    pcr_arr = np.zeros((len(K_range), nboot))
    intra_cluster_corr_arr = np.zeros((len(K_range), nboot))
    inter_cluster_corr_arr = np.zeros((len(K_range), nboot))

    # Loop over the K values
    for K_idx, K_this in enumerate(K_range):
        pcr_this = np.zeros(nboot)
        for iboot in tqdm(range(nboot), desc=f"K = {K_this}"):
            # Set up the k-means clustering
            kmeans = KMeans(
                n_clusters=K_this, random_state=None, n_init=10, max_iter=300
            )
            cluster_labels = kmeans.fit_predict(data)
            cluster_centroids = kmeans.cluster_centers_

            # Compute the cluster sizes
            Nk = np.array([np.sum(cluster_labels == k) for k in range(K_this)])

            # Compute the intra-cluster pattern correlation
            intra_corr_sum = 0
            intra_corr_count = 0
            for k in range(K_this):
                indices_k = np.where(cluster_labels == k)[0]
                for i in range(len(indices_k)):
                    for j in range(i + 1, len(indices_k)):
                        intra_corr_sum += pattern_correlation(
                            data[indices_k[i]], data[indices_k[j]]
                        )
                        intra_corr_count += 1
            intra_corr = intra_corr_sum / intra_corr_count

            # Compute the inter-cluster pattern correlation
            inter_corr_sum = 0
            inter_corr_count = 0
            for k in range(K_this):
                for l in range(k + 1, K_this):
                    indices_k = np.where(cluster_labels == k)[0]
                    indices_l = np.where(cluster_labels == l)[0]
                    for i in range(len(indices_k)):
                        for j in range(len(indices_l)):
                            inter_corr_sum += pattern_correlation(
                                data[indices_k[i]], data[indices_l[j]]
                            )
                            inter_corr_count += 1
            inter_corr = inter_corr_sum / inter_corr_count

            # Append the intra and inter cluster pattern correlations
            intra_cluster_corr_arr[K_idx, iboot] = intra_corr
            inter_cluster_corr_arr[K_idx, iboot] = inter_corr

            # Compute the pattern correlation ratio
            numerator = np.sum(Nk * (Nk - 1)) * inter_corr
            denominator = np.sum(Nk[:, np.newaxis] * Nk[np.newaxis, :]) * intra_corr
            pcr_this[iboot] = numerator / denominator

        # Append the pattern correlation ratio
        pcr_arr[K_idx, :] = pcr_this

    return (pcr_arr, intra_cluster_corr_arr, inter_cluster_corr_arr)


# Perform bootstrapped clustering with progressive centroid updating
def bootstrapped_clustering(data, K, nboot=100, max_reassign=10):
    """
    Performs bootstrapped clustering with progressive centroid updating.

    Parameters:
    -----------
    data: np.ndarray
        The dataset to be clustered.
    K: int
        The number of clusters.
    nboot: int
        The number of bootstrap iterations.
    max_reassign: int
        The number of times to reassign points to stabilize clusters.

    Returns:
    --------
    best_cluster_labels: np.ndarray
        The most representative cluster assignments.
    best_centroids: np.ndarray
        The most representative cluster centroids.
    """

    all_cluster_sets = []
    all_centroids = []

    for iboot in tqdm(range(nboot), desc="Bootstrapping"):
        # Bootstrap resampling
        bootstrap_indices = np.random.choice(data.shape[0], data.shape[0], replace=True)
        data_bootstrap = data[bootstrap_indices]

        # Initialize KMeans clustering
        kmeans = KMeans(n_clusters=K, random_state=None, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(data_bootstrap)
        cluster_centroids = kmeans.cluster_centers_

        # Iteratively refine clusters
        for _ in range(max_reassign):
            new_labels = np.zeros_like(cluster_labels)
            for i, sample in enumerate(data_bootstrap):
                distances = np.linalg.norm(cluster_centroids - sample, axis=1)
                new_labels[i] = np.argmin(distances)

            # Update centroids dynamically
            for k in range(K):
                cluster_centroids[k] = np.mean(data_bootstrap[new_labels == k], axis=0)

            cluster_labels = new_labels

        all_cluster_sets.append(cluster_labels)
        all_centroids.append(cluster_centroids)

    # Compute cluster set similarity across bootstraps
    best_index = select_most_representative_cluster(all_cluster_sets, all_centroids)

    return all_cluster_sets[best_index], all_centroids[best_index]


def select_most_representative_cluster(all_cluster_sets, all_centroids):
    """
    Identifies the most representative cluster set based on pattern correlation.

    Parameters:
    -----------
    all_cluster_sets: list of np.ndarray
        List of cluster assignments from all bootstrap iterations.
    all_centroids: list of np.ndarray
        List of cluster centroids from all bootstrap iterations.

    Returns:
    --------
    best_index: int
        Index of the most representative cluster set.
    """
    nboot = len(all_cluster_sets)
    correlation_matrix = np.zeros((nboot, nboot))

    for i in range(nboot):
        for j in range(nboot):
            if i != j:
                corr_sum = 0
                for k in range(len(all_centroids[i])):
                    corr_sum += pattern_correlation(
                        all_centroids[i][k], all_centroids[j][k]
                    )
                correlation_matrix[i, j] = corr_sum / len(all_centroids[i])

    # Find the cluster set with highest mean correlation
    mean_correlations = np.mean(correlation_matrix, axis=1)
    best_index = np.argmax(mean_correlations)

    return best_index

# Set up the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the df to load from
    obs_df_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_obs_tas_UK_1960-2017_DJF_2_April.csv"
    model_red_df_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_model_tas_lead_dt_bc_UK_1960-2017_DJF.csv"
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    saved_obs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
    saved_model_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"
    var_name = "psl"
    region = "NA"  # wider North Atlantic region
    season = "DJF"
    freq = "day"

    # Load the data
    obs_df = pd.read_csv(obs_df_path)

    # Remove the "Unnamed: 0" column
    obs_df.drop(columns=["Unnamed: 0"], inplace=True)

    # add a new column for the year
    obs_df["year"] = obs_df["time"].apply(
        lambda x: (
            x.split("-")[0]
            if x.split("-")[1] == "12"
            else str(int(x.split("-")[0]) - 1)
        )
    )

    # Set up the years
    years_list = list(range(1961, 2018))

    # print the unique years in the obs df
    print(obs_df["year"].unique())

    # asserrt that the years are correct
    assert np.all([str(year) in obs_df["year"].unique() for year in years_list])

    # make sure t

    # print the head of the data
    print(obs_df.head())

    # print the tail of the data
    print(obs_df.tail())

    # import the lats and lons for testing
    lats_path = os.path.join(
        metadata_dir,
        f"HadGEM3-GC31-MM_{var_name}_{region}_1961_{season}_{freq}_lats.npy",
    )
    lons_path = os.path.join(
        metadata_dir,
        f"HadGEM3-GC31-MM_{var_name}_{region}_1961_{season}_{freq}_lons.npy",
    )

    # if the file does not exist then raise an error
    if not os.path.exists(lats_path):
        raise ValueError(f"File does not exist: {lats_path}")

    # Load the lats
    lats = np.load(lats_path)

    # if the file does not exist then raise an error
    if not os.path.exists(lons_path):
        raise ValueError(f"File does not exist: {lons_path}")

    # Load the lons
    lons = np.load(lons_path)

    # Set up an array to store the data
    obs_data_arr = np.zeros((len(obs_df), len(lats), len(lons)))

    # loop over the years
    for i, year in tqdm(
        enumerate(obs_df["effective_dec_year"].unique()), desc="Looping over years"
    ):
        # get the data for the year
        year_df = obs_df[obs_df["effective_dec_year"] == year]

        # Extract the time from this row
        time_this = year_df["time"].values[0]

        # parse the date string
        year_this, month_this, day_this = map(int, time_this.split("-"))

        # create a cftime datetime object
        time_this = cftime.DatetimeGregorian(
            year_this, month_this, day_this, 11, 0, 0, 0
        )

        # Set up the path to the times from the obs data
        obs_times_path = os.path.join(
            metadata_dir, f"ERA5_{var_name}_{region}_{year}_{season}_{freq}_times.npy"
        )

        # if the file does not exist then raise an error
        if not os.path.exists(obs_times_path):
            raise ValueError(f"File does not exist: {obs_times_path}")

        # Load the times
        obs_times = np.load(obs_times_path)

        # converted to datetime objects
        obs_times = cftime.num2date(
            obs_times, units="hours since 1900-01-01", calendar="gregorian"
        )

        # find the index of time this in obs times
        time_this_index = np.where(obs_times == time_this)[0][0]

        # Form the path to the obs data
        obs_data_path = os.path.join(
            saved_obs_dir, f"ERA5_{var_name}_{region}_{year}_{season}_{freq}.npy"
        )

        # if the file does not exist then raise an error
        if not os.path.exists(obs_data_path):
            raise ValueError(f"File does not exist: {obs_data_path}")

        # Load the data
        obs_data = np.load(obs_data_path)

        # extract the data for the time this
        obs_data_this = obs_data[time_this_index]

        # add the data to the obs data array
        obs_data_arr[i] = obs_data_this

    # print the first and last 2D array in obs_data_arr
    print(obs_data_arr[0])
    print(obs_data_arr[-1])

    # print the shape of obs_data_arr
    print(obs_data_arr.shape)

    # Apply latitude weighting
    obs_data_arr = apply_latitude_weights(obs_data_arr, lats)

    # reshape the data into two dimensions
    obs_data_arr = obs_data_arr.reshape(
        (obs_data_arr.shape[0], obs_data_arr.shape[1] * obs_data_arr.shape[2])
    )

    # set up the scaler
    scaler = StandardScaler()

    # fit the scaler
    scaler.fit(obs_data_arr)

    # transform the data
    obs_data_arr = scaler.fit_transform(obs_data_arr)

    # # nomralise the data
    # obs_data_arr = normalise_data(obs_data_arr)

    # # Calculate the pattern correlation ratio
    # pcr, intra_cluster_corr, inter_cluster_corr = calculate_pcr(
    #     data=obs_data_arr,
    #     K_range=range(2, 15),
    #     nboot=10,
    # )

    # # print the shape of pcr
    # print(pcr.shape)

    # # print the values of pcr
    # print(pcr)

    # # print the intra cluster correlation
    # print("correlations between members of the same cluster")
    # print(intra_cluster_corr)

    # # print the inter cluster correlation
    # print("correlations between members of different clusters")
    # print(inter_cluster_corr)

    # # Set up a figure with one column and 3 rows
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # # set up the axes
    # ax[0].plot(range(2, 15), pcr.mean(axis=1), marker="o")

    # # set the title
    # ax[0].set_title("Pattern Correlation Ratio")

    # # set the x label
    # ax[0].set_xlabel("Number of clusters")

    # # set the y label
    # ax[0].set_ylabel("PCR")

    # # set the xticks
    # ax[0].set_xticks(range(2, 15))

    # # set the xticks
    # ax[0].set_xticklabels(range(2, 15))

    # # set up the axes for the next plot
    # ax[1].plot(range(2, 15), intra_cluster_corr.mean(axis=1), marker="o")

    # # set the title
    # ax[1].set_title("Intra Cluster Pattern Correlation")

    # # set the x label
    # ax[1].set_xlabel("Number of clusters")

    # # set the y label
    # ax[1].set_ylabel("Intra Cluster Correlation")

    # # set the xticks
    # ax[1].set_xticks(range(2, 15))

    # # set the xticks
    # ax[1].set_xticklabels(range(2, 15))

    # # set up the axes for the next plot
    # ax[2].plot(range(2, 15), inter_cluster_corr.mean(axis=1), marker="o")

    # # set the title
    # ax[2].set_title("Inter Cluster Pattern Correlation")

    # # set the x label
    # ax[2].set_xlabel("Number of clusters")

    # # set the y label
    # ax[2].set_ylabel("Inter Cluster Correlation")

    # # set the xticks
    # ax[2].set_xticks(range(2, 15))

    # # set the xticks
    # ax[2].set_xticklabels(range(2, 15))

    # # show the plot
    # plt.show()

    # Set up our optimal K
    optimal_K = 5

    # Perform bootstrapped clustering
    cluster_labels, cluster_centroids = bootstrapped_clustering(
        data=obs_data_arr, K=optimal_K, nboot=100, max_reassign=10
    )

    # print the cluster labels
    print(cluster_labels)

    # print the cluster centroids
    print(cluster_centroids)

    # print the shape of the cluster centroids
    print(cluster_centroids.shape)

    # print the shape of the cluster labels
    print(cluster_labels.shape)

    # # de-normalise the cluster centroids
    # cluster_centroids = scaler.inverse_transform(cluster_centroids)

    # reshape the cluster centroids
    cluster_centroids = cluster_centroids.reshape(
        (cluster_centroids.shape[0], len(lats), len(lons))
    )

    # Apply the pivot detrend to the obs df
    obs_df_dt = pivot_detrend_obs(
        df=obs_df,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c_min"
    )

    # add a new column to the dataframe for cluster labels
    obs_df_dt["cluster_labels"] = cluster_labels

    # print the head of the dataframe
    print(obs_df_dt.head())

    # # Set up a figure to plot the scatter
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [4, 1]}, layout="compressed")

    # # set up the colours
    # colours = ["r", "b", "g", "y", "m"]

    # # loop over the cluster labels
    # for i in range(optimal_K):
    #     # get the data for this cluster
    #     cluster_data = obs_df_dt[obs_df_dt["cluster_labels"] == i]

    #     # plot the scatter
    #     axs[0].scatter(cluster_data["effective_dec_year"], cluster_data["data_c_min_dt"], label=f"Cluster {i+1}", color=colours[i])

    # # add the legend
    # axs[0].legend()

    # # set the x label
    # axs[0].set_xlabel("December year")

    # # set the y label
    # axs[0].set_ylabel("Temperature (C)")

    # # set the title
    # axs[0].set_title("ERA5 DJF block min T days by cluster, 1961-2017")

    # # loop over the cluster labels
    # for i in range(optimal_K):
    #     # get the data for this cluster
    #     cluster_data = obs_df_dt[obs_df_dt["cluster_labels"] == i]

    #     # plot the boxplot
    #     axs[1].boxplot(
    #         cluster_data["data_c_min_dt"],
    #         positions=[i + 1],
    #         patch_artist=True,
    #         boxprops=dict(facecolor="white", color=colours[i]),
    #         whiskerprops=dict(color=colours[i]),
    #         capprops=dict(color=colours[i]),
    #         flierprops=dict(markerfacecolor=colours[i], markeredgecolor=colours[i]),
    #         medianprops=dict(color=colours[i]),
    #         vert=True,
    #         widths=0.5,
    #     )

    # # set the xticks
    # axs[1].set_xticks([i + 1 for i in range(optimal_K)])

    # # show the plot
    # plt.show()

    # apply the function to assign regimes
    assigned_labels = assign_regimes(
        obs_data=obs_data_arr,
        cluster_centroids=cluster_centroids,
        threshold=1.0
    )

    # print the assigned labels
    print(assigned_labels)

    # print the shape of the assigned labels
    print(assigned_labels.shape)

    # add the assigned labels to the dataframe
    obs_df_dt["assigned_labels"] = assigned_labels

    # shift the assigned labels by 1
    obs_df_dt["assigned_labels"] = obs_df_dt["assigned_labels"] + 1

    # print the head of obs_df_dt
    print(obs_df_dt.head())

    # ptoiny the tail of obs_df_dt
    print(obs_df_dt.tail())

    # redo the same figure, but using the updated labels
    # Set up a figure to plot the scatter
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [4, 1]}, layout="compressed")

    # set up the colours
    colours = ["grey", "r", "b", "g", "y", "m"]

    # loop over the cluster labels
    for i in range(optimal_K + 1):
        # get the data for this cluster
        cluster_data = obs_df_dt[obs_df_dt["assigned_labels"] == i]

        # if i == 0:
        if i == 0:
            # plot the scatter
            axs[0].scatter(cluster_data["effective_dec_year"], cluster_data["data_c_min_dt"], label=f"Neutral", color=colours[i])
        else:
            # plot the scatter
            axs[0].scatter(cluster_data["effective_dec_year"], cluster_data["data_c_min_dt"], label=f"Cluster {i}", color=colours[i])

    # add the legend
    axs[0].legend()

    # set the x label
    axs[0].set_xlabel("December year")

    # set the y label
    axs[0].set_ylabel("Temperature (C)")

    # set the title
    axs[0].set_title("ERA5 DJF block min T days by cluster, 1961-2017, re-assigned")

    # set up the plots directory
    plots_dir = "/home/users/benhutch/unseen_multi_year/plots/"

    # Set up the current date time
    now = datetime.now()

    # Set up a fname for this plot
    fname = os.path.join(plots_dir, f"ERA5_{var_name}_{region}_{optimal_K}_clusters_{now}.png")

    # create the path to the file
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # create the full path to the files
    full_path = os.path.join(plots_dir, fname)

    # loop over the cluster labels
    for i in range(optimal_K + 1):
        # get the data for this cluster
        cluster_data = obs_df_dt[obs_df_dt["assigned_labels"] == i]

        # plot the boxplot
        axs[1].boxplot(
            cluster_data["data_c_min_dt"],
            positions=[i],
            patch_artist=True,
            boxprops=dict(facecolor="white", color=colours[i]),
            whiskerprops=dict(color=colours[i]),
            capprops=dict(color=colours[i]),
            flierprops=dict(markerfacecolor=colours[i], markeredgecolor=colours[i]),
            medianprops=dict(color=colours[i]),
            vert=True,
            widths=0.5,
        )

    # set the xticks
    axs[1].set_xticks([i for i in range(optimal_K + 1)])

    # save the plot
    plt.savefig(full_path, dpi=300)

    # show the plot
    plt.show()

        # Set up a figure with 2 cols and 2 rows
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}, layout="compressed")

    # loop over the cluster centroids
    for i, ax in enumerate(axs.flatten()):
        # if i is greater than the number of clusters then break
        if i >= optimal_K:
            break
        
        # set up the plot
        im = ax.contourf(lons, lats, cluster_centroids[i], cmap="coolwarm", extend="both")

        # add coastlines
        ax.coastlines()

        # calculate the % of days in each cluster
        cluster_size = np.sum(obs_df_dt["assigned_labels"] == i) / len(obs_df_dt) * 100

        # add the title
        ax.set_title(f"Cluster {i+1} ({cluster_size:.1f}%)")

    # add a colorbar
    fig.colorbar(im, ax=axs, orientation="horizontal", label="MSLP", pad=0.1, shrink=0.8)

    # Set up a super title
    fig.suptitle("K-means clustering of MSLP (bootstrapped)")

    # Set up a fname for this figure
    fname = os.path.join(plots_dir, f"ERA5_{var_name}_{region}_{optimal_K}_clusters_map_{now}.png")

    # create the path to the file
    full_path_clusters = os.path.join(plots_dir, fname)

    # save the plot
    plt.savefig(full_path_clusters, dpi=300)

    # show the plot
    plt.show()

    # load the red dots array
    red_dots_df = pd.read_csv(model_red_df_path)

    # print the type of effective dec year in the obs df
    print(type(obs_df_dt["effective_dec_year"].values[0]))

    # print the type of effective dec year in the red dots df
    print(type(red_dots_df["effective_dec_year"].values[0]))

    # convert the effective dec year to a string
    obs_df_dt["effective_dec_year"] = obs_df_dt["effective_dec_year"].astype(str)

    # extract the year from the effective dec year
    obs_df_dt["effective_dec_year"] = obs_df_dt["effective_dec_year"].apply(lambda x: x.split("-")[0])

    # Set up the array to append the data to
    red_dots_arr = np.zeros((len(red_dots_df), len(lats), len(lons)))

    # set up the model years
    model_years = np.array([str(year) for year in range(1960, 2018)])

    # set up the members list
    members_list = np.array([10, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # use the function to load the model psl arr
    model_psl_arr = load_model_data(
        var_name=var_name,
        region=region,
        season=season,
        years_list=model_years,
        temp_res=freq,
        save_dir="/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/",
        model="HadGEM3-GC31-MM",
        arr_shape=3750,
    )

    # set up a fname for the model psl array
    model_psl_arr_fname = os.path.join(
        saved_model_dir,
        f"HadGEM3-GC31-MM_{var_name}_{region}_{season}_day_{model_years[0]}-{model_years[-1]}.npy",
    )

    # if the file does not exist then save it
    if not os.path.exists(model_psl_arr_fname):
        # save the model psl array
        np.save(model_psl_arr_fname, model_psl_arr)

    # Set up the tuples list to append to
    tuples_list = []

    # format model years as ints
    model_years = np.array([int(year) for year in model_years])

    # loop over the rows in the red dots df
    # Loop over the rows in the red dots df
    for i, row in tqdm(red_dots_df.iterrows(), desc="Looping over red dots df"):
        # Extract the init year and winter year
        init_year = row["init_year"]
        member = row["member"]
        lead = row["lead"]

        # Include the effective dec year
        effective_dec_year = row["effective_dec_year"]

        # Strip the first 4 characters and format as an int
        effective_dec_year_int = int(effective_dec_year[:4])

        # print the init year and member and lead
        print(init_year, member, lead, effective_dec_year_int)

        # print the type of effective dec year
        print(type(effective_dec_year))

        # make sure init year is an int
        init_year = int(init_year)

        # Find the index of the year in model_years
        init_year_idx = np.where(model_years == init_year)[0][0]

        # Find the index of the member in members list
        member_idx = np.where(members_list == member)[0][0]

        # Get the lead idx
        lead_idx = int(lead - 1)

        # Save the indices as a tuple
        tuples_list.append((init_year_idx, member_idx, lead_idx, effective_dec_year_int))

        # Extract the data
        red_dots_arr[i, :, :] = model_psl_arr[init_year_idx, member_idx, lead_idx, :, :]

    # assign the red dots arr to the patterns defined in the obs
    red_dots_arr_labels = assign_regimes(
        obs_data=red_dots_arr,
        cluster_centroids=cluster_centroids,
        threshold=1.0
    )

    # add these labels to the red dots df
    red_dots_df["assigned_labels"] = red_dots_arr_labels

    # Set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, gridspec_kw={"width_ratios": [4, 1]}, layout="compressed")

    # set up the colours
    colours = ["grey", "r", "b", "g", "y", "m"]

    # loop over the cluster labels
    for i in range(optimal_K + 1):
        # get the data for this cluster
        cluster_data = red_dots_df[red_dots_df["assigned_labels"] == i]

        # if i == 0:
        if i == 0:
            # plot the scatter
            axs[0].scatter(cluster_data["effective_dec_year"], cluster_data["data_c_min"], label=f"Neutral", color=colours[i])
        else:
            # plot the scatter
            axs[0].scatter(cluster_data["effective_dec_year"], cluster_data["data_c_min"], label=f"Cluster {i}", color=colours[i])

    # add the legend
    axs[0].legend()

    # set the x label
    axs[0].set_xlabel("December year")

    # set the y label
    axs[0].set_ylabel("Temperature (C)")

    # set the title
    axs[0].set_title("HadGEM3-GC31-MM DJF extreme block min T days by cluster, 1960-2017, re-assigned")

    # loop over the cluster labels
    for i in range(optimal_K + 1):
        # get the data for this cluster
        cluster_data = red_dots_df[red_dots_df["assigned_labels"] == i]

        # plot the boxplot
        axs[1].boxplot(
            cluster_data["data_tas_c_min_dt_bc"],
            positions=[i],
            patch_artist=True,
            boxprops=dict(facecolor="white", color=colours[i]),
            whiskerprops=dict(color=colours[i]),
            capprops=dict(color=colours[i]),
            flierprops=dict(markerfacecolor=colours[i], markeredgecolor=colours[i]),
            medianprops=dict(color=colours[i]),
            vert=True,
            widths=0.5,
        )

    # set the xticks
    axs[1].set_xticks([i for i in range(optimal_K + 1)])

    # set up the filename
    fname = os.path.join(plots_dir, f"HadGEM3-GC31-MM_{var_name}_{region}_{optimal_K}_clusters_{now}.png")

    # create the path to the file
    full_path = os.path.join(plots_dir, fname)

    # save the plot
    plt.savefig(full_path, dpi=300)

    # show the plot
    plt.show()
    
    # # fit just a single kmeans to the data
    # kmeans = KMeans(n_clusters=optimal_K, random_state=None, n_init=10, max_iter=300)
    # cluster_labels = kmeans.fit_predict(obs_data_arr)
    # cluster_centroids = kmeans.cluster_centers_

    # # reshape the cluster centroids
    # cluster_centroids = cluster_centroids.reshape(
    #     (cluster_centroids.shape[0], len(lats), len(lons))
    # )

    # # Set up a figure with 2 cols and 2 rows
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}, layout="compressed")

    # # loop over the cluster centroids
    # for i, ax in enumerate(axs.flatten()):
    #     # set up the plot
    #     im = ax.contourf(lons, lats, cluster_centroids[i], cmap="coolwarm", extend="both")

    #     # add coastlines
    #     ax.coastlines()

    #     # calculate the % of days in each cluster
    #     cluster_size = np.sum(cluster_labels == i) / len(cluster_labels) * 100

    #     # add the title
    #     ax.set_title(f"Cluster {i+1} ({cluster_size:.1f}%)")

    # # add a colorbar
    # fig.colorbar(im, ax=axs, orientation="horizontal", label="MSLP", pad=0.1, shrink=0.8)

    # # Set up a super title
    # fig.suptitle("K-means clustering of MSLP (single fit)")

    # print the time taken
    print("Time taken: ", time.time() - start_time)
    print("Finished!")

    return None


if __name__ == "__main__":
    main()
# %%
