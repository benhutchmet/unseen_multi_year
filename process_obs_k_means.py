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
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    saved_obs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
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
    optimal_K = 4

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

    # de-normalise the cluster centroids
    cluster_centroids = scaler.inverse_transform(cluster_centroids)

    # reshape the cluster centroids
    cluster_centroids = cluster_centroids.reshape(
        (cluster_centroids.shape[0], len(lats), len(lons))
    )

    # Set up a figure with 2 cols and 2 rows
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    # loop over the cluster centroids
    for i, ax in enumerate(axs.flatten()):
        # set up the plot
        im = ax.contourf(lons, lats, cluster_centroids[i], cmap="coolwarm", extend="both")

        # add coastlines
        ax.coastlines()

        # add the title
        ax.set_title(f"Cluster {i+1}")

    # add a colorbar
    fig.colorbar(im, ax=axs, orientation="horizontal", label="MSLP", pad=0.1, shrink=0.8)

    # print the time taken
    print("Time taken: ", time.time() - start_time)
    print("Finished!")

    return None


if __name__ == "__main__":
    main()
# %%
