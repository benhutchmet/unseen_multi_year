# Setting up the dictoinaries for multi-year UNSEEN testing

# define the uk grid from Clarke et al. 2017
uk_grid = {"lon1": -10, "lon2": 3, "lat1": 50, "lat2": 60}

# Define the UK wind box
wind_gridbox = {"lat1": 50, "lat2": 59.5, "lon1": -6, "lon2": 2}

# Define the UK wind box
wind_gridbox_south = {"lat1": 50, "lat2": 54, "lon1": -6, "lon2": 2}

# define the north sea grid from kay et al. 2023
north_sea_kay = {
    "lon1": 1,  # degrees east
    "lon2": 7,
    "lat1": 53,  # degrees north
    "lat2": 59,
}

uk_n_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 57, "lat2": 70}

uk_s_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 38, "lat2": 51}

uk_n_box_tight = {"lon1": -10, "lon2": 5, "lat1": 57, "lat2": 70}

uk_s_box_tight = {"lon1": -10, "lon2": 5, "lat1": 38, "lat2": 51}

# wind gridbox subset
wind_gridbox_subset = {
    "lon1": -6,
    "lon2": 2,
    "lat1": 51, # top of south delta P gridbox
    "lat2": 57, # bottom of north delta P gridbox
}

# Set up the path to the obs
obs_path = "/home/users/benhutch/ERA5/surface_wind_ERA5.nc"

regrid_hadgem_obs_path = "/home/users/benhutch/ERA5/surface_wind_ERA5_regrid_HadGEM.nc"

# Set up the gws base dir
gws_base = "/gws/nopw/j04/canari/users/benhutch"

# Define the dimensions for the gridbox for the azores
azores_grid_corrected = {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40}

# Define the dimensions for the gridbox for iceland
iceland_grid = {"lon1": 155, "lon2": 164, "lat1": 63, "lat2": 70}

# Define the dimensions for the gridbox for the azores
iceland_grid_corrected = {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70}
