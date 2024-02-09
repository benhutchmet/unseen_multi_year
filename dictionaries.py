# Setting up the dictoinaries for multi-year UNSEEN testing

# define the uk grid from Clarke et al. 2017
uk_grid = {
    'lon1': -10,
    'lon2': 3,
    'lat1': 50,
    'lat2': 60
}

# define the north sea grid from kay et al. 2023
north_sea_kay = {
    'lon1': 1, # degrees east
    'lon2': 7,
    'lat1': 53, # degrees north
    'lat2': 59
}

# Set up the path to the obs
obs_path = "/home/users/benhutch/ERA5/surface_wind_ERA5.nc"

regrid_hadgem_obs_path = "/home/users/benhutch/ERA5/surface_wind_ERA5_regrid_HadGEM.nc"

# Set up the gws base dir
gws_base = "/gws/nopw/j04/canari/users/benhutch"

# Create a nested dictionary of the data_paths for each model and variable

