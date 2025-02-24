"""
download_ERA5_jasmin.py
========================

This script sends a request to the CDS API to download the ERA5 data for a specific domain in monthly chunks at the hourly resolution.

Usage:
======

    $ python download_ERA5_jasmin.py --year <year> --month <month>

Arguments:
==========

    --year : int : year of the data to download
    --month : int : month of the data to download
    
Returns:
========

    netcdf files : .nc : netcdf files saved to the /gws/nopw/j04/canari/users/benhutch/ERA5 folder.

Author:
=======

    Ben W. Hutchins, University of Reading, 2025
"""

# Local imports
import os
import cdsapi
import argparse

# Define a function to check the files which exist
def check_files_exist():
    """
    Check the files which already exist in the target directory.
    """

    # Set up the target directory
    target_dir = '/gws/nopw/j04/canari/users/benhutch/ERA5/year_month/'

    # Set up the years to check for
    years = list(range(1940,2024))

    # Set up the months to check for
    months = list(range(1,13))

    # Set up a list for the missing files
    missing_files = []

    # Loop over the years and months
    for year in years:
        for month in months:
            # Set up the file name
            file_name = f'ERA5_EU_85000_zg_T_U_V{year}_{str(month).zfill(2)}.nc'

            # Check if the file exists
            if os.path.exists(target_dir + file_name):
                print(f'File exists for year: {year} and month: {month}')
            else:
                print(f'File does not exist for year: {year} and month: {month}')
                missing_files.append(file_name)

    return missing_files

# define the function to download the ERA5 data
def download_ERA5_to_jasmin(
    year: int,
    month: int,
) -> None:
    """
    Download ERA5 data for a given year and month to be used in jasmin.
    
    Parameters
    ----------
    
    year: int
        The year to download data for.
    
    month: int
        The month to download data for.

    Returns
    -------

    None

    """

    m = str(month).zfill(2) # make sure it is 01, 02 etc

    if m in ['04','06','09','11']:
        days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
    elif m in ['01','03','05','07','08','10','12']:
        days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    else:
        if year in [1940,1944,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020,2024]:
            days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']
        else:
            days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']

    #print(str(YEAR))
    #print(str(MONTH))
    y = str(year)

    target = '/gws/nopw/j04/canari/users/benhutch/ERA5/year_month/ERA5_EU_85000_zg_T_U_V' + str(y) + '_' + str(m) + '.nc'

    # if the file already exists, skip
    if os.path.exists(target):
        print(f'File already exists: {target}')
        return None

    client = cdsapi.Client()

    dataset = 'reanalysis-era5-pressure-levels' # On pressure levels
    request = {
        'product_type': ['reanalysis'],
        'variable': ['geopotential','temperature','u_component_of_wind','v_component_of_wind'],
        'year': [y],
        'month': [m],
        'day': days,
        'area': [72,-40,34,35], # N/W/S/E
        'time': ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
        'pressure_level': ['850'],
        'data_format': 'netcdf',
    }
    
    client.retrieve(dataset, request, target)
    
    return None

# define the main function
def main():
    
    # set up the argument parser
    parser = argparse.ArgumentParser(description='Download ERA5 data for RACC')
    year = parser.add_argument('--year', type=int, help='Year to download data for')
    month = parser.add_argument('--month', type=int, help='Month to download data for')

    # parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print('=====================================')
    print('Downloading ERA5 data for the following arguments:')
    print(f'Year: {args.year}')
    print(f'Month: {args.month}')
    print('=====================================')

    # missing_files = check_files_exist()

    # # Print the missing files
    # print('=====================================')
    # print('Missing files:')
    # print(missing_files)
    # print('=====================================')

    # return None

    # Run the function
    download_ERA5_to_jasmin(
        year=args.year,
        month=args.month,
    )

    return None

# run the main function
if __name__ == '__main__':
    main()