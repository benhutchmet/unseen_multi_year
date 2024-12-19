def project(cube, target_proj, nx=None, ny=None):
    """
    Nearest neighbour regrid to a specified target projection.

    Return a new cube that is the result of projecting a cube with 1 or 2
    dimensional latitude-longitude coordinates from its coordinate system into
    a specified projection e.g. Robinson or Polar Stereographic.
    This function is intended to be used in cases where the cube's coordinates
    prevent one from directly visualising the data, e.g. when the longitude
    and latitude are two dimensional and do not make up a regular grid.

    Args:
        * cube
            An instance of :class:`iris.cube.Cube`.
        * target_proj
            An instance of the Cartopy Projection class, or an instance of
            :class:`iris.coord_systems.CoordSystem` from which a projection
            will be obtained.
    Kwargs:
        * nx
            Desired number of sample points in the x direction for a domain
            covering the globe.
        * ny
            Desired number of sample points in the y direction for a domain
            covering the globe.

    Returns:
        An instance of :class:`iris.cube.Cube` and a list describing the
        extent of the projection.

    .. note::

        This function assumes global data and will if necessary extrapolate
        beyond the geographical extent of the source cube using a nearest
        neighbour approach. nx and ny then include those points which are
        outside of the target projection.

    .. note::

        Masked arrays are handled by passing their masked status to the
        resulting nearest neighbour values.  If masked, the value in the
        resulting cube is set to 0.

    .. warning::

        This function uses a nearest neighbour approach rather than any form
        of linear/non-linear interpolation to determine the data value of each
        cell in the resulting cube. Consequently it may have an adverse effect
        on the statistics of the data e.g. the mean and standard deviation
        will not be preserved.

    .. warning::

        If the target projection is non-rectangular, e.g. Robinson, the target
        grid may include points outside the boundary of the projection. The
        latitude/longitude of such points may be unpredictable.

    """
    try:
        lon_coord, lat_coord = _get_lon_lat_coords(cube)
    except IndexError:
        raise ValueError(
            "Cannot get latitude/longitude "
            "coordinates from cube {!r}.".format(cube.name())
        )

    if lat_coord.coord_system != lon_coord.coord_system:
        raise ValueError(
            "latitude and longitude coords appear to have "
            "different coordinates systems."
        )

    if lon_coord.units != "degrees":
        lon_coord = lon_coord.copy()
        lon_coord.convert_units("degrees")
    if lat_coord.units != "degrees":
        lat_coord = lat_coord.copy()
        lat_coord.convert_units("degrees")

    # Determine source coordinate system
    if lat_coord.coord_system is None:
        # Assume WGS84 latlon if unspecified
        warnings.warn(
            "Coordinate system of latitude and longitude "
            "coordinates is not specified. Assuming WGS84 Geodetic."
        )
        orig_cs = iris.coord_systems.GeogCS(
            semi_major_axis=6378137.0, inverse_flattening=298.257223563
        )
    else:
        orig_cs = lat_coord.coord_system

    # Convert to cartopy crs
    source_cs = orig_cs.as_cartopy_crs()

    # Obtain coordinate arrays (ignoring bounds) and convert to 2d
    # if not already.
    source_x = lon_coord.points
    source_y = lat_coord.points
    if source_x.ndim != 2 or source_y.ndim != 2:
        source_x, source_y = _meshgrid(source_x, source_y)

    # Calculate target grid
    target_cs = None
    if isinstance(target_proj, iris.coord_systems.CoordSystem):
        target_cs = target_proj
        target_proj = target_proj.as_cartopy_projection()

    # Resolution of new grid
    if nx is None:
        nx = source_x.shape[1]
    if ny is None:
        ny = source_x.shape[0]

    target_x, target_y, extent = cartopy.img_transform.mesh_projection(
        target_proj, nx, ny
    )

    # Determine dimension mappings - expect either 1d or 2d
    if lat_coord.ndim != lon_coord.ndim:
        raise ValueError(
            "The latitude and longitude coordinates have "
            "different dimensionality."
        )

    latlon_ndim = lat_coord.ndim
    lon_dims = cube.coord_dims(lon_coord)
    lat_dims = cube.coord_dims(lat_coord)

    if latlon_ndim == 1:
        xdim = lon_dims[0]
        ydim = lat_dims[0]
    elif latlon_ndim == 2:
        if lon_dims != lat_dims:
            raise ValueError(
                "The 2d latitude and longitude coordinates "
                "correspond to different dimensions."
            )
        # If coords are 2d assume that grid is ordered such that x corresponds
        # to the last dimension (shortest stride).
        xdim = lon_dims[1]
        ydim = lon_dims[0]
    else:
        raise ValueError(
            "Expected the latitude and longitude coordinates "
            "to have 1 or 2 dimensions, got {} and "
            "{}.".format(lat_coord.ndim, lon_coord.ndim)
        )

    # Create array to store regridded data
    new_shape = list(cube.shape)
    new_shape[xdim] = nx
    new_shape[ydim] = ny
    new_data = ma.zeros(new_shape, cube.data.dtype)

    # Create iterators to step through cube data in lat long slices
    new_shape[xdim] = 1
    new_shape[ydim] = 1
    index_it = np.ndindex(*new_shape)
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        slice_it = cube.slices([lat_coord, lon_coord])
    elif lat_coord.ndim == 2 and lon_coord.ndim == 2:
        slice_it = cube.slices(lat_coord)
    else:
        raise ValueError(
            "Expected the latitude and longitude coordinates "
            "to have 1 or 2 dimensions, got {} and "
            "{}.".format(lat_coord.ndim, lon_coord.ndim)
        )

    #    # Mask out points outside of extent in source_cs - disabled until
    #    # a way to specify global/limited extent is agreed upon and code
    #    # is generalised to handle -180 to +180, 0 to 360 and >360 longitudes.
    #    source_desired_xy = source_cs.transform_points(target_proj,
    #                                                   target_x.flatten(),
    #                                                   target_y.flatten())
    #    if np.any(source_x < 0.0) and np.any(source_x > 180.0):
    #        raise ValueError('Unable to handle range of longitude.')
    #    # This does not work in all cases e.g. lon > 360
    #    if np.any(source_x > 180.0):
    #        source_desired_x = (source_desired_xy[:, 0].reshape(ny, nx) +
    #                            360.0) % 360.0
    #    else:
    #        source_desired_x = source_desired_xy[:, 0].reshape(ny, nx)
    #    source_desired_y = source_desired_xy[:, 1].reshape(ny, nx)
    #    outof_extent_points = ((source_desired_x < source_x.min()) |
    #                           (source_desired_x > source_x.max()) |
    #                           (source_desired_y < source_y.min()) |
    #                           (source_desired_y > source_y.max()))
    #    # Make array a mask by default (rather than a single bool) to allow mask
    #    # to be assigned to slices.
    #    new_data.mask = np.zeros(new_shape)

    # Step through cube data, regrid onto desired projection and insert results
    # in new_data array
    for index, ll_slice in zip(index_it, slice_it):
        # Regrid source data onto target grid
        index = list(index)
        index[xdim] = slice(None, None)
        index[ydim] = slice(None, None)
        index = tuple(index)  # Numpy>=1.16 : index with tuple, *not* list.
        new_data[index] = cartopy.img_transform.regrid(
            ll_slice.data,
            source_x,
            source_y,
            source_cs,
            target_proj,
            target_x,
            target_y,
        )

    #    # Mask out points beyond extent
    #    new_data[index].mask[outof_extent_points] = True

    # Remove mask if it is unnecessary
    if not np.any(new_data.mask):
        new_data = new_data.data

    # Create new cube
    new_cube = iris.cube.Cube(new_data)

    # Add new grid coords
    x_coord = iris.coords.DimCoord(
        target_x[0, :],
        "projection_x_coordinate",
        units="m",
        coord_system=copy.copy(target_cs),
    )
    y_coord = iris.coords.DimCoord(
        target_y[:, 0],
        "projection_y_coordinate",
        units="m",
        coord_system=copy.copy(target_cs),
    )

    new_cube.add_dim_coord(x_coord, xdim)
    new_cube.add_dim_coord(y_coord, ydim)

    # Add resampled lat/lon in original coord system
    source_desired_xy = source_cs.transform_points(
        target_proj, target_x.flatten(), target_y.flatten()
    )
    new_lon_points = source_desired_xy[:, 0].reshape(ny, nx)
    new_lat_points = source_desired_xy[:, 1].reshape(ny, nx)
    new_lon_coord = iris.coords.AuxCoord(
        new_lon_points,
        standard_name="longitude",
        units="degrees",
        coord_system=orig_cs,
    )
    new_lat_coord = iris.coords.AuxCoord(
        new_lat_points,
        standard_name="latitude",
        units="degrees",
        coord_system=orig_cs,
    )
    new_cube.add_aux_coord(new_lon_coord, [ydim, xdim])
    new_cube.add_aux_coord(new_lat_coord, [ydim, xdim])

    coords_to_ignore = set()
    coords_to_ignore.update(cube.coords(contains_dimension=xdim))
    coords_to_ignore.update(cube.coords(contains_dimension=ydim))
    for coord in cube.dim_coords:
        if coord not in coords_to_ignore:
            new_cube.add_dim_coord(coord.copy(), cube.coord_dims(coord))
    for coord in cube.aux_coords:
        if coord not in coords_to_ignore:
            new_cube.add_aux_coord(coord.copy(), cube.coord_dims(coord))
    discarded_coords = coords_to_ignore.difference([lat_coord, lon_coord])
    if discarded_coords:
        warnings.warn(
            "Discarding coordinates that share dimensions with "
            "{} and {}: {}".format(
                lat_coord.name(),
                lon_coord.name(),
                [coord.name() for coord in discarded_coords],
            )
        )

    # TODO handle derived coords/aux_factories

    # Copy metadata across
    new_cube.metadata = cube.metadata

    return new_cube, extent