import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cftime
import pytest

fpath = (
    "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest"
    "/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_193412-195911.nc"
)
files = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"
all_vars = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/*/*.nc"
all_vars_v = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/v*/*.nc"


def test_open_dataset():
    da = xr.open_dataset(fpath)
    return da


def test_open_multiple_files():
    da = xr.open_mfdataset(files, combine="by_coords")
    return da


# FutureWarning: In xarray version 0.15 the default behaviour of `open_mfdataset`
#   will change. To retain the existing behavior, pass
#   combine='nested'. To use future default behavior, pass
#   combine='by_coords'.


def test_return_shape_of_file():
    da = xr.open_dataset(fpath)
    shape = da.tas.shape
    return shape
    # returns (time, lat, lon)


def test_subset_by_date():
    ds = xr.open_mfdataset(files)
    subset = ds.sel(time=slice("1922-01-16", "1931-12-16"))
    assert subset.tas.shape == (120, 145, 192)


def test_subset_by_date_other_method_incorrect():
    try:
        ds = xr.open_mfdataset(files)
        subset = ds.time.loc["1922-01-16":"1931-12-16"]
        assert subset.shape == (120, 145, 192)
    except AssertionError as ex:
        pass


def test_subset_by_other_method():
    ds = xr.open_mfdataset(files)
    subset = ds.time.loc["1922-01-16":"1931-12-16"]
    assert subset.shape == (120,)
    # limits to only time data


# takes a very long time to run

# def test_open_multiple_variable_files():
#     try:
#         ds = xr.open_mfdataset(all_vars)
#         return ds
#     except xr.MergeError as ex:
#         pass


# E                   xarray.core.merge.MergeError: conflicting values for variable 'time_bnds' on
#                       objects to be combined:
# E                   first value: <xarray.Variable (time: 1752, bnds: 2)>
# E                   dask.array<shape=(1752, 2), dtype=float64, chunksize=(300, 2)>
# E                   second value: <xarray.Variable (time: 1752, bnds: 2)>
# E                   dask.array<shape=(1752, 2), dtype=float64, chunksize=(1752, 2)>


def test_open_multiple_variable_files_2():
    ds = xr.open_mfdataset(all_vars_v)
    print(ds)
    return ds


def test_subset_by_variable_incorrect():
    try:
        ds = xr.open_mfdataset(all_vars_v)
        subset = ds.sel(variable="tas")
        return subset
    except ValueError as ex:
        pass


def test_return_variables_available():
    ds = xr.open_mfdataset(all_vars_v)
    variables = ds.data_vars
    assert len(variables) == 5  # time, lat, lon, vas, va
    return variables


def test_subset_by_variable():
    ds = xr.open_mfdataset(all_vars_v)
    subset = ds[["vas"]]
    return subset


def test_avg_subset_along_time_incorrect():
    try:
        ds = xr.open_mfdataset(all_vars_v)
        subset = ds[["vas"]]
        max = subset.max(dim="time")
        assert max.shape == (144, 192)
    except AttributeError as ex:
        pass


def test_avg_subset_along_time():
    ds = xr.open_mfdataset(all_vars_v)
    subset = ds[["vas"]]
    max = subset.vas.max(dim="time")
    assert max.shape == (144, 192)


def test_max_of_a_time_slice():
    ds = xr.open_mfdataset(files)
    time_slice = ds.sel(time=slice("1922-01-16", "1931-12-16"))
    maximum = time_slice.tas.max(dim="time")
    assert maximum.shape == (145, 192)


def test_mean_of_lat_on_time_slice():
    ds = xr.open_mfdataset(files)
    time_slice = ds.sel(time=slice("1922-01-16", "1931-12-16"))
    mean = time_slice.tas.mean(dim="lat")
    assert mean.shape == (120, 192)


def test_plot():
    ds = xr.open_dataset(
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc"
    )
    dr = ds["tas"]
    ax = plt.axes(projection=ccrs.PlateCarree())
    dr.isel(time=0).plot.pcolormesh(ax=ax, vmin=230, vmax=300)
    ax.coastlines()
    plt.show()


def test_regridding_bilinear():
    ds = xr.open_dataset(
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc"
    )
    # regrid from 1.25x1.875 to 2.5x3,75
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.375, 89.375, 2.5)),
            "lon": (["lon"], np.arange(0.9375, 359.0625, 3.75)),
        }
    )
    regridder = xe.Regridder(ds, ds_out, "bilinear")
    da_out = regridder(ds.tas)
    ax = plt.axes(projection=ccrs.PlateCarree())
    da_out.isel(time=0).plot.pcolormesh(ax=ax, vmin=230, vmax=300)
    regridder.clean_weight_file()
    ax.coastlines()
    plt.show()


def test_regridding_nearest_s2d():
    ds = xr.open_dataset(
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc"
    )
    # regrid from 1.25x1.875 to 2.5x3,75
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.375, 89.375, 2.5)),
            "lon": (["lon"], np.arange(0.9375, 359.0625, 3.75)),
        }
    )
    regridder = xe.Regridder(ds, ds_out, "nearest_s2d")
    da_out = regridder(ds.tas)
    ax = plt.axes(projection=ccrs.PlateCarree())
    da_out.isel(time=0).plot.pcolormesh(ax=ax, vmin=230, vmax=300)
    regridder.clean_weight_file()
    ax.coastlines()
    plt.show()


def test_regridding_conservative():
    try:
        ds = xr.open_dataset(
            "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc"
        )
        # regrid from 1.25x1.875 to 2.5x3.75
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], np.arange(-89.375, 89.375, 2.5)),
                "lon": (["lon"], np.arange(0.9375, 359.0625, 3.75)),
            }
        )
        regridder = xe.Regridder(ds, ds_out, "conservative")
        da_out = regridder(ds.tas)
        ax = plt.axes(projection=ccrs.PlateCarree())
        da_out.isel(time=0).plot.pcolormesh(ax=ax, vmin=230, vmax=300)
        regridder.clean_weight_file()
        ax.coastlines()
        plt.show()
    except KeyError as err:
        pass  # need grid corner info

    # In some regridding cases, might need to get the areacello or areacella variable to
    # get the full description of the grid cells required for accurate regridding.
    # e.g. /badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp45/fx/atmos/fx/r0i0p0/latest/areacella
    # /areacella_fx_HadGEM2-ES_rcp45_r0i0p0.nc


def test_regridding_patch():
    ds = xr.open_dataset(
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc"
    )
    # regrid from 1.25x1.875 to 2.5x3.75
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.375, 89.375, 2.5)),
            "lon": (["lon"], np.arange(0.9375, 359.0625, 3.75)),
        }
    )
    regridder = xe.Regridder(ds, ds_out, "patch")
    da_out = regridder(ds.tas)
    ax = plt.axes(projection=ccrs.PlateCarree())
    da_out.isel(time=0).plot.pcolormesh(ax=ax, vmin=230, vmax=300)
    regridder.clean_weight_file()
    ax.coastlines()
    plt.show()


def test_avg_all_vars_time_slice():
    ds = xr.open_mfdataset(all_vars_v)
    time_slice = ds.sel(time=slice("1922-01-16", "1931-12-16"))
    for var in ["vas", "va"]:
        maximum = time_slice[var].max(dim="time")
        # assert maximum.shape == (144, 192) variables are different shapes
        return maximum


def test_vars_as_params(try_parametrisation_vars):
    ds = xr.open_mfdataset(all_vars_v)
    time_slice = ds.sel(time=slice("1922-01-16", "1931-12-16"))
    var = try_parametrisation_vars
    maximum = time_slice[var].max(dim="time")
    # assert maximum.shape == (144, 192) variables are different shapes
    return maximum


def test_sel_multi_level_file(tmpdir):
    ds = xr.open_mfdataset(
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp45/mon/atmos/Amon/r1i1p1/latest/ta/ta_Amon_HadGEM2-ES_rcp45_r1i1p1_209912-212411.nc"
    )
    subset = ds.ta.sel(
        time=slice("2100-12-16", "2120-12-16"),
        plev=slice(85000, 3000),
        lat=slice(50, 59),
        lon=slice(2, 352),
    )
    assert (2 <= subset["lon"].data).all() & (subset["lon"].data <= 352).all()
    assert (50 <= subset["lat"].data).all() & (subset["lat"].data <= 59).all()
    assert (3000 <= subset["plev"].data).all() & (subset["plev"].data <= 85000).all()
    assert (cftime.Datetime360Day(2100, 12, 16) <= subset["time"].data).all() & (
        subset["time"].data <= cftime.Datetime360Day(2120, 12, 16)
    ).all()
    subset.to_netcdf(path=tmpdir.mkdir("test_dir").join("example_dataset.nc"))
