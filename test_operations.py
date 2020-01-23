import xarray as xr
import numpy as np
# import xesmf as xe

fpath = '/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest' \
        '/tas/tas_Amon_HadGEM2-ES_historical_r1i1p1_193412-195911.nc'
files = '/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/tas/*.nc'
all_vars = '/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/*/*.nc'
all_vars_v = '/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/v*/*.nc'


def test_open_dataset():
    da = xr.open_dataset(fpath)
    return da


def test_open_multiple_files():
    da = xr.open_mfdataset(files)
    return da


def test_return_shape_of_file():
    da = xr.open_dataset(fpath)
    shape = da.tas.shape
    return shape
    # returns (time, lat, lon)


def test_subset_by_date():
    ds = xr.open_mfdataset(files)
    subset = ds.sel(time=slice('1922-01-16', '1931-12-16'))
    assert subset.tas.shape == (120, 145, 192)


def test_subset_by_date_other_method_incorrect():
    try:
        ds = xr.open_mfdataset(files)
        subset = ds.time.loc['1922-01-16':'1931-12-16']
        assert subset.shape == (120, 145, 192)
    except AssertionError as ex:
        pass


def test_subset_by_other_method():
    ds = xr.open_mfdataset(files)
    subset = ds.time.loc['1922-01-16':'1931-12-16']
    assert subset.shape == (120,)
    # limits to only time data


def test_open_multiple_variable_files():
    try:
        ds = xr.open_mfdataset(all_vars)
        return ds
    except xr.MergeError as ex:
        pass


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
        subset = ds.sel(variable='tas')
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
    subset = ds[['vas']]
    return subset


def test_avg_subset_along_time_incorrect():
    try:
        ds = xr.open_mfdataset(all_vars_v)
        subset = ds[['vas']]
        max = subset.max(dim='time')
        assert max.shape == (144, 192)
    except AttributeError as ex:
        pass


def test_avg_subset_along_time():
    ds = xr.open_mfdataset(all_vars_v)
    subset = ds[['vas']]
    max = subset.vas.max(dim='time')
    assert max.shape == (144, 192)


def test_max_of_a_time_slice():
    ds = xr.open_mfdataset(files)
    time_slice = ds.sel(time=slice('1922-01-16', '1931-12-16'))
    maximum = time_slice.tas.max(dim='time')
    assert maximum.shape == (145, 192)


def test_mean_of_lat_on_time_slice():
    ds = xr.open_mfdataset(files)
    time_slice = ds.sel(time=slice('1922-01-16', '1931-12-16'))
    mean = time_slice.tas.mean(dim='lat')
    assert mean.shape == (120, 192)


# def test_regridding():
#     ds = xr.open_mfdataset('/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/atmos/Amon/r1i1p1/latest/ta/*.nc')
#
#     ds_out = xr.Dataset({'lat': (['lat'], np.arange(16, 75, 1.0)),
#                          'lon': (['lon'], np.arange(200, 330, 1.5)),
#                          }
#                         )
#
#     regridder = xe.Regridder(ds, ds_out, 'bilinear')
#     regridder.clean_weight_file()
#     ds_out = regridder(ds)
#     print(ds_out)