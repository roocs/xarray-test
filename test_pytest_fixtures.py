import xarray as xr

# example pytest fixture taken from the internet


def test_divisible_by_3(input_value):
    assert input_value % 3 == 0


def test_divisible_by_6(input_value):
    try:
        assert input_value % 6 == 0
    except AssertionError as err:
        pass


def test_customer_records(make_customer_record):
    customer_1 = make_customer_record("Lisa")
    customer_2 = make_customer_record("Mike")
    assert customer_1["name"] == "Lisa"
    assert customer_2["name"] == "Mike"


def test_parametrisation(try_parametrisation):
    assert try_parametrisation == "purple"


def test_open_dataset(create_netcdf_file):
    da = create_netcdf_file
    print(da)
    return da
