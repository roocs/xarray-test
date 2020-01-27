import pytest
from netCDF4 import Dataset

# conftest allows fixture functions to be used in multiple test files
@pytest.fixture
def input_value():
    input = 39
    return input


# this fixture returns a function which generates data
@pytest.fixture
def make_customer_record():
    def _make_customer_record(name):
        return {"name": name, "orders": []}

    return _make_customer_record


# tries the test with both parameters - to see if the test passes with both
@pytest.fixture(scope="module", params=["purple", "pink"])
def try_parametrisation(request):
    colour = request.param
    return colour


@pytest.fixture
def create_netcdf_file(tmpdir):
    p = tmpdir.mkdir("test_dir").join("test_file.nc")
    test_file = Dataset(p, "w", format="NETCDF4")
    return test_file
