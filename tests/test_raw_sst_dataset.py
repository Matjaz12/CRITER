import os
import sys
import unittest
from parameterized import parameterized
import netCDF4
import numpy as np
from numpy.testing import assert_almost_equal

sys.path.append("../")
from datasets.raw_sst_data import get_raw_sst_data
from datasets.raw_sst_data import _get_sin_and_cos_doy, _get_sst, _get_land_mask


class Test_Raw_SST_Data(unittest.TestCase):
    """Test the Raw SST Dataset"""

    @parameterized.expand(
        [
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", 100, 256, 256),
            ("CHEMS_L3_SST_Adriatic.nc", 200, 144, 180),
            ("CHEMS_SST_Atlantic.nc", 120, 256, 256),
        ]
    )
    def test_shapes(self, filename, n_samples, height, width):
        """Test shape of the measurement tensor, land mask, longitude, latitude, doy_cos, doy_sin."""
        data_path = os.path.join(os.environ["DATA_PATH"], filename)
        data = get_raw_sst_data(data_path, n_samples)
        assert data.sst.shape == (n_samples, height, width)
        assert data.land_mask.shape == (height, width)
        assert data.lon.shape == (width,)
        assert data.lat.shape == (height,)
        assert data.doy_cos.shape == (n_samples,)
        assert data.doy_sin.shape == (n_samples,)

    @parameterized.expand(
        [
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", 100),
            ("CHEMS_L3_SST_Adriatic.nc", 200),
            ("CHEMS_SST_Atlantic.nc", 120),
        ]
    )
    def test_effects_of_change_spatial_res_on_lon_and_lat(self, filename, n_samples):
        """
        Test if changing the spatial resolution, preserves the strictly
        increasing nature of longitude and latitude information.
        """
        data_path = os.path.join(os.environ["DATA_PATH"], filename)
        data = get_raw_sst_data(data_path, n_samples)

        # check if lon and lat are strictly increasing after pre-processing
        assert np.all(np.diff(data.lat) > 0)
        assert np.all(np.diff(data.lon) > 0)

    @parameterized.expand(
        [
            (
                "SST_L3_CMEMS_2006-2021_Mediterranean.nc",
                "time",
                "sea_surface_temperature",
                100,
            ),
            (
                "CHEMS_L3_SST_Adriatic.nc",
                "time",
                "adjusted_sea_surface_temperature",
                100,
            ),
            ("CHEMS_SST_Atlantic.nc", "time", "adjusted_sea_surface_temperature", 100),
        ]
    )
    def test_helper_functions_output_range(
        self, filename, time_var_name, sst_var_name, n_samples
    ):
        """
        Test if helper functions yield expected values: test if doy_cos and doy_sin are correctly computed,
        test the codomain and realtion ship and trig identities, test if temperature is above 0^C (273.15^K),
        test if the domain of land masks is zeros and ones.
        """
        data_path = os.path.join(os.environ["DATA_PATH"], filename)
        data = netCDF4.Dataset(data_path, mode="r")

        # check if outputs are as expected
        doy_sin, doy_cos = _get_sin_and_cos_doy(time_var_name, n_samples, data)
        assert np.all((doy_sin >= -1) & (doy_sin <= 1))
        assert np.all((doy_cos >= -1) & (doy_cos <= 1))
        assert_almost_equal(doy_sin**2 + doy_cos**2, 1)
        sst = _get_sst(sst_var_name, False, n_samples, data)
        assert np.all(sst > 273.15)
        sst = _get_sst(sst_var_name, True, n_samples, data)
        assert np.all(sst > 0)
        land_mask = _get_land_mask(sst_var_name, n_samples, data)
        assert np.unique(land_mask).tolist() == [0, 1]


if __name__ == "__main__":
    unittest.main()
