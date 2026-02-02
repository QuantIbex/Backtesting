#%%
"""
Test suite for class BT.Grops

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT


class TestGroups(unittest.TestCase):
    """Obvious"""

    # TODO: test defaults


    def test_none(self):
        """Obvious"""
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6)

        # Defaults
        expected = BT.Groups.none(prices=prices, ref_date=prices.index.values[[-1]])
        actual = BT.Groups.none(prices=prices)
        pd.testing.assert_frame_equal(actual, expected)

        # One date as 
        dt_ind = [-4]
        grps = [prices.columns.values]
        ref_date = prices.index.values[dt_ind]
        expected = pd.DataFrame(grps, index = ref_date, columns = prices.columns.values)
        actual = BT.Groups.none(prices=prices, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)

        # Several dates
        dt_ind = [-6, -4, -2]
        grps = [prices.columns.values]
        ref_date = prices.index.values[dt_ind]
        expected = pd.DataFrame(grps, index = ref_date, columns = prices.columns.values)
        actual = BT.Groups.none(prices=prices, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
