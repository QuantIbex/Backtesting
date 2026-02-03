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

    def test_none(self):
        """Obvious"""
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6)

        # Defaults
        expected = BT.Groups.none(prices=prices, ref_date=prices.index.values[[-1]])
        actual = BT.Groups.none(prices=prices)
        pd.testing.assert_frame_equal(actual, expected)

        # One date as 
        dt_ind = [-4]
        ref_date = prices.index.values[dt_ind]
        grps = [prices.columns.values]
        expected = pd.DataFrame(grps, index = ref_date, columns = prices.columns.values)
        actual = BT.Groups.none(prices=prices, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)

        # Several dates
        dt_ind = [-6, -4, -2]
        ref_date = prices.index.values[dt_ind]
        grps = [prices.columns.values]
        expected = pd.DataFrame(grps, index = ref_date, columns = prices.columns.values)
        actual = BT.Groups.none(prices=prices, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)

    def test_labels(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6+1)]
        lbs = [["Group_1", "Group_1", "Group_2", "Group_2", "Group_3", "Group_3"],
               ["Group_1", "Group_2", "Group_3", "Group_1", "Group_2", "Group_3"],
               ["Group_3", "Group_2", "Group_1", "Group_3", "Group_2", "Group_1"],
               ["Group_1", "Group_2", "Group_2", "Group_3", "Group_3", "Group_3"]]
        labels = pd.DataFrame(lbs, index=inds, columns=cols)

        # Defaults
        expected = BT.Groups.labels(labels=labels, ref_date=labels.index.values[[-1]])
        actual = BT.Groups.labels(labels=labels)
        pd.testing.assert_frame_equal(actual, expected)

        # One date
        ref_date = labels.index.values[[-2]]
        expected = labels.loc[ref_date, :]
        actual = BT.Groups.labels(labels=labels, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)

        # One date as 
        ref_date = labels.index.values[[-2, -3]]
        expected = labels.loc[ref_date, :]
        actual = BT.Groups.labels(labels=labels, ref_date=ref_date)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
