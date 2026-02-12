#%%
"""
Test suite for class BT.Utils

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestMetrics(unittest.TestCase):
    """Obvious"""

    def test_all_same_index_columns(self):
        """Obvious"""

        # Only 1 df
        dfs = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1)]
        actual = BT.Utils.all_same_index_columns(dfs)
        self.assertTrue(actual)

        # Several matching dfs
        dfs = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=2),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=3),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=4)]
        actual = BT.Utils.all_same_index_columns(dfs)
        self.assertTrue(actual)

        # Some unmatching dfs
        dfs = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=2),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=3),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=4)]
        dfs[2] = dfs[2].rename(columns={dfs[2].columns[2]: 'TOTO'})
        actual = BT.Utils.all_same_index_columns(dfs)
        self.assertFalse(actual)

    def test_weighted_mean_dfs(self):
        """Obvious"""
        dfs = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=2),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=3),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=4)]

        # Defaults
        expected = BT.Utils.weighted_mean_dfs(dfs = dfs, weights = [0.25, 0.25, 0.25, 0.25])
        actual = BT.Utils.weighted_mean_dfs(dfs = dfs)        
        pd.testing.assert_frame_equal(actual, expected)

        # No weights
        expected = (dfs[0] + dfs[1] + dfs[2] + dfs[3]) / 4
        actual = BT.Utils.weighted_mean_dfs(dfs = dfs, weights = None)
        pd.testing.assert_frame_equal(actual, expected)

        # Equal weights
        expected = (dfs[0] + dfs[1] + dfs[2] + dfs[3]) / 4
        actual = BT.Utils.weighted_mean_dfs(dfs = dfs, weights = [0.25, 0.25, 0.25, 0.25])
        pd.testing.assert_frame_equal(actual, expected)

        # Unequal weights
        wgts = [1, 2, 3, 4]
        expected = (dfs[0] + 2 * dfs[1] + 3 * dfs[2] + 4 *dfs[3]) / sum(wgts)
        actual = BT.Utils.weighted_mean_dfs(dfs = dfs, weights = wgts)
        pd.testing.assert_frame_equal(actual, expected)

    def test_is_frequency_date(self):
        """Obvious"""

        # Friday
        dt = pd.Timestamp('2026-02-27')
        freq = "W-FRI"
        actual = BT.Utils.is_frequency_date(date = dt, freq = freq, start_date = None, end_date = None)
        self.assertTrue(actual)

        # Saturday
        dt = pd.Timestamp('2026-02-27')
        freq = "W-SAT"
        actual = BT.Utils.is_frequency_date(date = dt, freq = freq, start_date = None, end_date = None)
        self.assertFalse(actual)

        # Business month end
        dt = pd.Timestamp('2026-02-27')
        freq = "BME" 
        actual = BT.Utils.is_frequency_date(date = dt, freq = freq, start_date = None, end_date = None)
        self.assertTrue(actual)

        # Calendar month end
        dt = pd.Timestamp('2026-02-27')
        freq = "ME" 
        actual = BT.Utils.is_frequency_date(date = dt, freq = freq, start_date = None, end_date = None)
        self.assertFalse(actual)


if __name__ == "__main__":
    unittest.main()
