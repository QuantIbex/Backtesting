#%%
"""
Test suite for class BT.Metrics

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    Did not work : python3 tests/test_Metrics.py
"""

import unittest
from Backtesting import BT
import numpy as np
import pandas as pd

class TestMetrics(unittest.TestCase):

    def test_momentum(self):
        """
        Tests method momentum
        """
        
        n_periods = 12
        n_assets = 6
        np.random.seed(0)  # for reproducibility        
        inds = pd.date_range(start="2019-12-31", periods=n_periods + 1, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, n_assets+1)]
        px = 90 + 20 * np.random.rand(n_periods + 1, n_assets)
        prices = pd.DataFrame(px, index=inds, columns=cols)

        # 
        lback = 6
        skp = 0
        n_row = prices.shape[0]

        px_end = prices.iloc[[n_row - lback- skp - 1], :].values
        px_start = prices.iloc[[n_row - skp - 1], :]
        expected = px_start / px_end - 1
        actual = BT.Metrics.momentum(prices, lookback=lback, skip=skp, only_last=True)
        
        pd.testing.assert_frames_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
