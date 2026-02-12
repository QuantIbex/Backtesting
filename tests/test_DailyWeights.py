#%%
"""
Test suite for class BT.DailyWeights

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT


class TestDailyWeights(unittest.TestCase):
    """Obvious"""

    def _local_test_data_weights(self):
        """Obvious"""
        alloc_chg = np.array([[100, 200, 300, 400, 0], [0, 0, 0, 0, 0], [-10, -20, -30, -40, 100],
                              [-10, -20, -30, -40, 100], [0, 0, 0, 0, 0], [10, -20, 30, -40, 20]])
        close_prices = BT.Utils.generate_random_prices(n_periods=alloc_chg.shape[0], 
                                                       n_assets=alloc_chg.shape[1], seed=1).round()
        
        cls_val = np.zeros((6, 5))
        eod_val = np.zeros((6, 5))
        eod_val[0, :] = cls_val[0, :] + alloc_chg[0, :]
        for ii in range(1, cls_val.shape[0]):
            # ii = 1
            cls_val[ii, :] = eod_val[ii-1, :] * (1 + close_prices.pct_change().iloc[ii, :].values)
            eod_val[ii, :] = cls_val[ii, :] + alloc_chg[ii, :]

        close_alloc = pd.DataFrame(cls_val, index = close_prices.index, columns = close_prices.columns)
        eod_alloc = pd.DataFrame(eod_val, index = close_prices.index, columns = close_prices.columns)

        close_weights = close_alloc.div(close_alloc.sum(axis=1), axis=0)
        eod_weights = eod_alloc.div(eod_alloc.sum(axis=1), axis=0)

        return {"close_prices": close_prices, "close_alloc": close_alloc, "eod_alloc": eod_alloc,
                "close_weights": close_weights, "eod_weights":eod_weights}

    def test_equal_weights(self):
        """Obvious"""
        dt = pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05"])
        cols = ["A", "B", "C"]
        px = np.array([[np.nan, np.nan, 300],
                      [101, np.nan, 301],
                      [102, 202, 302],
                      [103, 203, 303],
                      [104, 204, np.nan]])
        prices = pd.DataFrame(px, index = dt, columns=cols)
                
        wgt = [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [0.5, 0.5, 0.0]]
        expected = pd.DataFrame(wgt, index = dt, columns=cols)
        actual = BT.DailyWeights.equal_weights(prices = prices)
        pd.testing.assert_frame_equal(actual, expected)
        
    def test_drifting_weights(self):
        """Obvious"""
        wgt_0 = [[0.1, 0.2, 0.3, 0.4, 0.0]]
        prices = BT.Utils.generate_random_prices(n_periods=6, n_assets=len(wgt_0[0]), seed=1)
        start_weights = pd.DataFrame(wgt_0, index=[prices.index[0]], columns=prices.columns)
        notional = 1

        allocation = notional * prices.div(prices.values[0, :], axis=1).mul(start_weights.values, axis=1)
        expected = allocation.div(allocation.sum(axis=1), axis=0)
        actual = BT.DailyWeights.drifting_weights(start_weights = start_weights, prices = prices)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute_eod_weights(self):
        """Obvious"""

        res = self._local_test_data_weights()
        close_prices = res["close_prices"]
        close_weights = res["close_weights"]
        eod_weights = res["eod_weights"]

        expected = eod_weights
        expected.iloc[-1, :] = np.nan
        actual = BT.DailyWeights.compute_eod_weights(close_weights = close_weights, close_prices=close_prices)
        pd.testing.assert_frame_equal(actual, expected)
        
if __name__ == "__main__":
    unittest.main()
