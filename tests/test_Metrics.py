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

    def _generate_test_prices(self, n_periods: int, n_assets: int) -> pd.DataFrame:
        """
        Helper method. Generate a datafram with random prices.
        """
        if False:
            n_periods = 12
            n_assets = 6
        np.random.seed(0)  # for reproducibility        
        inds = pd.date_range(start="2019-12-31", periods=n_periods + 1, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, n_assets+1)]
        px = 90 + 20 * np.random.rand(n_periods + 1, n_assets)
        prices = pd.DataFrame(px, index=inds, columns=cols)
        return prices

    def test_momentum_default_values(self):
        """Obvious"""
        prices = self._generate_test_prices(n_periods = 12, n_assets = 6)

        lback = 5
        expected = BT.Metrics.momentum(prices, lookback=lback, skip=0, only_last=True)
        actual = BT.Metrics.momentum(prices, lookback=lback)
        pd.testing.assert_frame_equal(actual, expected)

    def test_momentum_no_skipped_values(self):
        """Obvious"""
        prices = self._generate_test_prices(n_periods = 12, n_assets = 6)

        # Case: all periods
        lback = 5
        n_row = prices.shape[0]
        dt_end = prices.index[lback:n_row]
        dt_start = prices. index[:(n_row - lback)]
        px_end = prices.loc[dt_end, :].values
        px_start = prices.loc[dt_start, :].values
        expected = pd.DataFrame(np.nan, index = prices.index, columns = prices.columns)
        expected.loc[dt_end, :] = px_end / px_start - 1
        actual = BT.Metrics.momentum(prices, lookback=lback, skip=0, only_last=False)
        pd.testing.assert_frame_equal(actual, expected)

        # Case: only last
        lback = 5
        expected = BT.Metrics.momentum(prices, lookback=lback, only_last=False).iloc[[-1]]
        actual = BT.Metrics.momentum(prices, lookback=lback, only_last=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_momentum_skipped_values(self):
        """Obvious"""
        prices = self._generate_test_prices(n_periods = 12, n_assets = 6)

        # Case: all periods
        lback = 5
        skp = 2
        n_row = prices.shape[0]
        px_dt_end = prices.index[(lback):(n_row - skp)]
        px_dt_start = prices. index[:(n_row - lback - skp)]
        dt_end = prices.index[(lback + skp):n_row]
        px_end = prices.loc[px_dt_end, :].values
        px_start = prices.loc[px_dt_start, :].values
        expected = pd.DataFrame(np.nan, index = prices.index, columns = prices.columns)
        expected.loc[dt_end, :] = px_end / px_start - 1
        actual = BT.Metrics.momentum(prices, lookback=lback, skip=skp, only_last=False)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute_single_metric(self):
        """Obvious"""
        prices = self._generate_test_prices(n_periods = 12, n_assets = 6)
        data = {"prices": prices}
        specs = {"type": "MOMENTUM", "lookback": 5, "skip": 2, "only_last": False}
        
        expected = BT.Metrics.momentum(prices=data["prices"], lookback=specs.get("lookback"),
                                       skip=specs.get("skip"), only_last=specs.get("only_last"))
        actual = BT.Metrics.compute_single_metric(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_aggregate_metrics(self):
        """Obvious"""
        metrics = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=2),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=3),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=4)]


        # Equally-weighted mean
        specs = {"method": "mean"}
        expected = BT.Utils.weighted_mean_dfs(dfs = metrics, weights = None) 
        actual = BT.Metrics.aggregate_metrics(specs = specs, metrics= metrics)
        pd.testing.assert_frame_equal(actual, expected)

        # Non-equally-weighted mean
        wgts = [1, 2, 3, 4]
        specs = {"method": "mean", "weights": wgts}
        expected = BT.Utils.weighted_mean_dfs(dfs = metrics, weights = wgts) 
        actual = BT.Metrics.aggregate_metrics(specs = specs, metrics= metrics)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
