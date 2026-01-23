#%%
"""
Test suite for class BT.Metrics

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
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

    def test_compute_single(self):
        """Obvious"""
        prices = self._generate_test_prices(n_periods = 12, n_assets = 6)
        data = {"prices": prices}
        specs = {"type": "MOMENTUM", "lookback": 5, "skip": 2, "only_last": False}
        
        expected = BT.Metrics.momentum(prices=data["prices"], lookback=specs.get("lookback"),
                                       skip=specs.get("skip"), only_last=specs.get("only_last"))
        actual = BT.Metrics.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_aggregate(self):
        """Obvious"""
        metrics = [BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=2),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=3),
            BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=4)]

        # Equally-weighted mean
        specs = {"method": "mean"}
        expected = BT.Utils.weighted_mean_dfs(dfs = metrics, weights = None) 
        actual = BT.Metrics.aggregate(specs = specs, metrics= metrics)
        pd.testing.assert_frame_equal(actual, expected)

        # Non-equally-weighted mean
        wgts = [1, 2, 3, 4]
        specs = {"method": "mean", "weights": wgts}
        expected = BT.Utils.weighted_mean_dfs(dfs = metrics, weights = wgts) 
        actual = BT.Metrics.aggregate(specs = specs, metrics = metrics)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute_one(self):
        """Obvious"""

        data = {"prices": BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1)}
        m_specs = {"type": "MOMENTUM", "lookback": 5, "skip": 2, "only_last": False}
        mom = BT.Metrics.momentum(prices=data["prices"], lookback=m_specs["lookback"],
                                       skip=m_specs["skip"], only_last=m_specs["only_last"])

        # Specs as dict
        specs1 = {"metrics": m_specs}
        expected = {"single_metrics": [mom], "global_metrics": mom}
        actual = BT.Metrics.compute(specs = specs1, data = data)
        self.assertEqual(len(actual["single_metrics"]), 1)
        pd.testing.assert_frame_equal(actual["single_metrics"][0], mom)
        pd.testing.assert_frame_equal(actual["global_metrics"], mom)

        # Specs in list
        specs2 = {"metrics": [m_specs]}
        expected = {"single_metrics": [mom], "global_metrics": mom}
        actual = BT.Metrics.compute(specs = specs2, data = data)
        self.assertEqual(len(actual["single_metrics"]), 1)
        pd.testing.assert_frame_equal(actual["single_metrics"][0], expected["single_metrics"][0])
        pd.testing.assert_frame_equal(actual["global_metrics"], expected["global_metrics"])

    def test_compute_several(self):
        """Obvious"""

        data = {"prices": BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=1)}
        m_specs_1 = {"type": "MOMENTUM", "lookback": 5, "skip": 2, "only_last": False}
        m_specs_2 = {"type": "MOMENTUM", "lookback": 7, "skip": 3, "only_last": False}
        a_specs = {"method": "mean", "weights": [1, 2]}
        mom_1 = BT.Metrics.momentum(prices=data["prices"], lookback=m_specs_1["lookback"],
                                       skip=m_specs_1["skip"], only_last=m_specs_1["only_last"])
        mom_2 = BT.Metrics.momentum(prices=data["prices"], lookback=m_specs_2["lookback"],
                                       skip=m_specs_2["skip"], only_last=m_specs_2["only_last"])
        agg_mom = BT.Metrics.aggregate(specs = a_specs, metrics=[mom_1, mom_2])
        
        # Specs in list
        specs = {"metrics": [m_specs_1, m_specs_2], "aggregate": a_specs}
        expected = {"single_metrics": [mom_1, mom_2], "global_metrics": agg_mom}
        actual = BT.Metrics.compute(specs = specs, data = data)
        self.assertEqual(len(actual["single_metrics"]), 2)
        pd.testing.assert_frame_equal(actual["single_metrics"][0], expected["single_metrics"][0])
        pd.testing.assert_frame_equal(actual["single_metrics"][1], expected["single_metrics"][1])
        pd.testing.assert_frame_equal(actual["global_metrics"], expected["global_metrics"])


if __name__ == "__main__":
    unittest.main()
