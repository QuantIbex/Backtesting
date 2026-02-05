#%%
"""
Test suite for class BT.Signals

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestSignals(unittest.TestCase):
    """Obvious"""

    def test_top_bottom___defaults(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13],
                         [11, 13, 15, np.nan, 12 ] ])
        ratings = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = BT.Signals.top_bottom(ratings = ratings, top = 0, bottom = 0)
        actual = BT.Signals.top_bottom(ratings = ratings)
        pd.testing.assert_frame_equal(actual, expected)

    def test_top_bottom(self):
        """Obvious"""
        inds = pd.date_range(start = "2019-12-31", periods = 4, freq = "ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13],
                         [11, 13, 15, np.nan, 12 ] ])
        sgn = np.array([[-1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, -1.0], [-1.0, 1.0, 0.0, 1.0, 0.0],
                         [-1.0, 1.0, 1.0, 0.0, 0.0 ] ])        
        ratings = pd.DataFrame(vals, index=inds, columns=cols)

        expected = pd.DataFrame(sgn, index=inds, columns=cols)
        actual = BT.Signals.top_bottom(ratings = ratings, top = 2, bottom = 1)
        pd.testing.assert_frame_equal(actual, expected)

    def test_signal_single(self):
        """Obvious"""
        inds = pd.date_range(start = "2019-12-31", periods = 4, freq = "ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13],
                         [11, 13, 15, np.nan, 12 ] ])
        ratings = pd.DataFrame(vals, index=inds, columns=cols)

        data = {"ratings": ratings}

        # top-bottom
        specs = {"var_name": "ratings", "type": "top-bottom", "top": 1, "bottom": 2}
        expected = BT.Signals.top_bottom(ratings = data["ratings"], 
                                         top = specs.get("top"), 
                                         bottom = specs.get("bottom"))
        actual = BT.Signals.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_aggregate(self):
        """Obvious"""

        def _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed=1):
            if False:
                n_periods = 12
                n_neg = 2
                n_pos = 2
                n_0 = 2
                seed=1
            np.random.seed(seed)  # for reproducibility        
            inds = pd.date_range(start="2019-12-31", periods=n_periods, freq="ME")
            cols = [f"Asset_{i}" for i in range(1, n_neg + n_0 + n_pos +1)]
            base = np.array([-1] * n_neg + [0] * n_0 + [1] * n_pos)  #

            sgnls = np.array([np.random.permutation(base) for _ in range(n_periods)])
            signals = pd.DataFrame(sgnls, index=inds, columns=cols)
            return signals

        signals = [
            _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed = 1),
            _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 1, n_0 = 3, seed = 2),
            _local_generate_signals(n_periods = 12, n_neg = 4, n_pos = 2, n_0 = 0, seed = 3),
            _local_generate_signals(n_periods = 12, n_neg = 1, n_pos = 2, n_0 = 3, seed = 4)]


        # Equally-weighted mean
        specs = {"method": "mean"}
        expected = BT.Utils.weighted_mean_dfs(dfs = signals, weights = None)
        actual = BT.Signals.aggregate(specs = specs, signals = signals)
        pd.testing.assert_frame_equal(actual, expected)

        # Non-equally-weighted mean
        wgts = [1, 2, 3, 4]
        specs = {"method": "mean", "weights": wgts}
        expected = BT.Utils.weighted_mean_dfs(dfs = signals, weights = wgts) 
        actual = BT.Signals.aggregate(specs = specs, signals = signals)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute___several(self):
        """Obvious"""

        def _local_generate_ratings(n_periods = 12, n_assets = 6, seed=1):
            if False:
                n_periods = 12
                n_assets = 6
                seed=1
            prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=seed)
            ratings = prices.rank(axis=1)
            return ratings

        data = {"ratings": _local_generate_ratings(n_periods = 12, n_assets = 6, seed=1)}
        s_specs_1 = {"var_name": "ratings", "type": "top-bottom", "top": 2, "bottom": 1}
        s_specs_2 = {"var_name": "ratings", "type": "top-bottom", "top": 1, "bottom": 2}
        s_specs_3 = {"var_name": "ratings", "type": "top-bottom", "top": 1, "bottom": 1}
        a_specs = {"method": "mean", "weights": [1, 2, 3]}
        sig_1 = BT.Signals.top_bottom(ratings=data["ratings"], top = s_specs_1["top"], bottom = s_specs_1["bottom"])
        sig_2 = BT.Signals.top_bottom(ratings=data["ratings"], top = s_specs_2["top"], bottom = s_specs_2["bottom"])
        sig_3 = BT.Signals.top_bottom(ratings=data["ratings"], top = s_specs_3["top"], bottom = s_specs_3["bottom"])
        agg_sig = BT.Signals.aggregate(specs = a_specs, signals = [sig_1, sig_2, sig_3])
        
        specs = {"signals": [s_specs_1, s_specs_2, s_specs_3], "aggregate": a_specs}
        expected = {"singles": [sig_1, sig_2, sig_3], "global": agg_sig}
        actual = BT.Signals.compute(specs = specs, data = data)
        self.assertEqual(len(actual["singles"]), 3)
        pd.testing.assert_frame_equal(actual["singles"][0], expected["singles"][0])
        pd.testing.assert_frame_equal(actual["singles"][1], expected["singles"][1])
        pd.testing.assert_frame_equal(actual["singles"][2], expected["singles"][2])
        pd.testing.assert_frame_equal(actual["global"], expected["global"])


if __name__ == "__main__":
    unittest.main()
