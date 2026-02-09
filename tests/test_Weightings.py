#%%
"""
Test suite for class BT.Weightings

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestWeightings(unittest.TestCase):
    """Obvious"""

    def test_equally_weighted(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods = 4, freq = "ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        sigs = np.array([[-1.0, -1.0,  0.0, +1.0, +1.0], 
                         [+1.0, +1.0, -1.0, +1.0, +1.0], 
                         [ 0.0,  0.0,  0.0,  0.0,  0.0],
                         [-1.0, +1.0,  0.0, -1.0, np.nan]])
        wgts = np.array([[-0.5, -0.5,  0.0, +0.5, +0.5], 
                         [+0.25, +0.25, -1.0, +0.25, +0.25], 
                         [ 0.0,  0.0,  0.0,  0.0,  0.0],
                         [-0.5, +1.0,  0.0, -0.5, 0.0]])
        signals = pd.DataFrame(sigs, index = inds, columns = cols)
        weightings = pd.DataFrame(wgts, index = inds, columns = cols)
        
        expected = weightings
        actual = BT.Weightings.equally_weighted(signals = signals)
        pd.testing.assert_frame_equal(actual, expected)

    def test_signal_single(self):
        """Obvious"""
        inds = pd.date_range(start = "2019-12-31", periods = 4, freq = "ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        sigs = np.array([[-1.0, -1.0,  0.0, +1.0, +1.0], 
                         [+1.0, +1.0, -1.0, +1.0, +1.0], 
                         [ 0.0,  0.0,  0.0,  0.0,  0.0],
                         [-1.0, +1.0,  0.0, -1.0, np.nan]])
        signals = pd.DataFrame(sigs, index=inds, columns=cols)
        data = {"signals": signals}

        # equally-weighted
        specs = {"var_name": "signals", "type": "equally-weighted"}
        expected = BT.Weightings.equally_weighted(signals = data["signals"])
        actual = BT.Weightings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_aggregate(self):
        """Obvious"""

        def _local_generate_weightings(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed=1):
            if False:
                n_periods = 12
                n_neg = 2
                n_pos = 2
                n_0 = 2
                seed=1
            np.random.seed(seed)  # for reproducibility        
            inds = pd.date_range(start="2019-12-31", periods=n_periods, freq="ME")
            cols = [f"Asset_{i}" for i in range(1, n_neg + n_0 + n_pos +1)]
            base = np.array([-1/n_neg] * n_neg + [0] * n_0 + [1/n_pos] * n_pos)  #

            wgts = np.array([np.random.permutation(base) for _ in range(n_periods)])
            weightings = pd.DataFrame(wgts, index=inds, columns=cols)
            return weightings

        weightings = [
            _local_generate_weightings(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed = 1),
            _local_generate_weightings(n_periods = 12, n_neg = 2, n_pos = 1, n_0 = 3, seed = 2),
            _local_generate_weightings(n_periods = 12, n_neg = 4, n_pos = 2, n_0 = 0, seed = 3),
            _local_generate_weightings(n_periods = 12, n_neg = 1, n_pos = 2, n_0 = 3, seed = 4)]

        # Equally-weighted mean
        specs = {"method": "mean"}
        expected = BT.Utils.weighted_mean_dfs(dfs = weightings, weights = None)
        actual = BT.Weightings.aggregate(specs = specs, weightings = weightings)
        pd.testing.assert_frame_equal(actual, expected)

        # Non-equally-weighted mean
        wgts = [1, 2, 3, 4]
        specs = {"method": "mean", "weights": wgts}
        expected = BT.Utils.weighted_mean_dfs(dfs = weightings, weights = wgts) 
        actual = BT.Weightings.aggregate(specs = specs, weightings = weightings)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute___several(self):
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

        data = {"signals_1": _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed=1),
                "signals_2": _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed=1),
                "signals_3": _local_generate_signals(n_periods = 12, n_neg = 2, n_pos = 2, n_0 = 2, seed=1)}
        w_specs_1 = {"var_name": "signals_1", "type": "equally-weighted"}
        w_specs_2 = {"var_name": "signals_2", "type": "equally-weighted"}
        w_specs_3 = {"var_name": "signals_3", "type": "equally-weighted"}
        a_specs = {"method": "mean", "weights": [1, 2, 3]}
        wgt_1 = BT.Weightings.equally_weighted(signals=data["signals_1"])
        wgt_2 = BT.Weightings.equally_weighted(signals=data["signals_2"])
        wgt_3 = BT.Weightings.equally_weighted(signals=data["signals_3"])
        agg_wgt = BT.Signals.aggregate(specs = a_specs, signals = [wgt_1, wgt_2, wgt_3])
        
        specs = {"weightings": [w_specs_1, w_specs_2, w_specs_3], "aggregate": a_specs}
        expected = {"singles": [wgt_1, wgt_2, wgt_3], "global": agg_wgt}
        actual = BT.Weightings.compute(specs = specs, data = data)
        self.assertEqual(len(actual["singles"]), 3)
        pd.testing.assert_frame_equal(actual["singles"][0], expected["singles"][0])
        pd.testing.assert_frame_equal(actual["singles"][1], expected["singles"][1])
        pd.testing.assert_frame_equal(actual["singles"][2], expected["singles"][2])
        pd.testing.assert_frame_equal(actual["global"], expected["global"])

if __name__ == "__main__":
    unittest.main()
