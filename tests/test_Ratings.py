#%%
"""
Test suite for class BT.Ratings

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestMetrics(unittest.TestCase):
    """Obvious"""

    def test_identity(self):
        """Obvious"""
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6)

        expected = prices
        actual = BT.Ratings.identity(metrics=prices)
        pd.testing.assert_frame_equal(actual, expected)

    def test_rank(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        rnks = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], 
                         [1.0, 5.0, 2.0, 4.0, 3.0], [1.0, 3.0, 4.0, np.nan, 2.0]])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = pd.DataFrame(rnks, index=inds, columns=cols)
        actual = BT.Ratings.rank(metrics=metrics)
        pd.testing.assert_frame_equal(actual, expected)

    def test_uscore___defaults(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = BT.Ratings.uscore(metrics=metrics, scaling="n-1")
        actual = BT.Ratings.uscore(metrics=metrics)
        pd.testing.assert_frame_equal(actual, expected)

    def test_uscore___n_minus_1(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        rtgs = np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25, 0.0], 
                         [0.0, 1.0, 0.25, 0.75, 0.5], [0.0, 2/3, 1.0, np.nan, 1/3]])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = pd.DataFrame(rtgs, index=inds, columns=cols)
        actual = BT.Ratings.uscore(metrics=metrics, scaling="n-1")
        pd.testing.assert_frame_equal(actual, expected)

    def test_uscore___n(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        rtgs = np.array([[0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 0.8, 0.6, 0.4, 0.2],
                         [0.2, 1.0, 0.4, 0.8, 0.6], [0.25, 0.75, 1.0, np.nan, 0.5]])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = pd.DataFrame(rtgs, index=inds, columns=cols)
        actual = BT.Ratings.uscore(metrics=metrics, scaling="n")
        pd.testing.assert_frame_equal(actual, expected)

    def test_uscore___n_plus_1(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        rtgs = np.array([[1/6, 2/6, 3/6, 4/6, 5/6], [5/6, 4/6, 3/6, 2/6, 1/6],
                         [1/6, 5/6, 2/6, 4/6, 3/6], [1/5, 3/5, 4/5, np.nan, 2/5]])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        
        expected = pd.DataFrame(rtgs, index=inds, columns=cols)
        actual = BT.Ratings.uscore(metrics=metrics, scaling="n+1")
        pd.testing.assert_frame_equal(actual, expected)

    def test_zscore(self):
        """Obvious"""
        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)

        mu = np.array([np.repeat(2.0, 5), np.repeat(2.0, 5), np.repeat(13.0, 5), np.repeat(12.75, 5)])
        sig =np.tile(np.reshape(metrics.std(axis=1).values, (-1, 1)), 5) 
        expected = (metrics - mu) / sig
        actual = BT.Ratings.zscore(metrics=metrics)
        pd.testing.assert_frame_equal(actual, expected)

    def test_rating_single(self):
        """Obvious"""

        inds = pd.date_range(start="2019-12-31", periods=4, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, 6)]
        vals = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [11, 15, 12, 14, 13], 
                         [11, 13, 15, np.nan, 12 ] ])
        rtgs = np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25, 0.0], 
                         [0.0, 1.0, 0.25, 0.75, 0.5], [0.0, 2/3, 1.0, np.nan, 1/3]])
        metrics = pd.DataFrame(vals, index=inds, columns=cols)
        data = {"metrics": metrics}

        # identity
        specs = {"var_name": "metrics", "type": "identity"}
        expected = BT.Ratings.identity(metrics = data["metrics"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

        # rank
        specs = {"var_name": "metrics", "type": "rank"}
        expected = BT.Ratings.rank(metrics = data["metrics"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

        # uscore - scaling: n - 1
        specs = {"var_name": "metrics", "type": "uscore", "scaling": "n-1"}
        expected = BT.Ratings.uscore(metrics = data["metrics"], scaling=specs["scaling"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

        # uscore - scaling: n
        specs = {"var_name": "metrics", "type": "uscore", "scaling": "n"}
        expected = BT.Ratings.uscore(metrics = data["metrics"], scaling=specs["scaling"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

        # uscore - scaling: n + 1
        specs = {"var_name": "metrics", "type": "uscore", "scaling": "n+1"}
        expected = BT.Ratings.uscore(metrics = data["metrics"], scaling=specs["scaling"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

        # zscore
        specs = {"var_name": "metrics", "type": "zscore"}
        expected = BT.Ratings.zscore(metrics = data["metrics"])
        actual = BT.Ratings.compute_single(specs=specs, data=data)
        pd.testing.assert_frame_equal(actual, expected)

    def test_aggregate(self):
        """Obvious"""

        def _local_generate_ratings(n_periods = 12, n_assets = 6, seed=1):
            if False:
                n_periods = 12
                n_assets = 6
                seed=1
            np.random.seed(seed)  # for reproducibility        
            inds = pd.date_range(start="2019-12-31", periods=n_periods, freq="ME")
            cols = [f"Asset_{i}" for i in range(1, n_assets+1)]
            base = np.arange(1, n_assets + 1)
            rtgs = np.array([np.random.permutation(base) for _ in range(n_periods)])
            ratings = pd.DataFrame(rtgs, index=inds, columns=cols)
            return ratings


        ratings = [_local_generate_ratings(n_periods = 12, n_assets = 6, seed=1),
            _local_generate_ratings(n_periods = 12, n_assets = 6, seed=2),
            _local_generate_ratings(n_periods = 12, n_assets = 6, seed=3),
            _local_generate_ratings(n_periods = 12, n_assets = 6, seed=4)]


        # Equally-weighted mean
        specs = {"method": "mean"}
        expected = BT.Utils.weighted_mean_dfs(dfs = ratings, weights = None)
        actual = BT.Ratings.aggregate(specs = specs, ratings = ratings)
        pd.testing.assert_frame_equal(actual, expected)

        # Non-equally-weighted mean
        wgts = [1, 2, 3, 4]
        specs = {"method": "mean", "weights": wgts}
        expected = BT.Utils.weighted_mean_dfs(dfs = ratings, weights = wgts) 
        actual = BT.Ratings.aggregate(specs = specs, ratings = ratings)
        pd.testing.assert_frame_equal(actual, expected)

    def test_compute___several(self):
        """Obvious"""

        def _local_generate_metrics(n_periods = 12, n_assets = 6, seed=1):
            if False:
                n_periods = 12
                n_assets = 6
                seed=1
            prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6, seed=seed)
            metrics = prices.pct_change(periods=6)
            return metrics

        data = {"metrics": _local_generate_metrics(n_periods = 12, n_assets = 6, seed=1)}
        r_specs_1 = {"var_name": "metrics", "type": "uscore", "scaling": "n-1"}
        r_specs_2 = {"var_name": "metrics", "type": "uscore", "scaling": "n"}
        r_specs_3 = {"var_name": "metrics", "type": "uscore", "scaling": "n+1"}
        a_specs = {"method": "mean", "weights": [1, 2, 3]}
        rat_1 = BT.Ratings.uscore(metrics=data["metrics"], scaling = r_specs_1["scaling"])
        rat_2 = BT.Ratings.uscore(metrics=data["metrics"], scaling = r_specs_2["scaling"])
        rat_3 = BT.Ratings.uscore(metrics=data["metrics"], scaling = r_specs_3["scaling"])
        agg_rat = BT.Ratings.aggregate(specs = a_specs, ratings=[rat_1, rat_2, rat_3])
        
        specs = {"ratings": [r_specs_1, r_specs_2, r_specs_3], "aggregate": a_specs}
        expected = {"singles": [rat_1, rat_2, rat_3], "global": agg_rat}
        actual = BT.Ratings.compute(specs = specs, data = data)
        self.assertEqual(len(actual["singles"]), 3)
        pd.testing.assert_frame_equal(actual["singles"][0], expected["singles"][0])
        pd.testing.assert_frame_equal(actual["singles"][1], expected["singles"][1])
        pd.testing.assert_frame_equal(actual["singles"][2], expected["singles"][2])
        pd.testing.assert_frame_equal(actual["global"], expected["global"])


if __name__ == "__main__":
    unittest.main()
