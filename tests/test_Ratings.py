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

#  test default values
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

    def test_rating_single___default_values(self):
        """Obvious"""
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = 6)

        # lback = 5
        # expected = BT.Metrics.momentum(prices, lookback=lback, skip=0, only_last=True)
        # actual = BT.Metrics.momentum(prices, lookback=lback)
        # pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
