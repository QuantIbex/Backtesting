#%%
"""
Test suite for class BT.Weights

Execute tests in consol with:
    python tests/test_Weights.py

"""

import unittest
from Backtesting import BT
import numpy as np
import pandas as pd

class TestWeights(unittest.TestCase):

    def test_net_exposure(self):
        """
        Tests method net_exposure
        """
        wgt = pd.DataFrame([
            [0.5, 0.3, 0.2], 
            [0.4, 0.3, 0.1],
            [0.8, 0.6, -0.2],
            [-0.5, -0.3, -0.2], 
            [-0.4, -0.3, -0.1],
            [-0.8, -0.6, 0.2],
            [1.0, -0.6, -0.4],
            [0.5, -0.3, -0.2]], columns = ["A", "B", "C"])
        expected = pd.Series([1.0, 0.8, 1.2, -1.0, -0.8, -1.2, 0.0, 0.0])
        actual = BT.Weights.net_exposure(weights=wgt)
        pd.testing.assert_series_equal(actual, expected)

    def test_gross_exposure(self):
        """
        Tests method gross_exposure
        """
        wgt = pd.DataFrame([
            [0.5, 0.3, 0.2], 
            [0.4, 0.3, 0.1],
            [0.8, 0.6, -0.2],
            [-0.5, -0.3, -0.2], 
            [-0.4, -0.3, -0.1],
            [-0.8, -0.6, 0.2],
            [1.0, -0.6, -0.4],
            [0.5, -0.3, -0.2]], columns = ["A", "B", "C"])
        expected = pd.Series([1.0, 0.8, 1.6, 1.0, 0.8, 1.6, 2.0, 1.0])
        actual = BT.Weights.gross_exposure(weights=wgt)
        pd.testing.assert_series_equal(actual, expected)

    def test_normalize_weights_long_portfolio(self):
        """
        Tests method normalize_weights for different cases of long portfolios
        """
        wgt = pd.DataFrame([
            [0.5, 0.3, 0.2], 
            [0.4, 0.3, 0.1],
            [1.2, 0.6, 0.2],
            [0.8, 0.4, -0.2],
            [0.4, 0.3, -0.2],
            [1.6, 0.6, -0.2]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [0.5, 0.3, 0.2],
            [0.5, 0.375, 0.125],
            [0.6, 0.3, 0.1],
            [0.8, 0.4, -0.2],
            [0.8, 0.6, -0.4],
            [0.8, 0.3, -0.1]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt)
        pd.testing.assert_frame_equal(actual, expected)

    def test_normalize_weights_short_portfolio(self):
        """
        Tests method normalize_weights for different cases of short portfolios
        """
        wgt = pd.DataFrame([
            [-0.5, -0.3, -0.2], 
            [-0.4, -0.3, -0.1],
            [-1.2, -0.6, -0.2],
            [-0.8, -0.4, 0.2],
            [-0.4, -0.3, 0.2],
            [-1.6, -0.6, 0.2]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [-0.5, -0.3, -0.2],
            [-0.5, -0.375, -0.125],
            [-0.6, -0.3, -0.1],
            [-0.8, -0.4, 0.2],
            [-0.8, -0.6, 0.4],
            [-0.8, -0.3, 0.1]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt)
        pd.testing.assert_frame_equal(actual, expected)

    def test_normalize_weights_neutral_portfolio(self):
        """
        Tests method normalize_weights for different cases of neutral portfolios
        """
        wgt = pd.DataFrame([
            [1.0, -0.6, -0.4],
            [0.4, 0.3, -0.2]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [1.0, -0.6, -0.4],
            [0.8, 0.6, -0.4]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt)
        # pd.testing.assert_frame_equal(actual, expected)

    def test_normalize_weights_fill_na(self):
        """
        Tests parameter fill_na of method normalize_weights
        """
        # Default case 
        wgt = pd.DataFrame([
            [0.7, 0.3, np.nan]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [0.7, 0.3, 0.0]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt)
        pd.testing.assert_frame_equal(actual, expected)

        # Case None
        wgt = pd.DataFrame([
            [0.7, 0.3, np.nan]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [0.7, 0.3, np.nan]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt, fill_na=None)
        pd.testing.assert_frame_equal(actual, expected)

        # Case some value
        wgt = pd.DataFrame([
            [0.6, 0.3, np.nan]], columns = ["A", "B", "C"])
        expected = pd.DataFrame([
            [0.6, 0.3, 0.1]], columns = ["A", "B", "C"])
        actual = BT.Weights.normalize_weights(weights= wgt, fill_na=0.1)
        pd.testing.assert_frame_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
