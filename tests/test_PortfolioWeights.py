#%%
"""
Test suite for class BT.PortfolioWeights

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestPortfolioWeights(unittest.TestCase):
    """Obvious"""

    def test_init(self):
        """Obvious"""
        pw = BT.PortfolioWeights()
        self.assertTrue(isinstance(pw._model_ptf, BT.ModelPortfolio))
        self.assertIsNone(pw.model_ptf)
        self.assertIsNone(pw.eod_weights)
        self.assertIsNone(pw.close_weights)

    def test_add_model_portfolio(self):
        """Obvious"""
        
        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)

        # Adding several model_portfolios
        model_ptf_1 = model_ptf.iloc[[0, 1], 0:3]
        model_ptf_2 = model_ptf.iloc[[2], 1:5]
        expected = pd.concat([model_ptf_1, model_ptf_2], axis=0).fillna(0)
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.add_model_portfolio(model_ptf = model_ptf_2)
        actual = pw.model_ptf
        pd.testing.assert_frame_equal(actual, expected)

    def test_get_rebalancing(self):
        """Obvious"""
        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)
        expected = model_ptf
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf)
        actual = pw.model_ptf
        pd.testing.assert_frame_equal(actual, expected)

    def test_internal_compute_ptf_weights___Errors(self):
        """Obvious"""
        
        model_ptf = pd.DataFrame({
            "A":[0.1, 0.2, 0.3, 0.4],
            "B":[0.2, 0.3, 0.4, 0.1],
            "C":[0.3, 0.4, 0.1, 0.2],
            "D":[0.4, 0.1, 0.2, 0.3]}, 
            index = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"]))
        close_prices = pd.DataFrame({
            "A":[100, 101, 102, 103],
            "B":[200, 204, 202, 206],
            "C":[300, 303, 309, 312],
            "D":[400, 404, 400, 408]}, 
            index = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"]))


        # Index of asset_prices not unique
        close_px = pd.DataFrame(index = pd.to_datetime(["2025-12-31", "2025-12-31", "2026-01-31"]),
                                    columns = ["A", "B", "C"])
        with self.assertRaises(ValueError) as ctx:
            BT.PortfolioWeights()._internal_compute_ptf_weights(
                model_ptf = model_ptf, close_prices = close_px)
        self.assertEqual(str(ctx.exception), "Index of input 'close_prices' must be unique.")

        # Columns of asset_prices not unique
        close_px = pd.DataFrame(index = pd.to_datetime(["2025-01-31"]),
                                    columns = ["A", "B", "A", "D"])
        with self.assertRaises(ValueError) as ctx:
            BT.PortfolioWeights()._internal_compute_ptf_weights(
                model_ptf = model_ptf, close_prices = close_px)
        self.assertEqual(str(ctx.exception), "Columns of input 'close_prices' must be unique.")

        # Index of asset_prices must cover index of model portfolio
        close_px = close_prices.set_index(pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-04-30"]))
        with self.assertRaises(ValueError) as ctx:
            BT.PortfolioWeights()._internal_compute_ptf_weights(
                model_ptf = model_ptf, close_prices = close_px)
        self.assertEqual(str(ctx.exception), "Index of input 'close_prices' must contain all elements of index of 'model_ptf'.")

        # Columns of asset_prices must cover index of model portfolio
        close_px = close_prices.copy()
        close_px.columns = ["A", "B", "C", "Z"]
        with self.assertRaises(ValueError) as ctx:
            BT.PortfolioWeights()._internal_compute_ptf_weights(
                model_ptf = model_ptf, close_prices = close_px)
        self.assertEqual(str(ctx.exception), "Columns of input 'close_prices' must contain all elements of columns of 'model_ptf'.")

    def test_internal_compute_ptf_weights(self):
        """Obvious"""
        model_ptf = pd.DataFrame({
            "A":[0.6, 0.0, 0.4],
            "B":[0.4, 0.6, 0.0],
            "C":[0.0, 0.4, 0.6]}, 
            index = pd.to_datetime(["2025-12-31", "2026-02-28", "2026-04-30"]))
        close_prices = pd.DataFrame({
            "A":[100, 101, 102, 103, 104, 105, 106, 107],
            "B":[200, 204, 202, 206, 204, 208, 206, 210],
            "C":[300, 303, 309, 318, 330, 345, 363, 384]}, 
            index = pd.to_datetime(["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
                                    "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]))

        s_dt = "2025-12-31"
        e_dt = "2026-02-28"
        mpf = model_ptf.loc[s_dt]
        rets = close_prices.loc[s_dt:e_dt, :].pct_change().fillna(0)
        alloc = 100 * (1 + rets).cumprod().mul(mpf)
        wgts_1 = alloc.div(alloc.sum(axis=1), axis=0)

        s_dt = "2026-02-28"
        e_dt = "2026-04-30"
        mpf = model_ptf.loc[s_dt]
        rets = close_prices.loc[s_dt:e_dt, :].pct_change().fillna(0)
        alloc = 100 * (1 + rets).cumprod().mul(mpf)
        wgts_2 = alloc.div(alloc.sum(axis=1), axis=0)

        s_dt = "2026-04-30"
        e_dt = "2026-06-30"
        mpf = model_ptf.loc[s_dt]
        rets = close_prices.loc[s_dt:e_dt, :].pct_change().fillna(0)
        alloc = 100 * (1 + rets).cumprod().mul(mpf)
        wgts_3 = alloc.div(alloc.sum(axis=1), axis=0)

        # Price index not chronological
        res = BT.PortfolioWeights._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices)
        expected_close = res["close_weights"]
        expected_eod = res["eod_weights"]
        res = BT.PortfolioWeights()._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices.iloc[[1, 2, 3, 4, 5, 6, 7, 0], :])
        actual_close = res["close_weights"]
        actual_eod = res["eod_weights"]
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

        # Price columns not in same order than model portfolio
        res = BT.PortfolioWeights._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices)
        expected_close = res["close_weights"]
        expected_eod = res["eod_weights"]
        res = BT.PortfolioWeights()._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices.iloc[:, [1, 2, 0]])
        actual_close = res["close_weights"]
        actual_eod = res["eod_weights"]
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)


        # Rebalancing NOT on last day
        expected_close = pd.concat([wgts_1, wgts_2[1:], wgts_3[1:]], axis = 0)
        expected_close.iloc[0] = np.nan
        expected_eod = pd.concat([wgts_1[:-1], wgts_2[:-1], wgts_3], axis = 0)
        expected_eod.iloc[-1] = np.nan        
        pw = BT.PortfolioWeights()
        res = pw._internal_compute_ptf_weights(model_ptf = model_ptf, close_prices = close_prices)
        actual_close = res["close_weights"]
        actual_eod = res["eod_weights"]
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

        # Rebalancing ON last day
        expected_close = expected_close["2025-12-31":"2026-04-30"]
        expected_eod = expected_eod["2025-12-31":"2026-04-30"]

        pw = BT.PortfolioWeights()
        res = pw._internal_compute_ptf_weights(model_ptf = model_ptf, close_prices = close_prices[:"2026-04-30"])
        actual_close = res["close_weights"]
        actual_eod = res["eod_weights"]
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

    def test_compute_ptf_weights___Errors(self):
        """Obvious"""

        model_ptf = pd.DataFrame({
            "A":[0.6, 0.0, 0.4],
            "B":[0.4, 0.6, 0.0],
            "C":[0.0, 0.4, 0.6]}, 
            index = pd.to_datetime(["2025-12-31", "2026-02-28", "2026-04-30"]))
        close_prices = pd.DataFrame({
            "A":[100, 101, 102, 103, 104, 105, 106, 107],
            "B":[200, 204, 202, 206, 204, 208, 206, 210],
            "C":[300, 303, 309, 318, 330, 345, 363, 384]}, 
            index = pd.to_datetime(["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
                                    "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]))

        # Close prices start after last available portfolio weight
        model_ptf_1 = model_ptf.loc["2025-12-31":"2026-02-28"]
        close_prices_1 = close_prices["2025-12-31":"2026-02-28"]
        model_ptf_2 = model_ptf.loc[["2026-04-30"]]
        close_prices_2 = close_prices["2026-05-31":]
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.compute_ptf_weights(close_prices = close_prices_1)
        pw.add_model_portfolio(model_ptf = model_ptf_2)
        with self.assertRaises(ValueError) as ctx:
            pw.compute_ptf_weights(close_prices = close_prices_2)
        self.assertEqual(str(ctx.exception), 
                         "Input 'close_prices' must start at latest from last available "
                         "portfolio weights.")
        model_ptf_1 = model_ptf.loc[["2026-02-28"]]
        close_prices_1 = close_prices["2025-12-31":"2026-02-28"]
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.compute_ptf_weights(close_prices = close_prices_1)
        pw.close_weights
        pw.eod_weights

    def test_compute_ptf_weights(self):
        """Obvious"""
        model_ptf = pd.DataFrame({
            "A":[0.6, 0.0, 0.4],
            "B":[0.4, 0.6, 0.0],
            "C":[0.0, 0.4, 0.6]}, 
            index = pd.to_datetime(["2025-12-31", "2026-02-28", "2026-04-30"]))
        close_prices = pd.DataFrame({
            "A":[100, 101, 102, 103, 104, 105, 106, 107],
            "B":[200, 204, 202, 206, 204, 208, 206, 210],
            "C":[300, 303, 309, 318, 330, 345, 363, 384]}, 
            index = pd.to_datetime(["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
                                    "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]))

        # Only one rebalancing and prices end on rebalancing date
        model_ptf_1 = model_ptf.loc[["2025-12-31"]]
        close_prices_1 = close_prices["2025-11-30":"2025-12-31"]
        expected_close = pd.DataFrame(index = model_ptf_1.index, columns = model_ptf_1.columns)
        expected_eod = model_ptf_1.copy()
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.compute_ptf_weights(close_prices = close_prices_1)
        actual_close = pw.close_weights
        actual_eod = pw.eod_weights
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

        # Only one close/eod weights computation
        res = BT.PortfolioWeights._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices)
        expected_close = res["close_weights"]
        expected_eod = res["eod_weights"]
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf)
        pw.compute_ptf_weights(close_prices = close_prices)
        actual_close = pw.close_weights
        actual_eod = pw.eod_weights
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

        # Several close/eod weights computation, cut on rebalancing date
        res = BT.PortfolioWeights._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices)
        expected_close = res["close_weights"]
        expected_eod = res["eod_weights"]
        model_ptf_1 = model_ptf.loc["2025-12-31":"2026-02-28"]
        close_prices_1 = close_prices["2025-12-31":"2026-02-28"]
        model_ptf_2 = model_ptf.loc[["2026-04-30"]]
        close_prices_2 = close_prices["2026-02-28":]
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.compute_ptf_weights(close_prices = close_prices_1)
        pw.add_model_portfolio(model_ptf = model_ptf_2)
        pw.compute_ptf_weights(close_prices = close_prices_2)
        actual_close = pw.close_weights
        actual_eod = pw.eod_weights
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

        # Several close/eod weights computation, cut NOT on rebalancing date
        res = BT.PortfolioWeights._internal_compute_ptf_weights(
            model_ptf = model_ptf, close_prices = close_prices)
        expected_close = res["close_weights"]
        expected_eod = res["eod_weights"]
        model_ptf_1 = model_ptf.loc["2025-12-31":"2026-02-28"]
        close_prices_1 = close_prices["2025-12-31":"2026-03-31"]
        model_ptf_2 = model_ptf.loc[["2026-04-30"]]
        close_prices_2 = close_prices["2026-03-31":]
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf_1)
        pw.compute_ptf_weights(close_prices = close_prices_1)
        pw.add_model_portfolio(model_ptf = model_ptf_2)
        pw.compute_ptf_weights(close_prices = close_prices_2)
        actual_close = pw.close_weights
        actual_eod = pw.eod_weights
        pd.testing.assert_frame_equal(actual_close, expected_close)
        pd.testing.assert_frame_equal(actual_eod, expected_eod)

    def test_get_turnover(self):
        """Obvious"""
        model_ptf = pd.DataFrame({
            "A":[0.6, 0.0, 0.4],
            "B":[0.4, 0.6, 0.0],
            "C":[0.0, 0.4, 0.6]}, 
            index = pd.to_datetime(["2025-12-31", "2026-02-28", "2026-04-30"]))
        close_prices = pd.DataFrame({
            "A":[100, 101, 102, 103, 104, 105, 106, 107],
            "B":[200, 204, 202, 206, 204, 208, 206, 210],
            "C":[300, 303, 309, 318, 330, 345, 363, 384]}, 
            index = pd.to_datetime(["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
                                    "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]))

        # Default values
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf)
        pw.compute_ptf_weights(close_prices = close_prices)
        expected = pw.get_turnover()
        actual = pw.get_turnover(rebalancings_only = False)
        pd.testing.assert_frame_equal(actual, expected)

        # No turnover to compute
        pw = BT.PortfolioWeights()
        actual = pw.get_turnover()
        self.assertIsNone(actual)

        # NOT restrited to rebalancings
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf)
        pw.compute_ptf_weights(close_prices = close_prices)
        expected = pw.eod_weights - pw.close_weights
        actual = pw.get_turnover(rebalancings_only = False)
        pd.testing.assert_frame_equal(actual, expected)

        # Restrited to rebalancings
        pw = BT.PortfolioWeights()
        pw.add_model_portfolio(model_ptf = model_ptf)
        pw.compute_ptf_weights(close_prices = close_prices)
        expected = pw.eod_weights - pw.close_weights
        expected = expected.loc[model_ptf.index]
        actual = pw.get_turnover(rebalancings_only = True)
        pd.testing.assert_frame_equal(actual, expected)

    # TODO: move to other class?
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
        actual = BT.PortfolioWeights.equal_weights(prices = prices)
        pd.testing.assert_frame_equal(actual, expected)

    # TODO: move to other class?
    def test_drifting_weights(self):
        """Obvious"""
        wgt_0 = [[0.1, 0.2, 0.3, 0.4, 0.0]]
        prices = BT.Utils.generate_random_prices(n_periods=6, n_assets=len(wgt_0[0]), seed=1)
        start_weights = pd.DataFrame(wgt_0, index=[prices.index[0]], columns=prices.columns)
        notional = 1

        allocation = notional * prices.div(prices.values[0, :], axis=1).mul(start_weights.values, axis=1)
        expected = allocation.div(allocation.sum(axis=1), axis=0)
        actual = BT.PortfolioWeights.drifting_weights(start_weights = start_weights, prices = prices)
        pd.testing.assert_frame_equal(actual, expected)


        
if __name__ == "__main__":
    unittest.main()

# %%
