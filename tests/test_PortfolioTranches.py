#%%
"""
Test suite for class BT.PortfolioTranches

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestPortfolioTranches(unittest.TestCase):
    """Obvious"""

    def test_Init_Errors(self):
        """Obvious"""
        with self.assertRaises(TypeError) as ctx:
            pt = BT.PortfolioTranches(nb_tranches = 3.0)
        self.assertEqual(str(ctx.exception), "Input 'nb_specs' must be an integer.")

        with self.assertRaises(ValueError) as ctx:
            pt = BT.PortfolioTranches(nb_tranches = -2)
        self.assertEqual(str(ctx.exception), "Input 'nb_specs' must be positive.")

        with self.assertRaises(ValueError) as ctx:
            pt = BT.PortfolioTranches(nb_tranches = 3, reset_period = 4)
        self.assertEqual(str(ctx.exception), "Input 'nb_tranches' must be either 'None', '1' or equal to 'nb_tranches'.")

        with self.assertRaises(ValueError) as ctx:
            pt = BT.PortfolioTranches(nb_tranches = 3, reset_period = 1, reset_type = "dummy")
        self.assertEqual(str(ctx.exception), "Input 'reset_type' must be either 'None' or 'equally-weighted'.")

    def test_Init_Warnings(self):
        """Obvious"""
        with self.assertWarns(UserWarning) as cm:
            BT.PortfolioTranches(nb_tranches = 1, reset_period = 1)
        self.assertEqual(str(cm.warning), "Discarding input 'reset_period' and setting to 'None'.")

        with self.assertWarns(UserWarning) as cm:
            BT.PortfolioTranches(nb_tranches = 1, reset_period = None, reset_type="equally-weighted")
        self.assertEqual(str(cm.warning), "Discarding input 'reset_type' and setting to 'None'.")

    def test_Init_Defaults(self):
        """Obvious"""
        actual = BT.PortfolioTranches()
        expected = BT.PortfolioTranches(nb_tranches = 1, reset_period = None, reset_type = None )
        self.assertEqual(actual.nb_tranches, expected.nb_tranches)
        self.assertEqual(actual.reset_period, expected.reset_period)
        self.assertEqual(actual.reset_type, expected.reset_type)

    def test_Init(self):
        """Obvious"""
        actual = BT.PortfolioTranches(nb_tranches = 1, reset_period = None, reset_type = None)
        self.assertEqual(actual.nb_tranches, 1)
        self.assertIsNone(actual.reset_period)
        self.assertIsNone(actual.reset_type)

        actual = BT.PortfolioTranches(nb_tranches = 3, reset_period = 1, reset_type = "equally-weighted")
        self.assertEqual(actual.nb_tranches, 3)
        self.assertEqual(actual.reset_period, 1)
        self.assertEqual(actual.reset_type, "equally-weighted")

    def test_add_model_portfolio___Errors(self):
        """Obvious"""

        # Input model_ptf not a dataframe
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(TypeError) as ctx:
            pt.add_model_portfolio(model_ptf = "dummy")
        self.assertEqual(str(ctx.exception), "Input 'model_ptf' must be a 'pd.DataFrame'.")

        # Index of model_ptf not a pd.datetimeIndex
        model_ptf = pd.DataFrame(index = ["A", "B"], columns = ["X", "Y"])
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be a 'pd.DatetimeIndex'.")

        # Index of model_ptf not unique
        inds = pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"])
        cols = ["A", "B", "C"]
        model_ptf = pd.DataFrame(index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be unique.")

        # Columns of model_ptf not unique
        inds = pd.to_datetime(["2025-01-31"])
        cols = ["A", "B", "A", "B"]
        model_ptf = pd.DataFrame(index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Columns of input 'model_ptf' must be unique.")

        # Trying to overwrite model portfolio for existing dates
        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_model_portfolio(model_ptf = model_ptf)
        with self.assertRaises(ValueError) as ctx:
            pt.add_model_portfolio(model_ptf = model_ptf.iloc[[1]])
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be posterior to existing model portfolios.")

    def test_add_model_portfolio(self):
        """Obvious"""

        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)

        # Adding first model_portfolios
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_model_portfolio(model_ptf = model_ptf)
        expected = model_ptf
        actual = pt.model_ptf
        pd.testing.assert_frame_equal(actual, expected)

        # Adding several model_portfolios
        model_ptf_1 = model_ptf.iloc[[0, 1], 0:3]
        model_ptf_2 = model_ptf.iloc[[2], 1:5]
        expected = pd.concat([model_ptf_1, model_ptf_2], axis=0).fillna(0)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_model_portfolio(model_ptf = model_ptf_1)
        pt.add_model_portfolio(model_ptf = model_ptf_2)
        actual = pt.model_ptf
        pd.testing.assert_frame_equal(actual, expected)

    def test_add_asset_prices___Errors(self):
        """Obvious"""

        # Input asset_prices not a dataframe
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(TypeError) as ctx:
            pt.add_asset_prices(asset_prices = "dummy")
        self.assertEqual(str(ctx.exception), "Input 'asset_prices' must be a 'pd.DataFrame'.")

        # Index of asset_prices not a pd.datetimeIndex
        asset_prices = pd.DataFrame(index = ["A", "B"], columns = ["X", "Y"])
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_asset_prices(asset_prices = asset_prices)
        self.assertEqual(str(ctx.exception), "Index of input 'asset_prices' must be a 'pd.DatetimeIndex'.")

        # Index of asset_prices not unique
        inds = pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"])
        cols = ["A", "B", "C"]
        asset_prices = pd.DataFrame(index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_asset_prices(asset_prices = asset_prices)
        self.assertEqual(str(ctx.exception), "Index of input 'asset_prices' must be unique.")

        # Columns of asset_prices not unique
        inds = pd.to_datetime(["2025-01-31"])
        cols = ["A", "B", "A", "B"]
        asset_prices = pd.DataFrame(index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        with self.assertRaises(ValueError) as ctx:
            pt.add_asset_prices(asset_prices = asset_prices)
        self.assertEqual(str(ctx.exception), "Columns of input 'asset_prices' must be unique.")

        # Trying provide asset prices that don't bind to existing asset prices
        px = [[100, 200, 300, 400], [101, 201, 301, 401], [102, 202, 302, 402], [103, 203, 303, 403]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31"])
        cols = ["A", "B", "C", "D"]
        asset_prices = pd.DataFrame(px, index = inds, columns = cols)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_asset_prices(asset_prices = asset_prices.iloc[[0, 1]])
        with self.assertRaises(ValueError) as ctx:
            pt.add_asset_prices(asset_prices = asset_prices.iloc[[2, 3]])
        self.assertEqual(str(ctx.exception), "Input 'model_ptf' must complement existing available data.")

    def test_add_asset_prices(self):
        """Obvious"""

        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        asset_prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104],
             "B": [200, 204, 202, 208, 206],
             "C": [300, 303, 309, 318, 330],
             "D": [400, 404, 400, 404, 400]}, index = inds)
        

        # Adding first model_portfolios
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_asset_prices(asset_prices = asset_prices)
        expected = pd.DataFrame(
            {"A": [1.0, 1.01, 1.02, 1.03, 1.04],
             "B": [1.0, 1.02, 1.01, 1.04, 1.03],
             "C": [1.0, 1.01, 1.03, 1.06, 1.10],
             "D": [1.0, 1.01, 1.0, 1.01, 1.0]}, index = inds)
        actual = pt.asset_growth_factor
        pd.testing.assert_frame_equal(actual, expected)

        # Adding several model_portfolios
        asset_prices_1 = asset_prices.iloc[0:3, 0:3]
        asset_prices_2 = asset_prices.iloc[2:5, 1:5]
        expected = pd.DataFrame(
            {"A": [1.0, 1.01, 1.02, 1.02, 1.02],
             "B": [1.0, 1.02, 1.01, 1.04, 1.03],
             "C": [1.0, 1.01, 1.03, 1.06, 1.10],
             "D": [1.0, 1.0, 1.0, 1.01, 1.0]}, index = inds)
        pt = BT.PortfolioTranches(nb_tranches = 1)
        pt.add_asset_prices(asset_prices = asset_prices_1)
        pt.add_asset_prices(asset_prices = asset_prices_2)
        actual = pt.asset_growth_factor
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
