#%%
"""
Test suite for class BT.DailyPrices

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestDailyPrices(unittest.TestCase):
    """Obvious"""

    def test_add_prices___Errors(self):
        """Obvious"""

        # Input prices not a dataframe
        dp = BT.DailyPrices()
        with self.assertRaises(TypeError) as ctx:
            dp.add_prices(prices = "dummy")
        self.assertEqual(str(ctx.exception), "Input 'prices' must be a 'pd.DataFrame'.")

        # Index of prices not a pd.datetimeIndex
        prices = pd.DataFrame(index = ["A", "B"], columns = ["X", "Y"])
        dp = BT.DailyPrices()
        with self.assertRaises(ValueError) as ctx:
            dp.add_prices(prices = prices)
        self.assertEqual(str(ctx.exception), "Index of input 'prices' must be a 'pd.DatetimeIndex'.")

        # Index of prices not unique
        inds = pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"])
        cols = ["A", "B", "C"]
        prices = pd.DataFrame(index = inds, columns = cols)
        dp = BT.DailyPrices()
        with self.assertRaises(ValueError) as ctx:
            dp.add_prices(prices = prices)
        self.assertEqual(str(ctx.exception), "Index of input 'prices' must be unique.")

        # Columns of prices not unique
        inds = pd.to_datetime(["2025-01-31"])
        cols = ["A", "B", "A", "B"]
        prices = pd.DataFrame(index = inds, columns = cols)
        dp = BT.DailyPrices()
        with self.assertRaises(ValueError) as ctx:
            dp.add_prices(prices = prices)
        self.assertEqual(str(ctx.exception), "Columns of input 'prices' must be unique.")

        # Trying provide prices for dates that don't bind to existing prices
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        prices = pd.DataFrame({
            "A": [100, 101, 102, 103, 104], 
            "B": [200, 201, 202, 203, 204], 
            "C": [300, 301, 302, 303, 304], 
            "D": [400, 401, 402, 403, 404]
        }, index = inds)
        dp = BT.DailyPrices()
        dp.add_prices(prices = prices.iloc[[0, 1],:])
        with self.assertRaises(ValueError) as ctx:
            dp.add_prices(prices = prices.iloc[[2, 3]])
        self.assertEqual(str(ctx.exception), "First date of input 'prices' must match with last date of existing prices.")

        # Trying provide prices for dates that don't bind to existing prices
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        prices = pd.DataFrame({
            "A": [100, 101, 102, 103, 104], 
            "B": [200, 201, 202, 203, 204], 
            "C": [300, 301, 302, 303, 304], 
            "D": [400, 401, 402, 403, 404]
        }, index = inds)
        dp = BT.DailyPrices()
        dp.add_prices(prices = prices.iloc[[0, 1],0:3])
        prices.iloc[1, 2] = 303
        with self.assertRaises(ValueError) as ctx:
            dp.add_prices(prices = prices.iloc[[1, 2, 3], 1:3])
        self.assertEqual(str(ctx.exception), "Start values of input 'prices' must match with last values of existing prices.")

    def test_add_asset_prices(self):
        """Obvious"""

        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104],
             "B": [200, 204, 202, 208, 206],
             "C": [300, 303, 309, 318, 330],
             "D": [400, 404, 400, 404, 400]}, index = inds)

        # Adding first model_portfolios
        pt = BT.DailyPrices()
        pt.add_prices(prices = prices)
        expected = prices
        actual = pt.prices
        pd.testing.assert_frame_equal(actual, expected)

        # Adding several model_portfolios
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"])
        prices_1 = pd.DataFrame(
            {"A": [100, 101, 102],
             "B": [200, 204, 202],
             "C": [300, 303, 309]}, index = inds[0:3])
        prices_2 = pd.DataFrame(
            {"B": [202, 208, 206],
             "C": [309, 318, 330],
             "D": [400, 404, 400]}, index = inds[2:])
        expected = pd.DataFrame(
            {"A": [100, 101, 102, np.nan, np.nan],
             "B": [200, 204, 202, 208, 206],
             "C": [300, 303, 309, 318, 330],
             "D": [np.nan, np.nan, 400, 404, 400]}, index = inds)
        pt = BT.DailyPrices()
        pt.add_prices(prices = prices_1)
        pt.add_prices(prices = prices_2)
        actual = pt.prices
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
