#%%
"""
Test suite for class BT.ModelPortfolio

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT

class TestModelPortfolio(unittest.TestCase):
    """Obvious"""

    def test_add_model_portfolio___Errors(self):
        """Obvious"""

        # Input model_ptf not a dataframe
        mp = BT.ModelPortfolio()
        with self.assertRaises(TypeError) as ctx:
            mp.add_model_portfolio(model_ptf = "dummy")
        self.assertEqual(str(ctx.exception), "Input 'model_ptf' must be a 'pd.DataFrame'.")

        # Index of model_ptf not a pd.datetimeIndex
        model_ptf = pd.DataFrame(index = ["A", "B"], columns = ["X", "Y"])
        mp = BT.ModelPortfolio()
        with self.assertRaises(ValueError) as ctx:
            mp.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be a 'pd.DatetimeIndex'.")

        # Index of model_ptf not unique
        inds = pd.to_datetime(["2025-01-31", "2025-01-31", "2025-02-28"])
        cols = ["A", "B", "C"]
        model_ptf = pd.DataFrame(index = inds, columns = cols)
        mp = BT.ModelPortfolio()
        with self.assertRaises(ValueError) as ctx:
            mp.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be unique.")

        # Columns of model_ptf not unique
        inds = pd.to_datetime(["2025-01-31"])
        cols = ["A", "B", "A", "B"]
        model_ptf = pd.DataFrame(index = inds, columns = cols)
        mp = BT.ModelPortfolio()
        with self.assertRaises(ValueError) as ctx:
            mp.add_model_portfolio(model_ptf = model_ptf)
        self.assertEqual(str(ctx.exception), "Columns of input 'model_ptf' must be unique.")

        # Trying to overwrite model portfolio for existing dates
        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)
        mp = BT.ModelPortfolio()
        mp.add_model_portfolio(model_ptf = model_ptf)
        with self.assertRaises(ValueError) as ctx:
            mp.add_model_portfolio(model_ptf = model_ptf.iloc[[1]])
        self.assertEqual(str(ctx.exception), "Index of input 'model_ptf' must be posterior to existing model portfolios.")

    def test_add_model_portfolio(self):
        """Obvious"""

        wgts = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2]]
        inds = pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"])
        cols = ["A", "B", "C", "D"]
        model_ptf = pd.DataFrame(wgts, index = inds, columns = cols)

        # Adding first model_portfolios
        mp = BT.ModelPortfolio()
        mp.add_model_portfolio(model_ptf = model_ptf)
        expected = model_ptf
        actual = mp.model_ptf
        pd.testing.assert_frame_equal(actual, expected)

        # Adding several model_portfolios
        model_ptf_1 = model_ptf.iloc[[0, 1], 0:3]
        model_ptf_2 = model_ptf.iloc[[2], 1:5]
        expected = pd.concat([model_ptf_1, model_ptf_2], axis=0).fillna(0)
        mp = BT.ModelPortfolio()
        mp.add_model_portfolio(model_ptf = model_ptf_1)
        mp.add_model_portfolio(model_ptf = model_ptf_2)
        actual = mp.model_ptf
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
