#%%
"""
Test suite for class BT.Grops

Execute tests in consol with:
    pdm run python -m unittest discover -s tests
    
"""

import unittest
import numpy as np
import pandas as pd
from Backtesting import BT


class TestAssetsHandler(unittest.TestCase):
    """Obvious"""

    def test_rebase_prices(self):
        """Obvious"""
        start_val = 200
        strt = np.array([[100, 200, 300, 400]])
        raw_prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(strt[0]), seed = 1)        
        growth_factor = (1 + raw_prices.pct_change().fillna(0)).cumprod()
        prices = growth_factor.mul(strt)

        expected = start_val * growth_factor 
        actual = BT.AssetsHandler.rebase_prices(prices = prices, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

    def test_single_period_buy_and_hold_portfolio_prices___defaults(self):
        """Obvious"""
        wgts = [[1, 2, 3, 4]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)
        weights = pd.DataFrame(wgts, index = prices.index[[0]], columns=prices.columns)
        
        expected = BT.AssetsHandler.single_period_buy_and_hold_portfolio_prices(
            weights = weights, prices = prices, start_value = 100)
        actual = BT.AssetsHandler.single_period_buy_and_hold_portfolio_prices(
            weights = weights, prices = prices)
        pd.testing.assert_frame_equal(actual, expected)

    def test_single_period_buy_and_hold_portfolio_prices(self):
        """Obvious"""
        start_val = 200
        wgts = [[1, 2, 3, 4]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)
        weights = pd.DataFrame(wgts, index = prices.index[[0]], columns=prices.columns)

        px = prices.values / np.tile(prices.values[0, :], (prices.shape[0], 1))
        norm_wgts = weights.values / weights.values.sum()
        px = px * np.tile(norm_wgts, (prices.shape[0], 1))
        px = start_val * np.sum(px, axis = 1)
        expected = pd.DataFrame(px, index = prices.index, columns=["Portfolio"])
        actual = BT.AssetsHandler.single_period_buy_and_hold_portfolio_prices(
            weights = weights, prices = prices, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

    def test_single_period_buy_and_hold_asset_groups_prices___defaults(self):
        """Obvious"""

        wgts = [[1, 3, 3, 5, 10]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)        
        weights = pd.DataFrame(wgts, index = prices.index[[0]], columns=prices.columns)

        expected = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            prices = prices, weights = weights, groups = None, start_value=100)
        actual = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            prices = prices, weights = weights)
        pd.testing.assert_frame_equal(actual, expected)

    def test_single_period_buy_and_hold_asset_groups_prices(self):
        """Obvious"""
        start_val = 200
        wgts = [[1, 3, 3, 5, 10]]
        grps = [["A", "A", "B", "B", "C"]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)        
        weights = pd.DataFrame(wgts, index = prices.index[[0]], columns=prices.columns)
        
        # Several groups
        groups =  pd.DataFrame(grps, index = prices.index[[0]], columns=prices.columns)
        reb_prices = prices.div(prices.iloc[0, :], axis=1)
        grp_1_prices = (weights["Asset_1"].values * reb_prices["Asset_1"] \
            + weights["Asset_2"].values * reb_prices["Asset_2"]) / weights[["Asset_1", "Asset_2"]].values.sum()
        grp_2_prices = (weights["Asset_3"].values * reb_prices["Asset_3"] \
            + weights["Asset_4"].values * reb_prices["Asset_4"]) / weights[["Asset_3", "Asset_4"]].values.sum()
        grp_3_prices = reb_prices["Asset_5"]

        expected = start_val * pd.concat([grp_1_prices, grp_2_prices, grp_3_prices], axis=1, keys=["A", "B", "C"])
        actual = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            groups = groups, weights = weights, prices = prices, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

        # Single group
        groups = pd.DataFrame("Z", index = prices.index[[0]], columns=prices.columns)
        ptf_prices = BT.AssetsHandler.single_period_buy_and_hold_portfolio_prices(
            weights=weights, prices=prices, start_value=start_val)
        expected = ptf_prices.rename(columns={"Portfolio": "Z"})
        actual = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            groups = groups, weights = weights, prices = prices, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

    def test_buy_and_hold_groups_prices(self):
        """Obvious"""
        start_val = 200
        wgts = [[1, 3, 3, 5, 10], [1, 3, 6, 3, 7], [10, 1, 3, 3, 7]]
        grps = [["A", "A", "B", "B", "C"], ["B", "B", "B", "C", "C"], ["A", "B", "B", "C", "C"]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)
        weights = pd.DataFrame(wgts, index = prices.index[[0, 6, 10]], columns = prices.columns)
        groups =  pd.DataFrame(grps, index = prices.index[[0, 6, 10]], columns = prices.columns)

        grp_prices_1 = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            prices = prices.loc[weights.index[0]:weights.index[1], :], weights = weights.iloc[[0]], groups = groups.iloc[[0]])
        grp_prices_2 = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            prices = prices.loc[weights.index[1]:weights.index[2], :], weights = weights.iloc[[1]], groups = groups.iloc[[1]])
        grp_prices_3 = BT.AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
            prices = prices.loc[weights.index[2]:, :], weights = weights.iloc[[2]], groups = groups.iloc[[2]])
        grp_ret_1 = grp_prices_1.pct_change().fillna(0)
        grp_ret_2 = grp_prices_2.pct_change().fillna(0)
        grp_ret_3 = grp_prices_3.pct_change().fillna(0)
        grt_ret = pd.concat([grp_ret_1, grp_ret_2.iloc[1:, :], grp_ret_3.iloc[1:, :]], axis=0).fillna(0)
        expected = start_val * (1 + grt_ret).cumprod()
        actual = BT.AssetsHandler.buy_and_hold_groups_prices(
            prices = prices, weights = weights, groups = groups, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

    def test_single_period_fixed_weights_groups_prices(self):
        """Obvious"""
        start_val = 200
        wgts = [[1, 3, 3, 5, 10]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)
        weights = pd.DataFrame(wgts, index = prices.index[[0]], columns=prices.columns)

        # Defaults
        expected = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices, weights = weights, groups = None, start_value = 100)
        actual = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices, weights = weights) 
        pd.testing.assert_frame_equal(actual, expected)

        # Single group
        groups =  pd.DataFrame("ABC", index = prices.index[[0]], columns=prices.columns)
        rets = prices.pct_change()
        grp_rets = (wgts[0][0] * rets.iloc[:, [0]].values \
                    + wgts[0][1] * rets.iloc[:, [1]].values \
                    + wgts[0][2] * rets.iloc[:, [2]].values \
                    + wgts[0][3] * rets.iloc[:, [3]].values \
                    + wgts[0][4] * rets.iloc[:, [4]].values) / \
                        (wgts[0][0] + wgts[0][1] + wgts[0][2] + wgts[0][3] + wgts[0][4])
        expected = start_val * (1 + pd.DataFrame(grp_rets, index = prices.index, columns = ["ABC"]).fillna(0)).cumprod()
        actual = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices, weights = weights, groups = groups, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

        # Several groups
        grps = [["A", "A", "B", "B", "C"]]
        groups =  pd.DataFrame(grps, index = prices.index[[0]], columns=prices.columns)
        grp_rets_1 = (wgts[0][0] * rets.iloc[:, [0]].values + wgts[0][1] * rets.iloc[:, [1]].values) / (wgts[0][0] + wgts[0][1])
        grp_rets_2 = (wgts[0][2] * rets.iloc[:, [2]].values + wgts[0][3] * rets.iloc[:, [3]].values) / (wgts[0][2] + wgts[0][3])
        grp_rets_3 = (wgts[0][4] * rets.iloc[:, [4]].values) / (wgts[0][4])
        grp_rets = pd.DataFrame(np.concat((grp_rets_1, grp_rets_2, grp_rets_3), axis=1), \
                     index=prices.index, columns=["A", "B", "C"])
        expected = start_val * (1 + grp_rets.fillna(0)).cumprod()
        actual = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices, weights = weights, groups = groups, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

    def test_fixed_weights_groups_prices(self):
        """Obvious"""
        start_val = 200
        wgts = [[1, 3, 3, 5, 10], [1, 3, 6, 3, 7], [10, 1, 3, 3, 7]]
        grps = [["A", "A", "B", "B", "C"], ["B", "B", "B", "C", "C"], ["A", "B", "B", "C", "C"]]
        prices = BT.Utils.generate_random_prices(n_periods = 12, n_assets = len(wgts[0]), seed = 1)
        weights = pd.DataFrame(wgts, index = prices.index[[0, 6, 10]], columns = prices.columns)
        groups =  pd.DataFrame(grps, index = prices.index[[0, 6, 10]], columns = prices.columns)

        grp_prices_1 = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices.loc[weights.index[0]:weights.index[1], :], weights = weights.iloc[[0]], groups = groups.iloc[[0]])
        grp_prices_2 = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices.loc[weights.index[1]:weights.index[2], :], weights = weights.iloc[[1]], groups = groups.iloc[[1]])
        grp_prices_3 = BT.AssetsHandler.single_period_fixed_weights_groups_prices(
            prices = prices.loc[weights.index[2]:, :], weights = weights.iloc[[2]], groups = groups.iloc[[2]])
        grp_ret_1 = grp_prices_1.pct_change().fillna(0)
        grp_ret_2 = grp_prices_2.pct_change().fillna(0)
        grp_ret_3 = grp_prices_3.pct_change().fillna(0)
        grt_ret = pd.concat([grp_ret_1, grp_ret_2.iloc[1:, :], grp_ret_3.iloc[1:, :]], axis=0).fillna(0)
        expected = start_val * (1 + grt_ret).cumprod()
        actual = BT.AssetsHandler.fixed_weights_groups_prices(
            prices = prices, weights = weights, groups = groups, start_value = start_val)
        pd.testing.assert_frame_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
