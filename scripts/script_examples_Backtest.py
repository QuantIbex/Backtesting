
#%%


# %% Initialization

import numpy as np
import pandas as pd
from Backtesting import BT


# %% Functions to generate toy data and specs
def _generate_toy_data():
    sctr = np.concatenate([
        np.tile(["A"], (25, 5)),
        np.tile(["B"], (25, 5)),
        np.tile(["C"], (25, 5)),
        np.tile(["D"], (25, 5)),
        np.tile(["E"], (25, 5))], axis=1)
    cntry = np.concatenate([
        np.tile(["Germany"], (25, 4)),
        np.tile(["France"], (25, 6)),
        np.tile(["Italy"], (25, 4)),
        np.tile(["Spain"], (25, 6)),
        np.tile(["Portugal"], (25, 5))], axis=1)

    bench_wgts =  np.random.rand(cntry.shape[0], cntry.shape[1])        # uniform in [0, 1]
    bench_wgts /= bench_wgts.sum(axis=1, keepdims=True)                 # normalize rows to sum to 1.0

    asset_prices = BT.Utils.generate_random_prices(n_periods = sctr.shape[0] + 12 - 1, n_assets = sctr.shape[1], seed = 1).round()
    asset_prices.iloc[:6, 2] = np.nan
    asset_sectors = pd.DataFrame(sctr, index = asset_prices.index[-sctr.shape[0]:], columns = asset_prices.columns)
    asset_countries = pd.DataFrame(cntry, index = asset_prices.index[-cntry.shape[0]:], columns = asset_prices.columns)
    bench_weights = pd.DataFrame(bench_wgts, index = asset_prices.index[-bench_wgts.shape[0]:], columns = asset_prices.columns)

    return {
        "asset_prices": asset_prices, 
        "asset_sectors": asset_sectors, 
        "asset_countries": asset_countries,
        "bench_weights": bench_weights
    }

def _generate_toy_specs():

    # TODO:
    #  - include trading fees (and taxes) table
    #  - include management fees
    #  - include liquidity filters
    #  - 
    #  - 

    product_specs = {
        "investment": 100e6,
        "period": ["2020-12-31", "2022-12-31"]
        }

    strategy_specs = {
        "strategies": {
            "Indexing":{
                "rebalancing": {
                    "frequency": "monthly",
                    "reference": "end-of-period"
                    },
                "model_portfolio": "bench_weights",
                "strat_name": ""
                },
            "Group_momentum": {
                "rebalancing": {
                    "frequency": "monthly", 
                    "reference": "end-of-period"
                    },
                "tranches": {
                    "number": 3,
                    "reset_frequency": "each",
                    "reset_type": "equally-weighted"
                    }
                }
        },
        "aggregation": {
            "type": "weighted"
        }
        }

    asset_groups = {
        "group_labels_sector": {
            "type": "labels",
            "var_name": "asset_sectors",
            "res_name_asset_group_labels": "sector_groups_labels",
            "res_name_asset_group_prices": "sector_groups_prices"},
        "group_labels_country": {
            "type": "labels",
            "var_name": "asset_countries",  
            "res_name_asset_group_labels": "country_groups_labels",
            "res_name_asset_group_prices": "country_groups_prices"},
        }

    group_metrics = {
        "mom_6_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "sector_groups_prices",
                "lookback": 6,
                "skip": 0},
            "res_name": "group_metric_mom_6_0"
            },
        "mom_12_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "sector_groups_prices",
                "lookback": 12,
                "skip": 0},
            "res_name": "group_metric_mom_12_0"
        }
    }

    group_ratings = {
        "uscore_6-12_0": {
            "ratings": [
                    {"var_name": "group_metric_mom_6_0", "type": "uscore", "scaling": "n+1"},
                    {"var_name": "group_metric_mom_12_0", "type": "uscore", "scaling": "n+1"}
                ], 
            "aggregate": {"method": "mean"},
            "res_name": "group_rating_uscore_6-12_0"
            }

    }

    group_signals = {
        "top-bottom_2-2":{
            "signals":{"var_name": "group_rating_uscore_6-12_0", "type": "top-bottom", "top": 2, "bottom": 2},
            "res_name": "group_signal_top-bottom_2-2"
        }
    }

    group_weightings = {
        "ew":{
            "weightings": {"var_name": "group_signal_top-bottom_2-2", "type": "equally-weighted"},
            "res_name": "group_weighting_ew"
        }
    }


    if False:
        asset_metrics = {
            "asset_mom_6_0": {"var_name": "asset_prices", "type": "momentum", "lookback": 6, "skip": 0},
            "asset_mom_6_1": {"var_name": "asset_prices", "type": "momentum", "lookback": 6, "skip": 1},
            "asset_mom_12_0": {"var_name": "asset_prices", "type": "momentum", "lookback": 12, "skip": 0},
            "asset_mom_12_1": {"var_name": "asset_prices", "type": "momentum", "lookback": 12, "skip": 1},
            "asset_mom_6-12_0" :{
                "metrics": [
                    {"var_name": "asset_prices", "type": "momentum", "lookback": 6, "skip": 0}, 
                    {"var_name": "asset_prices", "type": "momentum", "lookback": 12, "skip": 0}], 
                "aggregate": {"method": "mean", "weights": [1, 2]}},
            "asset_mom_6-12_1" :{
                "metrics": [
                    {"var_name": "asset_prices", "type": "momentum", "lookback": 6, "skip": 1}, 
                    {"var_name": "asset_prices", "type": "momentum", "lookback": 12, "skip": 1}], 
                "aggregate": {"method": "mean", "weights": [1, 2]}}
            }

    return {
        "product_specs": product_specs,
        "strategy_specs": strategy_specs,
        "asset_groups": asset_groups,
        "group_metrics": group_metrics,
        "group_ratings": group_ratings,
        "group_signals": group_signals,
        "group_weightings": group_weightings
    }

data = _generate_toy_data()
specs = _generate_toy_specs()


# %% BACKTESTING PROCEDURE


start_date = specs["product_specs"]["period"][0]
end_date = specs["product_specs"]["period"][1]
trading_days = data["asset_prices"].index.sort_values()
mask = (start_date <= trading_days) & (trading_days <= end_date)
trading_days = trading_days[mask]

# datastore = {}

# Iterate on trading days
for ii, ii_dt in enumerate(trading_days):
    # ii = 0
    # ii_dt = trading_days[ii]
    print(ii, ii_dt)

    ii_datastore = {"iteration": ii, "date": ii_dt}

    # Extract available data
    ii_prices = data["asset_prices"].loc[:ii_dt, :]
    ii_asset_sectors = data["asset_sectors"].loc[[ii_dt], :]
    ii_asset_countries = data["asset_countries"].loc[[ii_dt], :]
    ii_bench_weights = data["bench_weights"].loc[[ii_dt], :]

    # Apply filters
    # TODO: add filter on available prices (or not)?

    ii_datastore["prices"] = ii_prices
    ii_datastore["asset_sectors"] = ii_asset_sectors
    ii_datastore["asset_countries"] = ii_asset_countries
    ii_datastore["bench_weights"] = ii_bench_weights

    # Form groups
    print("Forming groups...", end = " ")
    for jj_specs in specs["asset_groups"].values():
        # jj = 1
        # jj_specs = list(specs["asset_groups"].values())[jj]

        jj_asset_groups_labels = BT.Groups.compute(data = ii_datastore, specs = jj_specs)
        jj_g_labls = jj_asset_groups_labels.rename({ii_dt: ii_prices.index[0]})
        jj_b_wgts = ii_bench_weights.rename({ii_dt: ii_prices.index[0]})

        jj_asset_groups_prices = BT.AssetsHandler.fixed_weights_groups_prices(
            prices = ii_prices, weights = jj_b_wgts, groups = jj_g_labls)
        
        ii_datastore[jj_specs["res_name_asset_group_labels"]] = jj_asset_groups_labels
        ii_datastore[jj_specs["res_name_asset_group_prices"]] = jj_asset_groups_prices
    print("Done.")

    # Compute group metrics
    print("Computing metrics...", end = " ")
    for jj_specs in specs["group_metrics"].values():
        # jj = 1
        # jj_specs = list(specs["group_metrics"].values())[jj]

        jj_res = BT.Metrics.compute(specs = jj_specs, data = ii_datastore)
        ii_datastore[jj_specs["res_name"]] = jj_res["global"]
    print("Done.")

    # Compute group ratings
    print("Computing ratings...", end = " ")
    for jj_specs in specs["group_ratings"].values():
        # jj = 0
        # jj_specs = list(specs["group_ratings"].values())[jj]

        jj_res = BT.Ratings.compute(specs = jj_specs, data = ii_datastore)
        ii_datastore[jj_specs["res_name"]] = jj_res["global"]
    print("Done.")

    # Compute group signals
    print("Computing signals...", end = " ")
    for jj_specs in specs["group_signals"].values():
        # jj = 0
        # jj_specs = list(specs["group_signals"].values())[jj]

        jj_res = BT.Signals.compute(specs = jj_specs, data = ii_datastore)
        ii_datastore[jj_specs["res_name"]] = jj_res["global"]
    print("Done.")

    # Compute group weightings
    print("Computing weightings...", end = " ")
    for jj_specs in specs["group_weightings"].values():
        # jj = 0
        # jj_specs = list(specs["group_weightings"].values())[jj]

        jj_res = BT.Weightings.compute(specs = jj_specs, data = ii_datastore)
        ii_datastore[jj_specs["res_name"]] = jj_res["global"]
    print("Done.")
