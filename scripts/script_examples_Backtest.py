
#%%


# %% Initialization

import numpy as np
import pandas as pd
from Backtesting import BT


# %% Functions to generate toy data and specs
def _generate_toy_data():
    sctr = np.array(
        [["A"]*25, 
        ["A"]*25, 
        ["A"]*13 + ["B"]*12, 
        ["B"]*25,
        ["B"]*19 + ["C"]*6, 
        ]).T

    cntry = np.array(
        [["Germany"]*25, 
        ["Germany"]*25, 
        ["Germany"]*13 + ["France"]*12,
        ["France"]*25,
        ["France"]*19 + ["Italy"]*6,
        ]).T

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
            "res_name_asset_group_label": "asset_sector_groups",
            "res_name_prices": "sector_group_prices"},
        "group_labels_country": {
            "type": "labels",
            "var_name": "asset_countries",  
            "res_name_asset_group_label": "asset_country_groups",
            "res_name_prices": "country_group_prices"},
        }

    group_metrics = {
        "group_mom_6_0": {
            "type": "momentum",
            "var_name": "sector_group_prices",
            "lookback": 6,
            "skip": 0,
            "res_name_metric": ""},
        "group_mom_12_0": {
            "type": "momentum",
            "var_name": "sector_group_prices",
            "lookback": 12,
            "skip": 0,
            "res_name_metric": ""}
    }

    group_ratings = {
        "group_rat_6-12_0": {
            "ratings": [
                    {"var_name": "group_mom_6_0", "type": "uscore", "scaling": "n+1"},
                    {"var_name": "group_mom_12_0", "type": "uscore", "scaling": "n+1"}
                ], 
            "aggregate": {"method": "mean"}
            }

    }

    group_signals = {
        "group_sig_6-12_0":{}
    }

    group_weightings = {}


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


# %% BACKTESTING PROCEDURE

data = _generate_toy_data()
specs = _generate_toy_specs()


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

    # Form groups

    for jj, jj_specs in specs["asset_groups"]:
        # jj = 0
        # jj_specs





