
#%%


# %% Initialization

import numpy as np
import pandas as pd
from Backtesting import BT


# %% Functions to generate toy data and specs
def _generate_toy_data():
    n_obs = 92
    sctr = np.concatenate([
        np.tile(["A"], (n_obs, 5)),
        np.tile(["B"], (n_obs, 5)),
        np.tile(["C"], (n_obs, 5)),
        np.tile(["D"], (n_obs, 5)),
        np.tile(["E"], (n_obs, 5))], axis=1)
    cntry = np.concatenate([
        np.tile(["Germany"], (n_obs, 4)),
        np.tile(["France"], (n_obs, 6)),
        np.tile(["Italy"], (n_obs, 4)),
        np.tile(["Spain"], (n_obs, 6)),
        np.tile(["Portugal"], (n_obs, 5))], axis=1)

    bench_wgts =  np.random.rand(cntry.shape[0], cntry.shape[1])        # uniform in [0, 1]
    bench_wgts /= bench_wgts.sum(axis=1, keepdims=True)                 # normalize rows to sum to 1.0

    asset_prices = BT.Utils.generate_random_prices(
        n_periods = n_obs, n_assets = sctr.shape[1], freq = "B", start = "2019-12-01", seed = 1).round()
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
        "period": ["2019-12-31", "2020-02-28"]
        }

    strategy_specs = {
        "strategies": {
            "indexing":{
                "model_portfolio": "bench_weights",
                "rebal_freq": "W-FRI",
                "strat_name": "strat_bench"
            },
            "sector_momentum": {
                "model_portfolio": "group_strat_sector_ref_weights",
                "rebal_freq": "W-FRI",
                "tranches": {
                    "number": 3,
                    "reset_frequency": "each",
                    "reset_type": "equally-weighted"
                    },
                "strat_name": "strat_sector_momentum",
            },
            "country_momentum": {
                "model_portfolio": "group_strat_country_ref_weights",
                "rebal_freq": "W-FRI",
                "tranches": {
                    "number": 4,
                    "reset_frequency": "each",
                    "reset_type": "equally-weighted"
                    },
                "strat_name": "strat_sector_momentum",
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
        "sector_mom_6_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "sector_groups_prices",
                "lookback": 6,
                "skip": 0},
            "res_name": "group_metric_sector_mom_6_0"
            },
        "sector_mom_12_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "sector_groups_prices",
                "lookback": 12,
                "skip": 0},
            "res_name": "group_metric_sector_mom_12_0"
        },
        "country_mom_6_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "country_groups_prices",
                "lookback": 6,
                "skip": 0},
            "res_name": "group_metric_country_mom_6_0"
            },
        "country_mom_12_0": {
            "metrics": {
                "type": "momentum",
                "var_name": "country_groups_prices",
                "lookback": 12,
                "skip": 0},
            "res_name": "group_metric_country_mom_12_0"
        }
    }

    group_ratings = {
        "sector_uscore_6-12_0": {
            "ratings": [
                    {"var_name": "group_metric_sector_mom_6_0", "type": "uscore", "scaling": "n+1"},
                    {"var_name": "group_metric_sector_mom_12_0", "type": "uscore", "scaling": "n+1"}
                ], 
            "aggregate": {"method": "mean"},
            "res_name": "group_rating_sector_uscore_6-12_0"
            },
        "country_uscore_6-12_0": {
            "ratings": [
                    {"var_name": "group_metric_country_mom_6_0", "type": "uscore", "scaling": "n+1"},
                    {"var_name": "group_metric_country_mom_12_0", "type": "uscore", "scaling": "n+1"}
                ], 
            "aggregate": {"method": "mean"},
            "res_name": "group_rating_country_uscore_6-12_0"
            }
    }

    group_signals = {
        "sector_top-bottom_2-2":{
            "signals":{"var_name": "group_rating_sector_uscore_6-12_0", "type": "top-bottom", "top": 2, "bottom": 2},
            "res_name": "group_signal_sector_top-bottom_2-2"
        },
        "country_top-bottom_2-2":{
            "signals":{"var_name": "group_rating_country_uscore_6-12_0", "type": "top-bottom", "top": 2, "bottom": 2},
            "res_name": "group_signal_country_top-bottom_2-2"
        }
    }

    group_weightings = {
        "sector_mom_ew": {
            "weightings": {"var_name": "group_signal_sector_top-bottom_2-2", "type": "equally-weighted"},
            "res_name": "group_weight_sector_tilts_mom",
        },
        "country_mom_ew": {
            "weightings": {"var_name": "group_signal_country_top-bottom_2-2", "type": "equally-weighted"},
            "res_name": "group_weight_country_tilts_mom",
        }

    }

    group_weights_scatterings = {
        "sector_strat": {
            "scatterings": {"var_name": "group_weight_sector_tilts_mom", 
                            "type": "bench_tilt",
                            "group_compositions": "sector_groups_labels"},
            "res_name": "group_strat_sector_ref_weights"
        },
        "country_strat": {
            "scatterings": {"var_name": "group_weight_country_tilts_mom", 
                            "type": "bench_tilt",
                            "group_compositions": "country_groups_labels"},
            "res_name": "group_strat_country_ref_weights"
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
        "group_weightings": group_weightings,
        "group_weights_scatterings": group_weights_scatterings
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

# Rebalancing frequencies
rebal_freqs = list(set([dct["rebal_freq"] for dct in specs["strategy_specs"]["strategies"].values()]))

# Iterate on trading days
for ii, ii_dt in enumerate(trading_days):
    # ii = 0
    # ii_dt = trading_days[ii]
    print(f"Iteration {ii}:  {ii_dt}")

    #----------------------------------------------------------------------------------------------#
    # COMPUTE MODEL PORTFOLIO                                                                      # 
    #----------------------------------------------------------------------------------------------#

    is_rebal_day = any([BT.Utils.is_frequency_date(date = ii_dt, freq=frq) for frq in rebal_freqs])
    if is_rebal_day:
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

        #--------------------------------------------------------------------------------#
        # COMPUTE GROUP-BASED WEIGHTS                                                    # 
        #--------------------------------------------------------------------------------#

        # Form groups    
        if specs.get("asset_groups") is not None:
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
        else:
            print("Skipped forming groups.")

        # Compute group metrics
        if specs.get("group_metrics") is not None:
            print("Computing group metrics...", end = " ")
            for jj_specs in specs["group_metrics"].values():
                # jj = 1
                # jj_specs = list(specs["group_metrics"].values())[jj]

                jj_res = BT.Metrics.compute(specs = jj_specs, data = ii_datastore)
                ii_datastore[jj_specs["res_name"]] = jj_res["global"]
            print("Done.")
        else:
            print("Skipped computing group metrics.")

        # Compute group ratings
        if specs.get("group_ratings") is not None:
            print("Computing group ratings...", end = " ")
            for jj_specs in specs["group_ratings"].values():
                # jj = 0
                # jj_specs = list(specs["group_ratings"].values())[jj]

                jj_res = BT.Ratings.compute(specs = jj_specs, data = ii_datastore)
                ii_datastore[jj_specs["res_name"]] = jj_res["global"]
            print("Done.")
        else:
            print("Skipped computing group ratings.")

        # Compute group signals
        if specs.get("group_signals") is not None:
            print("Computing group signals...", end = " ")
            for jj_specs in specs["group_signals"].values():
                # jj = 0
                # jj_specs = list(specs["group_signals"].values())[jj]

                jj_res = BT.Signals.compute(specs = jj_specs, data = ii_datastore)
                ii_datastore[jj_specs["res_name"]] = jj_res["global"]
            print("Done.")
        else:
            print("Skipped computing group signals.")

        # Compute group weightings
        if specs.get("group_weightings") is not None:
            print("Computing group weightings...", end = " ")
            for jj_specs in specs["group_weightings"].values():
                # jj = 0
                # jj_specs = list(specs["group_weightings"].values())[jj]

                jj_res = BT.Weightings.compute(specs = jj_specs, data = ii_datastore)            
                ii_datastore[jj_specs["res_name"]] = jj_res["global"]
            print("Done.")
        else:
            print("Skipped computing group weightings.")


        # TODO: create dedicated class for weights scatterings
        # Compute group weights scatterings
        if specs.get("group_weights_scatterings") is not None:
            print("Computing group weights scatterings...", end = " ")
            for jj_specs in specs["group_weights_scatterings"].values():
                # jj = 0
                # jj_specs = list(specs["group_weights_scatterings"].values())[jj]

                if jj_specs["scatterings"]["type"] == "bench_tilt":
                    jj_res = BT.AssetsHandler.tilt_group_weights(
                        weights = ii_bench_weights,
                        groups = ii_datastore[jj_specs["scatterings"]["group_compositions"]], 
                        group_weights_tilts = ii_datastore[jj_specs["scatterings"]["var_name"]],
                        long_only = True)
                else:
                    raise ValueError("Invalid group weights scattering type.")
                ii_datastore[jj_specs["res_name"]] = jj_res
            print("Done.")
        else:
            print("Skipped computing group weights sctterings.")


        #------------------------------------------------------------------------------------#
        # COMPUTE ASSET-BASED WEIGHTS                                                        # 
        #------------------------------------------------------------------------------------#

        # Compute asset metrics
        if specs.get("asset_metrics") is not None:
            print("Computing asset metrics...", end = " ")
            for jj_specs in specs["asset_metrics"].values():
                # jj = 1
                # jj_specs = list(specs["asset_metrics"].values())[jj]
                Warning("Asset metric computation not implemented yet")
            print("Done.")
        else:
            print("Skipped computing asset metrics.")

        # Compute asset ratings
        if specs.get("asset_ratings") is not None:
            print("Computing asset ratings...", end = " ")
            for jj_specs in specs["asset_ratings"].values():
                # jj = 1
                # jj_specs = list(specs["asset_ratings"].values())[jj]
                Warning("Asset ratings computation not implemented yet")
            print("Done.")
        else:
            print("Skipped computing asset ratings.")

        # Compute asset signals
        if specs.get("asset_signals") is not None:
            print("Computing asset signals...", end = " ")
            for jj_specs in specs["asset_signals"].values():
                # jj = 1
                # jj_specs = list(specs["asset_signals"].values())[jj]
                Warning("Asset signals computation not implemented yet")
            print("Done.")
        else:
            print("Skipped computing asset signals.")

        # Compute asset weightings
        if specs.get("weightings") is not None:
            print("Computing asset weightings...", end = " ")
            for jj_specs in specs["asset_weightings"].values():
                # jj = 1
                # jj_specs = list(specs["asset_weightings"].values())[jj]
                Warning("Asset weightings computation not implemented yet")
            print("Done.")
        else:
            print("Skipped computing asset weightings.")

        #------------------------------------------------------------------------------------#
        # COMPUTE STRATEGY WEIGHTS                                                           # 
        #------------------------------------------------------------------------------------#

        # TODO: implement portfolio tranches
        for jj_specs in specs["strategy_specs"]["strategies"].values():
            # jj = 0
            # jj_specs = list(specs["strategy_specs"]["strategies"].values())[jj]
            
            BT.Utils.is_frequency_date(date = ii_dt, freq = jj_specs["rebal_freq"])


            ii_datastore[jj_specs["model_portfolio"]]

    # ii_datastore["ref_weights"]

