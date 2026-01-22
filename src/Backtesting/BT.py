"""
Backtesting module
------------------
Contains classes to perform backtesting of various types of strategies.

TODO:
- 
- 
- 
- 
"""

#------------------------------------------------------------------------------#

# import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------#

class Backtester:
    def __init__(self):
        """Instantiate class object"""
        self.pars = {}
        self.vars = {}

    def set_parameters(self):
        """Set backtest parameters"""
        pass

    def set_variables(self):
        """Set backtest parameters"""
        pass

    def compute_model_portfolio(self):
        """Compute the model portfolio at a given rebalancing date parameters"""
        pass

    def run(self):
        """Runs the backtest and generates model portfolios
        
        Check in markdown document
        
        """

        # Iterate on days



        pass


#------------------------------------------------------------------------------#

class Metrics:
    def __init__(self):
        """Instantiate class object"""
        pass


    def compute(self, specs, data):
        pass

    def aggregate_metrics(self, specs, metrics):
        """
        Aggregate several metrics
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(metrics, list), "Input 'data' must be a list."


        # continue dev from here (specs["method"])
        # check that metrics are compatible?


    @staticmethod
    def compute_single_metric(specs, data):
        """
        Compute single metric
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if specs["type"].lower() == "momentum":
            return Metrics.momentum(
                prices = data["prices"],
                lookback = specs["lookback"],
                skip = specs.get("skip"),
                only_last = specs.get("only_last"))
        else:
            raise ValueError("Invalid choise of metric type!")


    @staticmethod
    def momentum(prices: pd.DataFrame,
                 lookback: int,
                 skip: int = 0,
                 only_last: bool = True) -> pd.DataFrame:
        """
        Momentum over `lookback` periods, excluding the most recent `skip` periods.
        If only_last is True, return only the last row (1-row DataFrame).
        """
        if skip is None:
            skip = 0
        if only_last is None:
            only_last = True

        assert isinstance(prices, pd.DataFrame), "Input 'prices' must be a Pandas DataFrame."
        assert isinstance(lookback, int), "Input 'lookback' must be an int."
        assert isinstance(skip, int), "Input 'skip' must be an int."
        assert isinstance(only_last, bool), "Input 'only_last' must be a bool."
        assert lookback > 0, "Input 'lookback' must be positive."
        assert skip >= 0, "Input 'skip' must be non-negative."
        assert lookback < prices.shape[0], "Input 'lookback' must be less the number of periods in 'prices'."
        assert skip < lookback, "Input 'skip' must be less than 'lookback'."

        mom = prices.shift(skip).pct_change(periods=lookback)
        if only_last:
            # TODO: improve computation for speed if needed
            return mom.iloc[[-1]]
        else:
            return mom

    @staticmethod
    def sharpe_momentum(prices: pd.DataFrame,
                 lookback: int,
                 skip: int = 0,
                 only_last: bool = False) -> pd.DataFrame:
        """
        Momentum over `lookback` periods, excluding the most recent `skip` periods.
        If only_last is True, return only the last row (1-row DataFrame).
        """
        raise ValueError("Not implemented yet")


#------------------------------------------------------------------------------#

class Weights:
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def net_exposure(weights):
        """
        Computes the net exposure
        """
        return weights.sum(axis=1)

    @staticmethod
    def gross_exposure(weights):
        """
        Computes the net exposure
        """
        return weights.abs().sum(axis=1)

    @staticmethod
    def normalize(weights, net = 1, gross = 1, keep_net_short = True, fill_na = 0):

        pass

    # TODO(CDO): fix this (problème when net short. Imlement normalize function instead)

    @staticmethod
    def normalize_weights(weights, net = 1, gross = 1, fill_na = 0):
        """
        Normalize weights such that each row has a net exposure of 1        
        """

        if fill_na is not None:
            weights = weights.fillna(fill_na)

        wgt_net = Weights.net_exposure(weights)
        


        return weights.div(wgt_net, axis=0)
    



#------------------------------------------------------------------------------#