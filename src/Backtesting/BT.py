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

class Utils:
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def generate_random_prices(n_periods: int, n_assets: int, seed:int = 0) -> pd.DataFrame:
        """
        Helper method. Generate a datafram with random prices.
        """
        if False:
            n_periods = 12
            n_assets = 6
        np.random.seed(seed)  # for reproducibility        
        inds = pd.date_range(start="2019-12-31", periods=n_periods + 1, freq="ME")
        cols = [f"Asset_{i}" for i in range(1, n_assets+1)]
        px = 90 + 20 * np.random.rand(n_periods + 1, n_assets)
        prices = pd.DataFrame(px, index=inds, columns=cols)
        return prices

    @staticmethod
    def all_same_index_columns(dfs: list) -> bool:
        """
        Verify that all DataFrames in a list share identical indices and columns, 
        by comparing each against the first one.
        This approach confirms exact matches, including order and type.
        Returns a boolean.
        """
        assert isinstance(dfs, list), "Input 'dfs' must be a list."

        if len(dfs) == 1:
            return True
        else:
            ref_index = dfs[0].index
            ref_cols = dfs[0].columns
            return all(
                df.index.equals(ref_index) and df.columns.equals(ref_cols)
                for df in dfs[1:]
            )

    @staticmethod
    def weighted_mean_dfs(dfs: list, weights: list = None) -> pd.DataFrame:
        """
        Compute weighted mean of a list of DataFrames.
        Returns a single DataFrame with same index/columns.
        """

        assert Utils.all_same_index_columns(dfs), "Input 'dfs' must contain compatible pd.DataFrame."

        # stack into 3D array: (n_dfs, n_rows, n_cols)
        arr = np.stack([df.to_numpy() for df in dfs], axis=0)

        # weighted mean along first axis
        wm = np.average(arr, axis=0, weights=weights)

        # reconstruct DataFrame
        return pd.DataFrame(wm, index=dfs[0].index, columns=dfs[0].columns)

#------------------------------------------------------------------------------#

class Metrics:
    def __init__(self):
        """Instantiate class object"""
        pass


    @staticmethod
    def compute(specs: dict, data: dict):
        """
        Compute single and aggregated metrics from the provided specifications and data.

        This method accepts a metric specification dictionary and a data dictionary,
        computes each requested metric via 'Metrics.compute_single', and optionally
        aggregates them using 'Metrics.aggregate' when multiple metrics are defined.
        The 'specs["metrics"]' entry may be either a single metric specification
        (dict) or a list of metric specifications. If multiple single metrics are
        computed, the aggregation behavior is controlled by 'specs["aggregate"]'.

        Parameters
        ----------
        specs : dict
            Dictionary defining the metrics to compute and, optionally, how to
            aggregate them. Must contain a 'metrics' key with either a dict
            (single metric specification) or a list of dicts (multiple metrics).
            When multiple metrics are provided, an 'aggregate' key is expected
            to define the aggregation configuration.
        data : dict
            Input data required to compute the metrics. This is passed unchanged
            to 'Metrics.compute_single' for each metric specification.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - 'single_metrics': list
                List of metric results returned by 'Metrics.compute_single' for
                each element in 'specs["metrics"]'.
            - 'global_metrics': Any
                Aggregated result returned by 'Metrics.aggregate' when multiple
                metrics are computed, or the single metric result when only one
                metric is specified.

        Raises
        ------
        AssertionError
            If 'specs' or 'data' is not a dictionary.
        TypeError
            If 'specs["metrics"]' is neither a dictionary nor a list.
        """

        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if isinstance(specs["metrics"], dict):
            m_specs = [specs["metrics"]]
        elif isinstance(specs["metrics"], list):
            m_specs = specs["metrics"]
        else:
            raise TypeError("Element 'metrics' of input 'specs' must be a list or a dict!")

        single_metrics = [Metrics.compute_single(specs = s, data = data) for s in m_specs]

        if len(single_metrics) > 1:
            global_metrics = Metrics.aggregate(specs = specs["aggregate"], metrics = single_metrics)
        else:
            global_metrics = single_metrics[0]

        return {"single_metrics": single_metrics, "global_metrics": global_metrics}


    @staticmethod
    def aggregate(specs, metrics):
        """
        Aggregate several metrics
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(metrics, list), "Input 'data' must be a list."

        if specs["method"].lower() == "mean":
            w = specs.get("weights")
            return Utils.weighted_mean_dfs(dfs = metrics, weights = w)
        else:
            raise ValueError("Invalid choice of aggregation method!")


    @staticmethod
    def compute_single(specs: list, data: list) -> pd.DataFrame:
        """
        Compute single metric
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if specs["type"].lower() == "momentum":
            return Metrics.momentum(
                prices = data[specs["var_name"]],
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