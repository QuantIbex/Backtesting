"""
Backtesting module
------------------
Contains classes to perform backtesting of various types of strategies.

TODO:
- 
- 
- 
- 


Prompt to get docstring
    Propose a professional and relevant docstring for the following method of a python class:
    
"""

#------------------------------------------------------------------------------#

# import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

#------------------------------------------------------------------------------#

class Backtester:
    """Main class for backtesting"""
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
    """Class of utility functions"""
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

class AssetsHandler:
    """
    Class to handle asset prices and returns, and aggregates groups asset 
    into portfolios using weights and group specifications.
    """
    def __init__(self):
        """Instantiate class object"""
        pass


    @staticmethod
    def rebase_prices(prices: pd.DataFrame, start_value: float = 100) -> pd.DataFrame:
        """Computes rebased prices"""

        return start_value * prices.div(prices.iloc[0, :], axis=1)

    @staticmethod
    def single_period_buy_and_hold_portfolio_prices(
            prices: pd.DataFrame, weights: pd.DataFrame, 
            start_value: float = 100) -> pd.Series:
        """Compute the price of a buy-and-hold portfolio for a single period, that is without rebalancing"""

        assert weights.shape[0] == 1, "Input 'weights' must have only 1 row."
        assert weights.index[0] == prices.index[0], \
            "Input 'weights'' date must correspond to the first date of in put 'asset_prices'"
        assert all(weights.columns == prices.columns), \
            "Columns of inputs 'weights' and 'asset_prices' must match exactly."
        
        rebased_prices = AssetsHandler.rebase_prices(prices = prices, start_value = start_value)
        norm_wghts = weights.values / weights.values.sum()

        ptf_prices = rebased_prices.mul(norm_wghts, axis=1).sum(axis=1).to_frame(name="Portfolio")
        return ptf_prices
    
    @staticmethod
    def single_period_buy_and_hold_asset_groups_prices(
            prices: pd.DataFrame, weights: pd.DataFrame,
            groups: pd.DataFrame = None, start_value: float = 100) -> pd.DataFrame:
        """Compute prices of buy-and-hold asset groups for a single period, that is without rebalancing"""

        if groups is None:
            return AssetsHandler.single_period_buy_and_hold_portfolio_prices(
                weights=weights, prices = prices, start_value = start_value)
        else:
            assert weights.shape[0] == 1, "Input 'weights' must have only 1 row."
            assert weights.index[0] == prices.index[0], \
                "Input 'weights'' date must correspond to the first date of in put 'prices'"
            assert all(weights.columns == prices.columns), \
                "Columns of inputs 'weights' and 'asset_prices' must match exactly."

            p_reb = AssetsHandler.rebase_prices(prices = prices, start_value = start_value)
            w_ser = weights.iloc[0]
            g_ser = groups.iloc[0]
            grp_sums = w_ser.groupby(g_ser).sum()
            w_reb = w_ser / grp_sums.loc[g_ser.values].values

            df = (p_reb * w_reb).T
            df.insert(0, "Group", g_ser)
            group_prices = df.groupby("Group").sum().T
            group_prices.index = prices.index
            group_prices.columns.name = None
            return group_prices

    @staticmethod
    def buy_and_hold_prices(
            prices: pd.DataFrame, weights: pd.DataFrame, groups: pd.DataFrame = None,  \
                start_value: float = 100) -> pd.DataFrame:
        """Compute prices of buy-and-hold asset groups, with potential rebalancings"""

        # assert all([dt in groups.index.values for dt in weights.index.values])

        rebal_dates = weights.index.sort_values()
        grp_rets_lst = [None] * len(rebal_dates)
        for ii, dt in enumerate(rebal_dates):
            # ii = 1
            # dt = weights.index[ii]
            ii_start = dt
            ii_end = rebal_dates[ii + 1] if ii + 1 < len(rebal_dates) else prices.index[-1]

            if ii_start < ii_end:
                ii_mask = (ii_start <= prices.index) & (prices.index <= ii_end)
                ii_prices = prices.loc[ii_mask]
                ii_weights = weights.loc[[ii_start]]
                ii_groups = None if groups is None else groups.loc[[ii_start]]

                ii_grp_prices = AssetsHandler.single_period_buy_and_hold_asset_groups_prices(
                    prices = ii_prices, weights = ii_weights, groups = ii_groups)

                if ii == 0:
                    grp_rets_lst[ii] = ii_grp_prices.pct_change()
                else:
                    grp_rets_lst[ii] = ii_grp_prices.pct_change().iloc[1:, :]
            
        grp_rets = pd.concat(grp_rets_lst, axis=0).fillna(0)
        return start_value * (1 + grp_rets).cumprod()



#------------------------------------------------------------------------------#

class Metrics:
    """Class to compute and handle metrics"""
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def compute(specs: dict, data: dict) -> dict:
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

        return {"singles": single_metrics, "global": global_metrics}

    @staticmethod
    def aggregate(specs: dict, metrics: list) -> pd.DataFrame:
        """
        Aggregate several metrics
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(metrics, list), "Input 'metrics' must be a list."

        if specs["method"].lower() == "mean":
            
            return Utils.weighted_mean_dfs(dfs = metrics, weights = specs.get("weights"))
        else:
            raise ValueError("Invalid choice of aggregation method!")

    @staticmethod
    def compute_single(specs: list, data: dict) -> pd.DataFrame:
        """
        Compute a single metric
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
    def TODO_sharpe_momentum(prices: pd.DataFrame,
                 lookback: int,
                 skip: int = 0,
                 only_last: bool = False) -> pd.DataFrame:
        """
        Momentum over `lookback` periods, excluding the most recent `skip` periods.
        If only_last is True, return only the last row (1-row DataFrame).
        """
        raise ValueError("Not implemented yet")

#------------------------------------------------------------------------------#

class Ratings:
    """Class to compute and handle ratings"""
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def compute(specs: dict, data: dict) -> dict:
        """
        Compute individual and aggregated ratings based on the provided specifications and input data.

        This method interprets the 'ratings' configuration in `specs` — either a single dictionary
        or a list of dictionaries — then delegates the computation of each individual rating to
        `Ratings.compute_single()`. If multiple ratings are produced, they are combined into a
        global rating using `Ratings.aggregate()`.

        Parameters
        ----------
        specs : dict
            Rating specification dictionary containing:
            - "ratings" : dict or list of dicts defining one or more rating configurations.
            - "aggregate" : dict specifying aggregation parameters (used when multiple ratings exist).
        data : dict
            Input data required to compute the ratings.

        Returns
        -------
        dict
            A dictionary with:
            - "singles": list of computed individual ratings.
            - "global": aggregated global rating (or the single rating if only one was computed).

        Raises
        ------
        TypeError
            If the 'ratings' element in `specs` is not a dict or list of dicts.
        AssertionError
            If `specs` or `data` are not dictionaries.
        """
        
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if isinstance(specs["ratings"], dict):
            r_specs = [specs["ratings"]]
        elif isinstance(specs["ratings"], list):
            r_specs = specs["ratings"]
        else:
            raise TypeError("Element 'ratings' of input 'specs' must be a list or a dict!")

        single_ratings = [Ratings.compute_single(specs = s, data = data) for s in r_specs]

        if len(single_ratings) > 1:
            global_ratings = Ratings.aggregate(specs = specs["aggregate"], ratings = single_ratings)
        else:
            global_ratings = single_ratings[0]

        return {"singles": single_ratings, "global": global_ratings}

    @staticmethod
    def aggregate(specs: dict, ratings: list) -> pd.DataFrame:
        """
        Aggregate several ratings
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(ratings, list), "Input 'ratings' must be a list."

        if specs["method"].lower() == "mean":
            return Utils.weighted_mean_dfs(dfs = ratings, weights = specs.get("weights"))
        else:
            raise ValueError("Invalid choice of aggregation method!")

    @staticmethod
    def compute_single(specs: list, data: dict) -> pd.DataFrame:
        """
        Compute a single rating
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if specs["type"].lower() == "identity":
            return Ratings.identity(metrics = data[specs["var_name"]])
        elif specs["type"].lower() == "rank":
            return Ratings.rank(metrics = data[specs["var_name"]])
        elif specs["type"].lower() == "uscore":
            return Ratings.uscore(metrics = data[specs["var_name"]], scaling = specs.get("scaling"))
        elif specs["type"].lower() == "zscore":
            return Ratings.zscore(metrics = data[specs["var_name"]])
        else:
            raise ValueError("Invalid choice of rating type!")

    @staticmethod
    def identity(metrics: pd.DataFrame) -> pd.DataFrame:
        """Compute ratings as identity (unchanged values)"""
        return metrics

    @staticmethod
    def rank(metrics: pd.DataFrame) -> pd.DataFrame:
        """Compute ratings based on rank in ascending order"""
        return metrics.rank(axis=1, ascending=True)
    
    @staticmethod
    def uscore(metrics: pd.DataFrame, scaling: int = "n-1") -> pd.DataFrame:
        """
        Compute ratings based on normalized ranks.
        Possible scalings are:
         - "n-1" (default): computed as (rank(x) - 1) / ( n - 1), and values are in [0, 1].
         - "n: computed as rank(x) / n, and values are in [1/n, 1].
         - "n+1": computed as rank(x) / ( n + 1), and values are in [1/(n+1), n/(n+1)].
        """

        rnks = Ratings.rank(metrics=metrics)
        cnts = metrics.count(axis=1)

        if (scaling is None) | (scaling == "n-1"):
            return (rnks - 1).div(cnts - 1, axis=0)
        elif scaling == "n":
            return rnks.div(cnts, axis=0)
        elif scaling == "n+1":
            return rnks.div(cnts + 1, axis=0)
        else:
            raise ValueError("Invalid choice of scaling type!")

    @staticmethod
    def zscore(metrics: pd.DataFrame) -> pd.DataFrame:
        """Compute ratings based on zscore"""
        return metrics.sub(metrics.mean(axis=1), axis=0).div(metrics.std(axis=1), axis=0)

#------------------------------------------------------------------------------#

class Signals:
    """Class to compute and handle signals"""
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def compute(specs: dict, data: dict) -> dict:
                
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if isinstance(specs["signals"], dict):
            s_specs = [specs["signals"]]
        elif isinstance(specs["signals"], list):
            s_specs = specs["signals"]
        else:
            raise TypeError("Element 'signals' of input 'specs' must be a list or a dict!")

        single_signals = [Signals.compute_single(specs = s, data = data) for s in s_specs]

        if len(single_signals) > 1:
            global_signals = Signals.aggregate(specs = specs["aggregate"], signals = single_signals)
        else:
            global_signals = single_signals[0]

        return {"singles": single_signals, "global": global_signals}

    @staticmethod
    def aggregate(specs: dict, signals: list) -> pd.DataFrame:
        """
        Aggregate several signals
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(signals, list), "Input 'signals' must be a list."

        if specs["method"].lower() == "mean":
            return Utils.weighted_mean_dfs(dfs = signals, weights = specs.get("weights"))
        else:
            raise ValueError("Invalid choice of aggregation method!")

    @staticmethod
    def compute_single(specs: list, data: dict) -> pd.DataFrame:
        """
        Compute a single signal
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if specs["type"].lower() == "top-bottom":
            return Signals.top_bottom(ratings = data[specs["var_name"]],
                                      top = specs.get("top"),
                                      bottom = specs.get("bottom"))
        else:
            raise ValueError("Invalid choice of rating type!")

    @staticmethod
    def top_bottom(ratings: pd.DataFrame, top: int = 0, bottom: int = 0) -> pd.DataFrame:
        """Compute signals based on the top-bottom approach"""

        if top is None:
            top = 0
        else:
            assert isinstance(top, int), "Input 'top' must be an 'int'"
            
        if bottom is None:
            bottom = 0
        else:
            assert isinstance(bottom, int), "Input 'bottom' must be an 'int'"
        
        assert all(top + bottom <= ratings.count(axis=1)), \
            "The sum of inputs 'top' and 'bottom' must be less or equal to the number of available ratings"

        signals = pd.DataFrame(0.0, index = ratings.index, columns=ratings.columns)
        if top > 0:
            is_long = ratings.rank(axis=1, ascending=False) <= top
            signals[is_long] = 1.0
        if bottom > 0:
            is_short = ratings.rank(axis=1, ascending=True) <= bottom
            signals[is_short] = -1.0

        return signals

#------------------------------------------------------------------------------#

class Weightings:
    """Class to compute and handle weightings"""
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def compute(specs: dict, data: dict) -> dict:
                
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if isinstance(specs["weightings"], dict):
            w_specs = [specs["weightings"]]
        elif isinstance(specs["weightings"], list):
            w_specs = specs["weightings"]
        else:
            raise TypeError("Element 'weightings' of input 'specs' must be a list or a dict!")

        single_weightings = [Weightings.compute_single(specs = s, data = data) for s in w_specs]

        if len(single_weightings) > 1:
            global_weightings = Weightings.aggregate(specs = specs["aggregate"], weightings = single_weightings)
        else:
            global_weightings = single_weightings[0]

        return {"singles": single_weightings, "global": global_weightings}

    @staticmethod
    def aggregate(specs: dict, weightings: list) -> pd.DataFrame:
        """
        Aggregate several weightings
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(weightings, list), "Input 'signals' must be a list."

        if specs["method"].lower() == "mean":
            return Utils.weighted_mean_dfs(dfs = weightings, weights = specs.get("weights"))
        else:
            raise ValueError("Invalid choice of aggregation method!")

    @staticmethod
    def compute_single(specs: list, data: dict) -> pd.DataFrame:
        """
        Compute a single weighting
        """
        assert isinstance(specs, dict), "Input 'specs' must be a dict."
        assert isinstance(data, dict), "Input 'data' must be a dict."

        if specs["type"].lower() == "equally-weighted":
            return Weightings.equally_weighted(signals = data[specs["var_name"]])
        else:
            raise ValueError("Invalid choice of weighting type!")

    @staticmethod
    def equally_weighted(signals: pd.DataFrame) -> pd.DataFrame:
        """Compute weightings based on the equally-weighted approach"""
        
        longs = pd.DataFrame(0.0, index = signals.index, columns = signals.columns)
        mask_long = signals > 0
        if mask_long.any(axis=None):
            longs[mask_long] = 1
            longs = longs.div(longs.sum(axis=1), axis=0).fillna(0)
        
        shorts = pd.DataFrame(0.0, index = signals.index, columns = signals.columns)
        mask_short = signals < 0
        if mask_short.any(axis=None):
            shorts[mask_short] = -1
            shorts = shorts.div(-shorts.sum(axis=1), axis=0).fillna(0)

        return longs + shorts


#------------------------------------------------------------------------------#

class Groups:
    """Class to compute and handle asset groups"""
    def __init__(self):
        """Instantiate class object"""
        pass
    

    @staticmethod
    def compute_prices(asset_prices: pd.DataFrame, asset_labels: pd.DataFrame) -> pd.DataFrame:
        """Computes groups' historical prices"""

        pass

    @staticmethod
    def none(prices: pd.DataFrame, ref_date = None) -> pd.DataFrame:
        """
        Returns asset names as group labels (each asset forms a group).
        If ref_date is None, return only labels for the last available date (1-row DataFrame).
        """
        
        if ref_date is None:
            ref_date = prices.index.values[[-1]]

        groups = pd.DataFrame(index=ref_date, columns=prices.columns)
        groups.loc[:, :] = prices.columns
        return groups

    @staticmethod
    def labels(labels: pd.DataFrame, ref_date = None) -> pd.DataFrame:
        """
        Returns groups as specified by labels.
        If ref_date is None, return only labels for the last available date (1-row DataFrame).
        """
        if ref_date is None:
            ref_date = labels.index.values[[-1]]

        return labels.loc[ref_date]
    
    # TODO: implement this methos
    @staticmethod
    def clustering(prices: pd.DataFrame, ref_date = None, method_specs=None) -> pd.DataFrame:
        """
        TBD.
        """
        # raise NotImplementedError("Method 'clustering' not implemented yet!")
        pass
        

#------------------------------------------------------------------------------#

class DailyWeights:
    """
    Class to handle daily open, close, and end-of-day weights.
    Close weights are weights at market close based existing holding and close prices.
    End-of day weights are weights based on holdings after allocation changes performed at the close.
    Open weights are weights based on existing holdings at market open and open prices.
    """
    def __init__(self):
        """Instantiate class object"""
        self.open_weights = None
        self.close_weights = None
        self.eod_weights = None
        self.open_prices = None
        self.close_prices = None

    @staticmethod
    def equal_weights(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute equal weights for assets having prices"""
        weights = pd.DataFrame(1.0, index = prices.index, columns=prices.columns)
        weights.mask(prices.isna(), 0.0, inplace=True)
        return weights.div(prices.count(axis=1), axis=0)
        
    @staticmethod
    def drifting_weights(start_weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute drifting weights"""

        assert start_weights.shape[0] == 1
        assert start_weights.index[0] == prices.index[0]

        rets = prices.pct_change().fillna(0)
        weights = (1 + rets).cumprod().mul(start_weights.values, axis=1)
        return weights.div(weights.sum(axis=1), axis=0)


    @staticmethod
    def compute_eod_weights(close_weights: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
        """Compute end-of-day prices from close weights and close prices."""

        wgt = close_weights.shift(-1) / (close_prices.shift(-1) / close_prices)
        wgt = wgt.fillna(0)
        wgt = wgt.div(wgt.sum(axis=1), axis=0)

        eod_weights = pd.DataFrame(index = wgt.index, columns = wgt.columns, dtype=float)

        # eod_weights = close_weights.copy()
        eod_weights.iloc[:-1, :] = wgt.iloc[:-1, :]
        return  eod_weights


#------------------------------------------------------------------------------#
# Obsolete classe
class zzz_Weights:
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