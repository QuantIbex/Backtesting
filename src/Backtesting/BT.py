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
# import numpy as np
# import pandas as pd
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
        
        Iterate on days
            Determine investable universe
                Filter on liquidity
                Filter on exclusions (ESG)
                Store investable universe
            Iterate on strategies
                Compute model portfolio (if needed)
                    Apply group strategy
                        Form groups
                            Filter assets on available data
                            Appla grouping method
                        Compute group-based model portfolio
                    Store reference weights
                    Compute asset-based model portfolio
                    Store asset weights
                Store tranches weights and allocations
                Aggregate tranches

            Aggregate strategies
                Compute strategies weights
                    Compute risk/return strategy metrics
            Store model portfolio
            Determine if new model portfolio is implemented
            Determine trades
            Compute allocations, fees, prices, weights
        
        """




        pass


#------------------------------------------------------------------------------#

class Weights:
    def __init__(self):
        """Instantiate class object"""
        pass

    @staticmethod
    def normalize_weights(weights):
        
        return weights



#------------------------------------------------------------------------------#