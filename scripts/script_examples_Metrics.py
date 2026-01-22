
#%%


#%% Instantiate object
if False:
    from Backtesting import BT
    mtrc = BT.Metrics()



# %% compute_single
if False:
    from Backtesting import BT
    import numpy as np
    import pandas as pd

    n_periods = 12
    n_assets = 6
    np.random.seed(0)  # for reproducibility
    
    inds = pd.date_range(start="2019-12-31", periods=n_periods + 1, freq="ME")
    cols = [f"Asset_{i}" for i in range(1, n_assets+1)]
    px = 90 + 20 * np.random.rand(n_periods + 1, n_assets)
    prices = pd.DataFrame(px, index=inds, columns=cols)

    # data = {"prices": prices}
    # specs = {"type": "momentum"}

    BT.Metrics.momentum(prices, lookback=6, skip=2)
    BT.Metrics.momentum(prices, lookback=6, skip=2, only_last=True)

    BT.Metrics.momentum(prices, lookback=12, skip=0)


    



