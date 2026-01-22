
#%%


#%% Instantiate object
if False:
    from Backtesting import BT
    wgt = BT.Weights()



# %%
if False:
    from Backtesting import BT
    import numpy as np
    import pandas as pd

    wgt = pd.DataFrame([
        [0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0], 
        [0.5, 0.3, 0.2],
        [0.4, 0.3, 0.1],
        [0.8, 0.4, -0.2],
        [0.7, 0.3, np.nan]], columns = ["A", "B", "C"])



    actual = BT.Weights.normalize_weights(weights= wgt)
    expected = 0



