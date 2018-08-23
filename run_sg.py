# Take 2
# %%
%reload_ext autoreload
%autoreload 2

import numpy as np

from house import *
from config import *

del house
house = House('data/train.csv','data/test.csv')

# %%


house.cleanRP()
house.sg_ordinals()
house.label_encode_engineer()

# house.all.sample(10)

fix_log=['GrLivArea','1stFlrSF','LotFrontage'] #'SalePrice',
fix_log_maybe=['BsmtUnfSF','BsmtFinSF1']
house.all[fix_log] = np.log1p(house.all[fix_log])
house.train()['SalePrice'] = np.log1p(house.train()['SalePrice'])

house.sg_statsmodels()

house.sk_random_forest(num_est=500)
