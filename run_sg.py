## Data loading
import numpy as np

# %%
%reload_ext autoreload
%autoreload 2

from house import *
from config import *
del house
house = House('data/train.csv','data/test.csv')
# %%

house.cleanRP()

house.sg_skewness(mut=0)
for feat in house.skewed_features:
    house.log_transform(house.train()[feat])

house.sg_ordinals()
house.label_encode_engineer()

# Log transform after inspecting Skewness
fix_log=['GrLivArea','SalePrice','1stFlrSF','LotFrontage']
fix_log_maybe=['BsmtUnfSF','BsmtFinSF1']

house.all[fix_log] = np.log1p(house.all[fix_log])

house.sk_random_forest(num_est=500)
