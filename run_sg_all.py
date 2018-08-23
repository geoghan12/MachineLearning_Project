## Data loading
# %%
%reload_ext autoreload
%autoreload 2

import numpy as np

from house import *
from config import *
from config2 import *
del house

# %%
house = House('data/train.csv','data/test.csv')

house.cleanRP()
# house.sg_skewness(mut=0)
# for feat in house.skewed_features:
#     house.log_transform(house.train()[feat])

house.sg_ordinals()
house.label_encode_engineer()

house.convert_types(HOUSE_CONFIG)

house.engineer_features(HOUSE_CONFIG)

# Log transform after inspecting Skewness
fix_log=['GrLivArea','1stFlrSF','LotFrontage'] #'SalePrice',
fix_log_maybe=['BsmtUnfSF','BsmtFinSF1']
house.all[fix_log] = np.log1p(house.all[fix_log])
house.train()['SalePrice'] = np.log1p(house.train()['SalePrice'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test=house.test_train_split()

house.train().sample(10)

model_rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
model_rf.fit(house.train().drop(['SalePrice','Utilities'],axis=1),house.train().SalePrice)
rf_pred = model_rf.predict(self.x_test)




house.sk_random_forest(num_est=500)

house.elastic_search() # doesn't work
house.x_train.describe()
house.all.Utilities.value_counts()
house.all.Utilities.isnull().sum()
# house.sg_statsmodels()
#####
# %%
int_house=House('data/train.csv','data/test.csv')
int_house.all.loc[int_house.all.Utilities==3]

int_house.cleanRP()

int_house.all['TotalSF'] = house.all['TotalBsmtSF'] + house.all['1stFlrSF'] + house.all['2ndFlrSF']
int_house.all.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)
int_house.all['Kitchen*Quality']=int_house.all['KitchenAbvGr']*house.all['KitchenQual']
int_house.all.drop(['KitchenAbvGr','KitchenQual'],axis=1,inplace=True)
int_house.all['Fireplaces*Quality']=int_house.all['Fireplaces']*house.all['FireplaceQu']
int_house.all.drop(['Fireplaces','FireplaceQu'],axis=1,inplace=True)

int_house.sg_skewness(mut=0)
for feat in int_house.skewed_features:
    int_house.log_transform(int_house.train()[feat])

int_house.sg_skewness(mut=1)

int_house.sg_ordinals()
int_house.label_encode_engineer()

int_house.sk_random_forest(num_est=500)
int_house.elastic_search()

##### Changing some variables to Binary
# %%
bin_house=House('data/train.csv','data/test.csv')

bin_house.cleanRP()

change_binary=['TotalBsmtSF','GarageArea','2ndFlrSF','MasVnrArea','WoodDeckSF','OpenPorchSF','PoolArea']
for var in change_binary:
    bin_house.all[var+'_B']=bin_house.all[var].apply(lambda x: 1 if x > 0 else 0)
    bin_house.all.drop([var],axis=1)
bin_house.all['YearBuilt'] = bin_house.all['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

bin_house.sg_ordinals()
bin_house.label_encode_engineer()
# Log transform after inspecting Skewness
fix_log=['GrLivArea','1stFlrSF','LotFrontage'] #'SalePrice',
fix_log_maybe=['BsmtUnfSF','BsmtFinSF1']
bin_house.all[fix_log] = np.log1p(bin_house.all[fix_log])
bin_house.train()['SalePrice'] = np.log1p(bin_house.train()['SalePrice'])

bin_house.sk_random_forest(num_est=500)


# %%
