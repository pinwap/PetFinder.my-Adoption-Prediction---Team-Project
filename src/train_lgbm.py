import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
import numpy as np
from features import build_features
from utils.metrics import qwk

train = pd.read_csv('data/train.csv')

y = train['AdoptionSpeed']
X = train.drop(['AdoptionSpeed', 'PetID'], axis=1) # Drop PetID, adoption speed ที่เป็นเฉลยออก
X = build_features(X)

# LightGBM parameters
params = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'metric': 'None'
}    

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #ใช้ StratifiedKFold เพื่อแบ่งข้อมูลเป็น 5 ส่วนโดยคำนึงถึงการกระจายของ target variable
oof = np.zeros(len(X)) # เตรียม array สำหรับเก็บผลลัพธ์การทำนายแบบ out-of-fold

for trn, val in folds.split(X, y):
    trn_data = lgb.Dataset(X.iloc[trn], label=y.iloc[trn])
    val_data = lgb.Dataset(X.iloc[val], label=y.iloc[val])
    
    model = lgb.train(
        params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        early_stopping_rounds=200,
        verbose_eval=300
    )
    oof[val] = model.predict(X.iloc[val]) #ทำนายค่า oof สำหรับข้อมูล validation 
print('CV QWK:', qwk(y, np.round(oof)))