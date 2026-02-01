import pandas as pd
import numpy as np
from features import build_features

test = pd.read_csv('data/test.csv')
ids = test['PetID']
X_test = build_features(test.drop(['PetID'], axis=1))

preds = model.predict(X_test)
preds = np.clip(np.round(preds), 0, 4)

submission = pd.DataFrame({
    'PetID': ids,
    'AdoptionSpeed': preds.astype(int)
})
submission.to_csv('./submissions/submission.csv', index=False)