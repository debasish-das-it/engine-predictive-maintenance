---
license: mit
tags:
  - tabular-classification
  - predictive-maintenance
  - xgboost
library_name: scikit-learn
---

# Engine Predictive Maintenance — XGBoost Classifier

Binary classifier predicting whether an engine requires maintenance (`engine_condition = 1`)
based on six sensor readings (RPM, lub-oil pressure/temp, fuel pressure,
coolant pressure/temp).

## Usage

```python
import joblib, pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="debasishdas1985/engine-predictive-maintenance-model", filename="best_engine_model.joblib")
model = joblib.load(path)
sample = pd.DataFrame([{
    "engine_rpm": 700, "lub_oil_pressure": 2.5, "fuel_pressure": 11.8,
    "coolant_pressure": 3.2, "lub_oil_temp": 84.1, "coolant_temp": 81.6,
}])
print(model.predict(sample), model.predict_proba(sample))
```

## Best hyperparameters
{'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}

## Test metrics
| metric | value |
|---|---|
| accuracy  | 0.6330 |
| precision | 0.7323 |
| recall    | 0.6585 |
| f1-score  | 0.6935 |
| ROC-AUC   | 0.6792 |
