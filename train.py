import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

import pickle


# Parameters.
params = {
  'colsample_bytree': 1.0,
  'gamma': 0.2,
  'learning_rate': 0.05,
  'max_depth': 7,
  'n_estimators': 200,
  'subsample': 0.8
}

n_splits = 5
output_file = f'model.pkl'

# Data preparation.

url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

df_full_train = pd.read_csv(url_train, names=column_names, na_values=' ?')
df_test = pd.read_csv(url_test, names=column_names, skiprows=1, na_values=' ?')

strings = list(df_full_train.dtypes[df_full_train.dtypes == 'object'].index)
for col in strings:
    df_full_train[col] = df_full_train[col].str.lower().str.strip().str.replace(' ', '_')

strings = list(df_test.dtypes[df_test.dtypes == 'object'].index)
for col in strings:
    df_test[col] = df_test[col].str.lower().str.strip().str.replace(' ', '_')

df_full_train['income'] = df_full_train['income'].apply(lambda x: 1 if '>50k' in x.strip() else 0)

df_test['income'] = df_test['income'].apply(lambda x: 1 if '>50k' in x.strip() else 0)

df_full_train['workclass'] = df_full_train['workclass'].fillna('unknown')
df_full_train['occupation'] = df_full_train['occupation'].fillna('unknown')
df_full_train['native-country'] = df_full_train['native-country'].fillna('unknown')

numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
             'hours-per-week']

categorical = ['workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']

# Training.

def train(df_train, y_train):
  dicts = df_train[categorical + numerical].to_dict(orient='records')
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dicts)

  model = XGBClassifier(
    **params,
    eval_metric='logloss',
    use_label_encoder=False)

  model.fit(X_train, y_train)

  return dv, model

def predict(df, dv, model):
  dicts = df[categorical + numerical].to_dict(orient='records')

  X = dv.transform(dicts)
  y_pred = model.predict_proba(X)[:, 1]

  return y_pred

# Validation

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.income.values
    y_val = df_val.income.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1


print('validation results.')
print('XGB. %.3f +- %.3f.' % (np.mean(scores), np.std(scores)))

# Training the final model.

print('training the final model.')

dv, model = train(df_train, df_train.income.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.income.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc. {auc}.')

# Save the model.

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}.')