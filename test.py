import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy.stats import f_oneway
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os
import pathlib

a1 = pd.read_excel('data/case_study1.xlsx')
a2 = pd.read_excel('data/case_study2.xlsx')

df1 = a1.copy()
df2 = a2.copy()


df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

df2 = df2.drop(columns_to_be_removed, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Checking common column names
for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)

# Merge the two dataframes, inner join so that no nulls are present
df = pd. merge(df1, df2, how='inner', left_on=[
               'PROSPECTID'], right_on=['PROSPECTID'])

# check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)

# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2',
          'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(
        pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)
# Since p-value is less than 0.05, we reject the null Hypothesis, and not
# removing any categorical columns

# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        numeric_columns.append(i)

# VIF sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0


for i in range(0, total_columns):

    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '---', vif_value)

    if vif_value <= 6:
        print('Appended - ', i, numeric_columns[i])
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index+1

    else:
        print('Dropped - ', i, numeric_columns[i])
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# check Anova for columns_to_be_kept


columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])

    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)


features = columns_to_be_kept_numerical + \
    ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Label encoding for the categorical features
['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()

df.loc[df['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']] = 3

df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()

df_encoded = pd.get_dummies(
    df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])


df_encoded.info()
k = df_encoded.describe()

# Machine Learing model fitting
# Data processing

# 1. Random Forest
y = df_encoded['Approved_Flag']
x = df_encoded. drop(['Approved_Flag'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy}')
print()
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# 2. xgboost


xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)


y = df_encoded['Approved_Flag']
x = df_encoded. drop(['Approved_Flag'], axis=1)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy:.2f}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# 3. Decision Tree


y = df_encoded['Approved_Flag']
x = df_encoded. drop(['Approved_Flag'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {accuracy:.2f}")
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# xgboost is giving me best results
# We will further finetune it

# Apply standard scaler


columns_to_be_scaled = ['Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
                        'max_recent_level_of_deliq', 'recent_level_of_deliq',
                        'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column

y = df_encoded['Approved_Flag']
x = df_encoded. drop(['Approved_Flag'], axis=1)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42)


xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()

# No improvement in metrices

# Hyperparameter tuning in xgboost
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42)

# Define the XGBClassifier with the initial set of hyperparameters
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

# Define the parameter grid for hyperparameter tuning

param_grid = {
    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'alpha': [1, 10, 100],
}

grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)

# Best Hyperparameters: {'alpha': 10, 'colsample_bytree': 0.9,
# 'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}


# Based on risk appetite of the bank, you will suggest P1,P2,P3,P4 to the business end user


with open('model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
