import pandas as pd
df=pd.read_csv(r"survey_results_public.csv")

df.head()

import matplotlib.pyplot as plt
df=df[['YearsCodePro', 'Country', 'EdLevel', 'Employment', 'ConvertedCompYearly']]
df=df.rename({'ConvertedCompYearly':'Salary'},axis=1)

# Replace missing value using predictions by randomforest regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np

features=['Country', 'EdLevel', 'Employment', 'YearsCodePro']
df[features]=SimpleImputer(strategy='most_frequent').fit_transform(df[features])
df_model=df[features+ ['Salary']]

df_model=pd.get_dummies(df_model, drop_first=True)
# Split into datasets for training
known=df_model[df_model['Salary'].notna()]
unknown=df_model[df_model['Salary'].isna()]

X=known.drop('Salary', axis=1)
y=np.log1p(known.Salary)
model=RandomForestRegressor(n_estimators=100, random_state=50).fit(X,y)

predicted_log=model.predict(unknown.drop('Salary', axis=1))
predicted=np.expm1(predicted_log)
df.loc[df.Salary.isna(), 'Salary']=predicted

def shorten_categories(categories, cutoff):
  categorical_map={}
  for i in range(len(categories)):
    if categories.values[i] <= cutoff:
      categorical_map[categories.index[i]]= 'Other'
    else:
      categorical_map[categories.index[i]]= categories.index[i]
  return categorical_map

categorical_map=shorten_categories(df.Country.value_counts(), 400)
df['Country']=df['Country'].map(categorical_map)
df.Country.value_counts()

df=df[df['Employment']=='Employed, full-time']
df=df.drop('Employment',axis=1)

# Visualize
fig,ax=plt.subplots(figsize=(12,7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary distribution by country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

df=df[df['Salary']<=1000000]
df=df[df['Salary']>=10000]
df=df[df['Country']!='Other']

# Clean the data
def clean_experience(x):
  if x=='More than 50 years':
    return 50
  if x=='Less than 1 year':
    return 0.5
  return float(x)

df['YearsCodePro']=df['YearsCodePro'].apply(clean_experience)


def clean_education(x):
  if 'Bachelor’s degree' in x:
    return 'Bachelor’s degree'
  if 'Professional degree' in x or "Master's degree" in x:
    return 'Post grad'
  return 'Less than a Bachelors'

df['EdLevel']=df['EdLevel'].apply(clean_education)


# Encoding
df_encoded=pd.get_dummies(df, columns=['Country', 'EdLevel', 'YearsCodePro'], drop_first=True)
df_encoded['Salary']=np.log1p(df['Salary'])

# Predictions
from sklearn.linear_model import LinearRegression
X=df_encoded.drop('Salary', axis=1)
y=df_encoded['Salary']
linear_reg=LinearRegression().fit(X,y.values)

from sklearn.metrics import mean_squared_error
linear_preds=linear_reg.predict(X)
rmse=np.sqrt(mean_squared_error(y.values, linear_preds))
print('MSE:', mean_squared_error(y.values, linear_preds))
print('RMSE score: ', rmse)
print('Mean RMSE score: ', rmse.mean())
print('\n')
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=50).fit(X,y.values)
rf_preds=rf.predict(X)
rmse=np.sqrt(mean_squared_error(y.values, rf_preds))
print('MSE:', mean_squared_error(y.values, rf_preds))
print('RMSE score: ', rmse)
print('Mean RMSE score: ', rmse.mean())
print('\n')
# RandomForestRegressor with Randomized GridsearchCV
from sklearn.model_selection import GridSearchCV
param_grid={
    'n_estimators': [50,100],
    'max_depth': [3,5],
    'min_samples_split':[2,4]
}
grid_search=GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X,y.values)
gridsearch_preds=grid_search.predict(X)
rmse=np.sqrt(mean_squared_error(y.values, gridsearch_preds))
print('MSE:', mean_squared_error(y.values, gridsearch_preds))
print('RMSE score: ', rmse)
print('Mean RMSE score: ', rmse.mean())
print('Best params',grid_search.best_params_)
print('Best score',grid_search.best_score_)
print('\n')
# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor(n_estimators=100, random_state=50,learning_rate=0.1).fit(X,y.values)
gb_preds=gb.predict(X)
rmse=np.sqrt(mean_squared_error(y.values, gb_preds))
print('MSE:', mean_squared_error(y.values, gb_preds))
print('RMSE score: ', rmse)
print('Mean RMSE score: ', rmse.mean())
print('\n')
# XG Boost
from xgboost import XGBRegressor
xgb=XGBRegressor(n_estimators=1000, random_state=50,learning_rate=0.02).fit(X,y.values)
xgb_preds=xgb.predict(X)
rmse=np.sqrt(mean_squared_error(y.values, xgb_preds))
print('MSE:', mean_squared_error(y.values, xgb_preds))
print('RMSE score: ', rmse)
print('Mean RMSE score: ', rmse.mean())

"""### Saving the model"""

import pickle

# Save the model to a file
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb,file)

# Load the model
def get_model():
  with open('xgb_model.pkl', 'rb') as file:
      loaded_model = pickle.load(file)
  return loaded_model

sample=pd.DataFrame({
    'Country': ['United States'],
    'EdLevel': ["Master's degree"],
    'YearsCodePro': [15]
})

sample_encoded=pd.get_dummies(sample)
sample_encoded=sample_encoded.reindex(columns=df_encoded.columns.drop('Salary'), fill_value=0)

loaded_model=get_model()
predicted_salary=np.expm1(loaded_model.predict(sample_encoded))
print('Predicted salary', predicted_salary)


