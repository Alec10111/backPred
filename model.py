import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from joblib import dump, load
import pickle
import time, datetime

casas = pd.read_csv('houseprices.csv')
categoricos = list(casas.select_dtypes(['object']).columns)
#numericos = list(casas.select_dtypes(['number']).columns)
print(list(casas.select_dtypes(['number']).columns))
numericos = ['YearBuilt','TotalBath','BedroomAbvGr','YearRemodAdd','SalePrice']
y = casas[numericos]['SalePrice']
numericos.remove('SalePrice')

X = casas[numericos]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
predictores = categoricos+numericos

print(numericos)
## Definimos las arrays de predictores y de respuesta


## Creamos las columnas nuevas

## Definimos la cañería para las columnas numéricas
steps_num = [
             ('Imputador', SimpleImputer(strategy='median')),
             ('BoxCox', PowerTransformer(method='yeo-johnson'))]
numeric_transformer = Pipeline(steps_num)

steps = [('prerocesado', numeric_transformer),
         ('predictor', RandomForestRegressor(max_features = 4, max_depth=20, n_jobs=-1))]
pipe = Pipeline(steps)
#cv_results= cross_validate(pipe, X, y, cv=5, return_train_score=True)
pipe.fit(X_tr,y_tr)
print(pipe.predict([[2000,2,4,2015]]))
#f =
f = open('/Users/David/HousePrice_Pred/src/model_1.joblib','wb')
pickle.dump(pipe,f)
print(pipe.score(X_te,y_te))

"""pipe = load('model_1.joblib')"""

## Lo mismo para las categórica
"""steps_cat = [('Imputador', SimpleImputer(strategy='median')),
             ('OneHot', OneHotEncoder(handle_unknown='ignore', sparse=False)),

categorical_transformer =Pipeline(steps_num)

## Ensamblo las dos cañerías con ColumnTransformer

preprocesado = ColumnTransformer(
                                transformers=[
                                            ('numerico', numeric_transformer, numericos),
                                            ('categorico', categorical_transformer, categoricos)])

steps = [('prerocesado', preprocesado),
         ('predictor', RandomForestRegressor(max_features = 10, max_depth=18, n_jobs=-1))]
​
pipe = Pipeline(steps)

cv_results = cross_validate(pipe, X, y, cv=5, return_train_score=True)"""
"""print(cv_results)
print(cv_results['train_score'].mean())
print(cv_results['test_score'].mean())"""
#print(casas.head())