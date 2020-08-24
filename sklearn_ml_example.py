import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
#load data
dataset_url = 'winequality-red.csv'
#data = pd.read_csv(dataset_url)
data = pd.read_csv(dataset_url, sep=';')
#print data to know it
print(data.head())
print(data.shape)
print(data.describe())
#Split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
x_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=123,
                                                  stratify=y)
#X_train_scaled = preprocessing.scale(x_train)
#print(X_train_scaled )
#print (X_train_scaled.mean(axis=0))
#print (X_train_scaled.std(axis=0))
scaler = preprocessing.StandardScaler().fit(x_train)
X_train_scaled = scaler.transform(x_train)
print(X_train_scaled.mean(axis=0))
print (X_train_scaled.std(axis=0))
X_test_scaled = scaler.transform(x_train)
print(X_test_scaled.mean(axis=0))
print( X_test_scaled.std(axis=0))
# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))
#list of tunable hyper parameter
print( pipeline.get_params())
#he hyperparameters we want to tune through cross-validation.
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

#cross validation :
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# Fit and tune model
clf.fit(x_train, y_train)
#print the best set of parameters found using CV:
print (clf.best_params_)
#get a small performance improvement by refitting the model on the entire training set.
print (clf.refit)
#predict anew set of data
y_pred = clf.predict(X_test)
# use the metrics we imported earlier to evaluate our model performance.
print (r2_score(y_test, y_pred))
print( mean_squared_error(y_test, y_pred))
#save the model to .pkl file
joblib.dump(clf, 'rf_regressor.pkl')
#to load data again to use it in future :
#clf2 = joblib.load('rf_regressor.pkl')
# Predict data set using loaded model
#clf2.predict(X_test)