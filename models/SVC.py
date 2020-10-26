# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:32:18 2020

@author: Dell
"""

import numpy as np 
import pandas as pd
import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import datetime
from tqdm import tqdm
from statistics import mean 
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import math
import sys
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pathlib


testdf=pd.read_csv("../data/test_data.csv")
traindf= pd.read_csv("../data/train_data.csv")

X_train=[]
y_train3days=[]
y_train7days=[]
y_train15days=[]
y_train30days=[]


for index,row in traindf.iterrows():
    y_train3days.append(float(row['lab_3']))
    y_train7days.append(float(row['lab_7']))
    y_train15days.append(float(row['lab_15']))
    y_train30days.append(float(row['lab_30']))
    

#Test data set-up
X_test=[]
y_test3days=[]
y_test7days=[]
y_test15days=[]
y_test30days=[]


for index,row in testdf.iterrows():
    y_test3days.append(float(row['lab_3']))
    y_test7days.append(float(row['lab_7']))
    y_test15days.append(float(row['lab_15']))
    y_test30days.append(float(row['lab_30']))
    

save_dir = os.path.join("model_predictions/","SVC/")
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) 

def SVR_regressor3(duration,X_train, y_train, X_test, y_test):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid)
    grid_search.fit(X_train, y_train)
    pred=grid_search.predict(X_test)   
    f1 = f1_score(y_test,pred)
    save_path=os.path.join(save_dir,"pred_{}.csv".format(duration))
    df = pd.DataFrame(pred)
    df.to_csv(save_path)
    # print("F1 score {} days: {}".format(f1, duration))   
    return

def SVR_regressor7(duration,X_train, y_train, X_test, y_test):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X_train, y_train)
    pred=grid_search.predict(X_test) 
    f1 = f1_score(y_test,pred)
    save_path=os.path.join(save_dir,"pred_{}.csv".format(duration))
    df = pd.DataFrame(pred)
    df.to_csv(save_path)
    # print("F1 score {} days: {}".format(f1, duration))
    return

def SVR_regressor15(duration,X_train, y_train, X_test, y_test):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid)
    grid_search.fit(X_train, y_train)
    pred=grid_search.predict(X_test) 
    f1 = f1_score(y_test,pred)
    
    save_path=os.path.join(save_dir,"pred_{}.csv".format(duration))
    df = pd.DataFrame(pred)
    df.to_csv(save_path)
    # print("F1 score {} days: {}".format(f1, duration))
    return

def SVR_regressor30(duration,X_train, y_train, X_test, y_test):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X_train, y_train)
    pred=grid_search.predict(X_test)

    f1 = f1_score(y_test,pred)
    save_path=os.path.join(save_dir,"pred_{}.csv".format(duration))
    df = pd.DataFrame(pred)
    df.to_csv(save_path)

    # print("F1 score {} days: {}".format(f1, duration))
    return


with open('../data/financial_features/X_train3days.pkl', 'rb') as f:
    X_train3days=pickle.load(f)
with open('../data/financial_features/X_train7days.pkl', 'rb') as f:
    X_train7days=pickle.load(f)
with open('../data/financial_features/X_train15days.pkl', 'rb') as f:
    X_train15days=pickle.load(f)
with open('../data/financial_features/X_train30days.pkl', 'rb') as f:
    X_train30days=pickle.load(f)
with open('../data/financial_features/X_test3days.pkl', 'rb') as f:
    X_test3days=pickle.load(f)
with open('../data/financial_features/X_test7days.pkl', 'rb') as f:
    X_test7days=pickle.load(f)
with open('../data/financial_features/X_test15days.pkl', 'rb') as f:
    X_test15days=pickle.load(f)
with open('../data/financial_features/X_test30days.pkl', 'rb') as f:
    X_test30days=pickle.load(f)


SVR_regressor3(duration=3,X_train=X_train3days, y_train=y_train3days, X_test=X_test3days, y_test=y_test3days)
SVR_regressor7(duration=7,X_train=X_train7days, y_train=y_train7days, X_test=X_test7days, y_test=y_test7days)
SVR_regressor15(duration=15,X_train=X_train15days, y_train=y_train15days, X_test=X_test15days, y_test=y_test15days)
SVR_regressor30(duration=30,X_train=X_train30days, y_train=y_train30days, X_test=X_test30days, y_test=y_test30days)











