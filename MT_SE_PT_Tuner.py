
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import plotly
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

def svc_param_selection(nfolds,bestburner,data):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X = data[int(bestburner[0][0])].iloc[:,1:], y = data[int(bestburner[0][0])].iloc[:,0].astype('category'))
    return(grid_search)

def rf_tuner(data,bestburner,model):
    n_estimators = [10,30,50,70,90,120,150, 300]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,
                  min_samples_split = min_samples_split,
                 min_samples_leaf = min_samples_leaf)
    gridF = GridSearchCV(model, hyperF, cv = 3, verbose = 1,
                          n_jobs = -1)
    bestF = gridF.fit(X = data[int(bestburner[0][0])].iloc[:,1:], y = data[int(bestburner[0][0])].iloc[:,0].astype('category'))
    return(bestF)

##Tuning parameters
def nb_tuner(data,bestburner,model):
    param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)}
    nbModel_grid = GridSearchCV(estimator=model, param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
    bestNBmodel = nbModel_grid.fit(X = data[int(bestburner[0][0])].iloc[:,1:], y = data[int(bestburner[0][0])].iloc[:,0].astype('category'))
    return(bestNBmodel)
