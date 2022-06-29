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


scalermin =  MinMaxScaler()
stdscaler = StandardScaler()

def dummy(x,y):
    strategies = ['most_frequent', 'stratified', 'uniform', 'constant']
    test_scores = []
    for s in strategies:
        if s =='constant':
            dclf = DummyClassifier(strategy = s, random_state = 0, constant =0)
        else:
            dclf = DummyClassifier(strategy = s, random_state = 0)
        dclf.fit(x, y)
        score = dclf.score(x, y)
        test_scores.append(score)
    return(test_scores)

def newY(ycomp):
    newY = []
    for i in range(ycomp.shape[0]):
        if sum(ycomp.iloc[i,:])>5:
            newY.append(1)
        else:
            newY.append(0)
    newDY = pd.DataFrame (newY, columns = ['Y_Gold'])
    return(newDY)

##Splits into train and test and gives the validation score
def turbine_model_eval(x,y,classfier):
    X_train, X_test, y_train, y_test = train_test_split(x, y.astype('category'), test_size=0.30, random_state=42)
    classfier.fit(X_train,y_train.values.ravel())
    print(f"The score using LogisticRegression after training on Golden Yval and testing on Validation data is",
          classfier.score(X_test,y_test.values.ravel())*100)

##Compares two models and gives the test on the new dataset
def turbine_model_testing(data1,data2,model,S = False,MS = 'Standard'):
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            print(f"The score using {model} after training on Data1 and testing on Data2 is",
            model.score(stdscaler.fit_transform(data2[1]),newY(data2[0]).values.ravel())*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            model.score(scalermin.fit_transform(data2[1]),newY(data2[0]).values.ravel())*100)
    else:
        model.fit(data1[1],newY(data1[0]).values.ravel())
        print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        model.score(data2[1],newY(data2[0]).values.ravel())*100)

def evalmodel_scaled(data,bestburner,classfier,scale):
    X = scale.fit_transform(data[int(bestburner[0][0])].iloc[:,1:])
    y = data[int(bestburner[0][0])].iloc[:,0].astype('category')
    classfier.fit(X,y)
    lrlist = []
    for i in range(len(data)):
        print(f"The score using LogisticRegression after training on burner{int(bestburner[0][0])+1} and testing on burner{i+1} is",
              round(burner_score_pred_scaled(i,classfier,data,scale)*100,5),'%')
        lrlist.append(burner_score_pred_scaled(i,classfier,data,scale))

from sklearn import svm
def svc_param_tuning(nfolds,data):
    svclf = svm.SVC()
    Cs = [0.001, 0.01, 0.1, 1, 10,20]
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X = data[1], y = newY(data[0]).values.ravel())
    return(grid_search)

def rf_param_tuner(data,model):
    n_estimators = [10,30,50,70,90]
    max_depth = [2,5, 8, 15]
    min_samples_split = [2, 5, 10, 15,20,30]
    min_samples_leaf = [1, 2, 5, 10]
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,
                  min_samples_split = min_samples_split,
                 min_samples_leaf = min_samples_leaf)
    gridF = GridSearchCV(model, hyperF, cv = 3, verbose = 1,
                          n_jobs = -1)
    bestF = gridF.fit(X = data[1], y = newY(data[0]).values.ravel())
    return(bestF)

def nb_param_tuner(data,model):
    param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)}
    nbModel_grid = GridSearchCV(estimator=model, param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
    bestNBmodel = nbModel_grid.fit(X = data[1], y = newY(data[0]).values.ravel())
    return(bestNBmodel)


def turbine_modelboost_testing(data1,data2,model,S = False,MS = 'Standard'):
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            print(f"The score using {model} after training on Data1 and testing on Data2 is",
            model.score(stdscaler.fit_transform(data2[1]),newY(data2[0]).values.ravel())*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            model.score(scalermin.fit_transform(data2[1]),newY(data2[0]).values.ravel())*100)
    else:
        model.fit(np.array(data1[1]),newY(data1[0]).values.ravel())
        print(f"\nThe score using {model}  after training on Data1 and testing on Data2 is",
        model.score(np.array(data2[1]),newY(data2[0]).values.ravel())*100)
