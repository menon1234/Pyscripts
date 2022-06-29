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

def WholeTurbinePT(df):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.columns + '_'+df_c.iloc[0,:]+ '_'+df_c.iloc[1,:]
    #drop the date
    SGT750 = df_c.iloc[1:,:]
    #Change the index to sequence
    SGT750.index = range(1,SGT750.shape[0]+1)
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    SGT750_y= SGT750.filter(regex='Flame_on/off')
    # datax = data.filter(regex="^(?!$)")
    # datax = data[data.columns.drop(list(data.filter(regex='Burner')))]
    SGT750_x = SGT750.filter(regex='Frequency Pulsation_mbar')
    SGT750_x['ExhaustTemp_Ring1'] = SGT750['MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C']
    SGT750_x['ExhaustTemp_Ring2'] = SGT750['MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C']
    SGT750_x['ExhaustTemp_Ring3'] = SGT750['MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']
    # datax1 = datax[datax.columns.drop(list(datax.filter(regex='NaN')))]
    # datax1 = data.filter(regex='Ring')
    return(SGT750_y,SGT750_x)


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
        model.fit(data1.iloc[:,1:],data1[0].iloc[:,0])
        print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        model.score(data2.iloc[:,1:],data2[0].iloc[:,0]*100)


##Splits into train and test and gives the validation score
def turbine_model_eval(x,y,classfier):
    X_train, X_test, y_train, y_test = train_test_split(x, y.astype('category'), test_size=0.30, random_state=42)
    classfier.fit(X_train,y_train.values.ravel())
    print(f"The score using LogisticRegression after training on Golden Yval and testing on Validation data is",
          classfier.score(X_test,y_test.values.ravel())*100)
