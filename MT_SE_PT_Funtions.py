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
scaler = StandardScaler()
####Confsion confusion_matrix
##Returns confusion matrix and the fplist,fnlist and the sum of FP,FN
def confusionmatrix(data,model):
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    ##Confusion Matrix
    count = 0
    for i in range(len(data)):
        cm = confusion_matrix(data[i].iloc[:,0].astype('category'),
                              model.predict(data[i].iloc[:,1:]))
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        fprlist.append(FP[0]/(FP[0]+TN[0]))
        tprlist.append(TP[0]/(TP[0]+FN[0]))
        fplist.append(cm[1,0])
        fnlist.append(cm[0,1])
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        #Update count
        count+=1
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix %i' %count);
        ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
        plt.show()
    print("The total Number of False positives in the list is",sum(fplist))
    print("The total Number of False negatives in the list is",sum(fnlist))
    return(sum(fplist),sum(fnlist),sum(fnlist)+sum(fplist))

def confusionmatrix_scaled(data,model,scale):
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    ##Confusion Matrix
    count = 0
    for i in range(len(data)):
        cm = confusion_matrix(data[i].iloc[:,0].astype('category'),
                              model.predict(scale.fit_transform(data[i].iloc[:,1:])))
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        fprlist.append(FP[0]/(FP[0]+TN[0]))
        tprlist.append(TP[0]/(TP[0]+FN[0]))
        fplist.append(cm[1,0])
        fnlist.append(cm[0,1])
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
        #Update count
        count+=1
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
        ax.set_title('Confusion Matrix %i' %count);
        ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
        plt.show()
    print("The total Number of False positives in the list is",sum(fplist))
    print("The total Number of False negatives in the list is",sum(fnlist))
    return(sum(fplist),sum(fnlist),sum(fnlist)+sum(fplist))


##PLot confusiongrid

def plotconfusiongrid(data,model):
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)
    fig.subplots_adjust(wspace=0.1)
    co = 0
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    for i in range(2):
        for j in range(4):
            cm = confusion_matrix(data[co].iloc[:,0].astype('category'),
                              model.predict(data[co].iloc[:,1:]))
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            fprlist.append(FP[0]/(FP[0]+TN[0]))
            tprlist.append(TP[0]/(TP[0]+FN[0]))
            fplist.append(cm[1,0])
            fnlist.append(cm[0,1])
            co = co + 1
            sns.heatmap(cm, annot=True, fmt='g', ax=ax[i,j],yticklabels=True, xticklabels=True) #annot=True to annotate cells, ftm='g' to disable scientific notation
    print("The total Number of False positives in the list is",sum(fplist))
    print("The total Number of False negatives in the list is",sum(fnlist))
    return(sum(fplist),sum(fnlist),sum(fnlist)+sum(fplist))

##Plot confusion grid scaled
def plotconfusiongrid_scale(data,model,scale):
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)
    fig.subplots_adjust(wspace=0.1)
    co = 0
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    for i in range(2):
        for j in range(4):
            cm = confusion_matrix(data[co].iloc[:,0].astype('category'),
                              model.predict(scale.fit_transform(data[co].iloc[:,1:])))
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            fprlist.append(FP[0]/(FP[0]+TN[0]))
            tprlist.append(TP[0]/(TP[0]+FN[0]))
            fplist.append(cm[1,0])
            fnlist.append(cm[0,1])
            co = co + 1
            sns.heatmap(cm, annot=True, fmt='g', ax=ax[i,j],yticklabels=True, xticklabels=True)
    print("The total Number of False positives in the list is",sum(fplist))
    print("The total Number of False negatives in the list is",sum(fnlist))
    return(sum(fplist),sum(fnlist),sum(fnlist)+sum(fplist))



###Evaluation functions
###Log coefficent Return
def log_coef(data,classfier):
    coef_list = []
    for i in range(len(data)):
        X = data[i].iloc[:,1:]
        y = data[i].iloc[:,0].astype('category')
        coef = classfier.fit(X,y)
        coef_list.append(coef.coef_)
    return(coef_list)

###Model Evaluater

def evalmodel(data,bestburner,classfier):
    X = data[int(bestburner[0][0])].iloc[:,1:]
    y = data[int(bestburner[0][0])].iloc[:,0].astype('category')
    classfier.fit(X,y)
    lrlist = []
    for i in range(len(data)):
        print(f"The score using LogisticRegression after training on burner{int(bestburner[0][0])+1}and testing on burner{i+1} is",
              round(burner_score_pred(i,classfier,data)*100,5),'%')
        lrlist.append(burner_score_pred(i,classfier,data))

def evalmodel_scaled(data,bestburner,classfier,scale):
    X = scale.fit_transform(data[int(bestburner[0][0])].iloc[:,1:])
    y = data[int(bestburner[0][0])].iloc[:,0].astype('category')
    classfier.fit(X,y)
    lrlist = []
    for i in range(len(data)):
        print(f"The score using LogisticRegression after training on burner{int(bestburner[0][0])+1} and testing on burner{i+1} is",
              round(burner_score_pred_scaled(i,classfier,data,scale)*100,5),'%')
        lrlist.append(burner_score_pred_scaled(i,classfier,data,scale))



def evalmodeltest(datatrain,datatest,bestburner,classfier,S = False,MS = 'Standard'):
    scalermin =  MinMaxScaler()
    scaler = StandardScaler()
    if S == True:
        if MS == 'Minmax':
            X = scalermin.fit_transform(datatrain[int(bestburner[0][0])].iloc[:,1:])
        elif MS == 'Standard':
            X = scaler.fit_transform(datatrain[int(bestburner[0][0])].iloc[:,1:])
    else:
        X = datatrain[int(bestburner[0][0])].iloc[:,1:]
    y = datatrain[int(bestburner[0][0])].iloc[:,0].astype('category')
    classfier.fit(X,y)
    lrlist = []
    for i in range(len(datatrain)):
        print(f"The score using LogisticRegression after training on burner {int(bestburner[0][0])+1} and testing on burner{i+1} is",
              round(burner_score_pred_new(i,classfier,datatest,S,MS)*100,5),'%')
        lrlist.append(burner_score_pred_new(i,classfier,datatest,S,MS))


def burner_score_pred_new(pos,model,data,S = False,MS = 'Standard'):
    ###Range of 1-8
    scalermin =  MinMaxScaler()
    scaler = StandardScaler()
    if S == True:
        if MS == 'Minmax':
            X = scalermin.fit_transform(data[pos-1].iloc[:,1:])
        elif MS == 'Standard':
            X = scaler.fit_transform(data[pos-1].iloc[:,1:])
    else:
        X = data[pos-1].iloc[:,1:]
    y = data[pos-1].iloc[:,0].astype('category')
    return(model.score(X,y))

def burner_score_pred_scaled(pos,model,data,scale):
    ###Range of 1-8
    y = data[pos-1].iloc[:,0].astype('category')
    X = scale.fit_transform(data[pos-1].iloc[:,1:])
    return(model.score(X,y))

##HighPrediction
def highestpredburner(datalist,model):
    lrlist = []
    for i in range(len(datalist)):
        X = datalist[i].iloc[:,1:]
        y = datalist[i].iloc[:,0].astype('category')
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        model.fit(X, y)
        for j in range(len(datalist)):
            # print(f"The score using LogisticRegression after training on burner {i+1} and testing on burner{j+1} is",
                  # round(burner_score_pred(j,model,datalist)*100,5),'%')
            lrlist.append(burner_score_pred(j,model,datalist))
    print("\nThe Burner which gives the highest accuracy after testing on all other burners are: ",
          round((lrlist.index(max(lrlist))/8)))
    print("The Burner which gives the lowest accuracy after testing on all other burners are:",
          round((lrlist.index(min(lrlist))/8)),'\n')
    return([round(math.ceil(lrlist.index(max(lrlist))/8))-1,max(lrlist),(lrlist.index(max(lrlist)))],
           [round(math.ceil(lrlist.index(min(lrlist))/8))-1,min(lrlist),(lrlist.index(min(lrlist)))])

def highestpredburnerscale(datalist,model,scale):
    lrlist = []
    for i in range(len(datalist)):
        X = scale.fit_transform(datalist[i].iloc[:,1:])
        y = datalist[i].iloc[:,0].astype('category')
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        model.fit(X, y)
        for j in range(len(datalist)):
            # print(f"The score using LogisticRegression after training on burner {i+1} and testing on burner{j+1} is",
                  # round(burner_score_pred_scaled(j,model,datalist,scale)*100,5),'%')
            lrlist.append(burner_score_pred_scaled(j,model,datalist,scale))
    print("\nThe Burner which gives the highest accuracy after testing on all other burners are: ",
          round((lrlist.index(max(lrlist))/8)))
    print("The Burner which gives the lowest accuracy after testing on all other burners are:",
          round((lrlist.index(min(lrlist))/8)),'\n')
    return([round(math.ceil(lrlist.index(max(lrlist))/8))-1,max(lrlist),(lrlist.index(max(lrlist)))],
           [round(math.ceil(lrlist.index(min(lrlist))/8))-1,min(lrlist),(lrlist.index(min(lrlist)))])


def burner_score_pred(pos,model,data):
    ###Range of 1-8
    y = data[pos-1].iloc[:,0].astype('category')
    X = data[pos-1].iloc[:,1:]
    return(model.score(X,y))
# SVC Tuning



def model_eval(model,data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0].astype('category')
    model.fit(X,y)
    return(model)
