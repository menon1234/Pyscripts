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

def Model_eval(data,model,S = False,MS = 'Standard'):
    X = data.iloc[:,1:]
    y = data.iloc[:,0].astype('category')
    if S == True:
        if MS == 'Standard':
            X = stdscaler.fit_transform(X)
        elif MS == 'Minmax':
            X = scalermin.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
    model.fit(X_train,y_train)
    print(f"The score using {model} after training on 70% of the data and testing on Validation data is",
              model.score(X_test,y_test)*100)

def Model_evaluation(x,y,classfier):
    X_train, X_test, y_train, y_test = train_test_split(x, y.astype('category'), test_size=0.30, random_state=42)
    classfier.fit(X_train,y_train.values.ravel())
    print(f"The score using LogisticRegression after training on Golden Yval and testing on Validation data is",
          classfier.score(X_test,y_test.values.ravel())*100)

def Model_testing_newdata(data1,data2,model,S = False,MS = 'Standard'):
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            print(f"The score using {model} after training on Data1 and testing on Data2 is",
            model.score(stdscaler.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
            return(model.score(stdscaler.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            model.score(scalermin.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
            return(model.score(scalermin.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
    else:
        model.fit(data1.iloc[:,1:],data1.iloc[:,0].astype('category'))
        print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)
        return(model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)



def XGBoost_testing(data1,data2,model):
        model.fit(np.array(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
        print(f"\nThe score using {model}  after training on Data1 and testing on Data2 is\n",
        model.score(np.array(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)

def confumatrix_XGB(data1,data2,model):
    ##Confusion Matrix
    model.fit(np.asarray(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
    cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(np.asarray(data2.iloc[:,1:])))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(np.asarray(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'),model)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)

def ploty(xcomp,ycomp,model,S = False,MS= 'Standard'):
    if S == True:
        if MS == 'Standard':
            plt.plot(model.predict(stdscaler.fit_transform(xcomp)),label="Predicted_y")
        elif MS == 'Minmax':
            plt.plot(model.predict(scalermin.fit_transform(xcomp)),label="Predicted_y")
    else:
        plt.plot(model.predict(xcomp),label="Predicted_y")
    plt.plot(np.array(ycomp),label = "True_y")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.show()

##Combined dataframe Confusion matrix
def confumatrixCDF(data1,data2,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(stdscaler.fit_transform(data2.iloc[:,1:])))
            # print(f"The score using {model} after training on Data1 and testing on Data2 is",
            # model.score(stdscaler.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(scalermin.fit_transform(data2.iloc[:,1:])))
            # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            # model.score(scalermin.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
    else:
        model.fit(data1.iloc[:,1:],data1.iloc[:,0].astype('category'))
        # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        # model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)
        cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(data2.iloc[:,1:]))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(data2.iloc[:,1:],data2.iloc[:,0].astype('category'),model,S,MS)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)

##Combined dataframe Confusion matrix
def confumatrixCDF_NN(data1,data2,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(stdscaler.fit_transform(data2.iloc[:,1:])))
            # print(f"The score using {model} after training on Data1 and testing on Data2 is",
            # model.score(stdscaler.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1.iloc[:,1:]),data1.iloc[:,0].astype('category'))
            cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(scalermin.fit_transform(data2.iloc[:,1:])))
            # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            # model.score(scalermin.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
    else:
        model.fit(data1.iloc[:,1:].astype(float),data1.iloc[:,0].astype('category'))
        # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        # model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)
        cm = confusion_matrix(data2.iloc[:,0].astype('category'), model.predict(data2.iloc[:,1:]))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(data2.iloc[:,1:].astype(float),data2.iloc[:,0].astype('category'),model,S,MS)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)

def XYploty(xcomp,ycomp,model):
    plt.plot(model.predict_classes(xcomp),label="Predicted_y")
    plt.plot(np.array(ycomp),label = "True_y")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.show()
##Combined dataframe Confusion matrix - X,ylabels
def XYconfumatrixCDF_NN(X,y,model):
    cm = confusion_matrix(y, model.predict_classes(X))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    XYploty(X,y,model)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)


def XYconfumatrixCDF_ALLModels(X_train, X_test, y_train, y_test,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(X_train),y_train)
            cm = confusion_matrix(y_test, model.predict(stdscaler.fit_transform(X_test)))
            # print(f"The score using {model} after training on Data1 and testing on Data2 is",
            # model.score(stdscaler.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(X_train),y_train)
            cm = confusion_matrix(y_test, model.predict(scalermin.fit_transform(X_test)))
            # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
            # model.score(scalermin.fit_transform(data2.iloc[:,1:]),data2.iloc[:,0].astype('category'))*100)
    else:
        model.fit(X_train,y_train)
        # print(f"The score using {model}  after training on Data1 and testing on Data2 is",
        # model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)
        cm = confusion_matrix(y_test, model.predict(X_test))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(X_test,y_test,model,S,MS)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)

def XYXGBoost_testing(X_train, X_test, y_train, y_test,model):
        model.fit(np.array(X_train),y_train)
        print(f"\nThe score using {model} after training on Data1 and testing on Data2 is\n",
        model.score(np.array(X_test),y_test)*100)

def XYXGBoost_confumatrix(X_train, X_test, y_train, y_test,model):
    ##Confusion Matrix
    model.fit(np.asarray(X_train),y_train)
    cm = confusion_matrix(y_test, model.predict(np.asarray(X_test)))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(np.asarray(X_test),y_test,model)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)
