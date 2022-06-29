## Importing of dataframe libraries
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

scalermin =  MinMaxScaler()
stdscaler = StandardScaler()
###LOC values
def WholeTurbine(df):
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
    return(SGT750)

def preprocessPT(data):
    datay = data.filter(regex='Flame_on/off')
    # datax = data.filter(regex="^(?!$)")
    # datax = data[data.columns.drop(list(data.filter(regex='Burner')))]
    datax = data.filter(regex='Frequency Pulsation_mbar')
    datax['ExhaustTemp_Ring1'] = data['Exhaust Temp Average Ring1_°C']
    datax['ExhaustTemp_Ring2'] = data['Exhaust Temp Average Ring 2_°C']
    datax['ExhaustTemp_Ring3'] = data['Exhaust Temp Average Ring 3_°C']
    # datax1 = datax[datax.columns.drop(list(datax.filter(regex='NaN')))]
    # datax1 = data.filter(regex='Ring')
    return(datay,datax)

def preprocess_pressure(data):
    datay = data.filter(regex='Flame_on/off')
    # datax = data.filter(regex="^(?!$)")
    # datax = data[data.columns.drop(list(data.filter(regex='Burner')))]
    datax = data.filter(regex='Frequency Pulsation_mbar')
    # datax1 = datax[datax.columns.drop(list(datax.filter(regex='NaN')))]
    # datax1 = data.filter(regex='Ring')
    return(datay,datax)

def newY(ycomp):
    newY = []
    for i in range(ycomp.shape[0]):
        if sum(ycomp.iloc[i,:])>5:
            newY.append(1)
        else:
            newY.append(0)
    newDY = pd.DataFrame (newY, columns = ['Y_Gold'])
    return(newDY)


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

def confumatrix(data1,data2,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    ycomp = newY(data2[0])
    if S == True:
        if MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            xcomp = scalermin.fit_transform(data2[1])
        elif MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            xcomp = stdscaler.fit_transform(data2[1])
    else:
        model.fit(data1[1],newY(data1[0]).values.ravel())
        xcomp = data2[1]
    cm = confusion_matrix(ycomp.values.ravel(), model.predict(xcomp))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(xcomp,ycomp,model)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)

def ploty(xcomp,ycomp,model):
    plt.plot(model.predict(xcomp),label="Predicted_y")
    plt.plot(np.array(ycomp),label = "True_y")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)
    plt.show()
# for i, (key, classifier) in enumerate(ConfusionmatrixTrainHighTestLow.items()):
#     print(classifier[0])
# f, axes = plt.subplots(1, 3, figsize=(20, 5), sharey='row')

# for i, (key, classifier) in enumerate(ConfusionmatrixTrainHighTestLow.items()):
#     disp = sns.heatmap(classifier[0], annot=True, fmt='g')
#     ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
#     ax.set_title(key);
#     ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
# #     if i!=0:
# #         disp.ax_.set_ylabel('')
# # f.text(0.4, 0.1, 'Predicted label', ha='left')
# # plt.subplots_adjust(wspace=0.40, hspace=0.1)


# # f.colorbar(disp.im_, ax=axes)
# plt.show()

def confumatrix_xgb(data1,data2,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    ycomp = newY(data2[0])
    if S == True:
        if MS == 'Minmax':
            model.fit(scalermin.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            xcomp = scalermin.fit_transform(data2[1])
        elif MS == 'Standard':
            model.fit(stdscaler.fit_transform(data1[1]),newY(data1[0]).values.ravel())
            xcomp = stdscaler.fit_transform(data2[1])
    else:
        model.fit(np.asarray(data1[1]),newY(data1[0]).values.ravel())
        xcomp = data2[1]
    cm = confusion_matrix(ycomp.values.ravel(), model.predict(np.asarray(xcomp)))
    tn, fp, fn, tp = cm.ravel()
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
    plt.show()
    ploty(np.asarray(xcomp),ycomp,model)
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    return(cm,tn, fp, fn, tp)
