#### Functions 
#WholeTurbinePreprocess                                                            - Argument - Raw csv file Returns X and Y. Preprocess dataset joining rows removing na values.
## TurbineXandY(preprocesseddf):
## valuecounts_allburners                                                          - pass list of 8 burners as argument 
####Getting the On/Off switching values from each burner
##Condition:Trained on Wholedf so it takes all the values of all the burners
##Indexing not done so it takes the First 8 columns and finds difference

## valueCount_stackedburners -                                                     -pass single stacked dataset as argument 
####Getting the On/Off switching values from each burner
##Condition:Trained on Wholedf so it takes all the values of all the burners
##Indexing not done so it takes the First 8 columns and finds difference
##def TrueandFalseY_CheckIndex(dataTrain,dataTest,model,S = False,MS = 'Standard')
##def Model_eval(data,split,model,S = False,MS = 'Standard'):                       -Evaluation of model
##def Model_training(model,data):                                                   -Returns the trained model
##def preprocess_combineburner_T_P_sans_SP(df,rm = False):

## Importing of dataframe libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score,matthews_corrcoef
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
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin

##Normal preprocess of the whole Turbine
def WholeTurbinePreprocess(df,rm = False):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.iloc[0,:]
    #drop the date
    SGT750 = df_c.iloc[2:,:]
    #Change the index to sequence
    SGT750.index = range(1,SGT750.shape[0]+1)
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    if rm:
        SGT750 = SGT750[(SGT750.sum(axis=1)==0) | (SGT750.sum(axis=1)==8)]
    else:
        pass
    return(SGT750)
from sklearn.metrics import f1_score
def ModelthresholdTesting(trained_model,X_test,y_test,threshold = None):
    y_pred = prob_predict(trained_model,np.asarray(X_test),threshold)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
    return(FNRFPR(y_test,y_pred),fn,fp, f1_score(y_test, y_pred, average='weighted'))

## Remove SuperUltraLow_Frequency from the dataset
##SP-SuperUltraLowFreqPulsation
def preprocess_Indiv_burner_T_P_sans_SULP(df,rm = False,form = 2):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.columns + '_'+df_c.iloc[0,:]+ '_'+df_c.iloc[1,:]
    #drop the date
    SGT750 = df_c.iloc[2:,:]
    #Change the index to sequence
    SGT750.index = range(1,SGT750.shape[0]+1)
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    if rm:
        SGT750 = SGT750[(SGT750.filter(regex = 'Flame_on/off').sum(axis=1)==0) | (SGT750.filter(regex = 'Flame_on/off').sum(axis=1)==8)]
    else:
        pass    
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP005_XE03_Medium Frequency Pulsation_mbar',
'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C','T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','Ultra Low Frequency Pulsation_mbar','MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    datalist = [Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8]
    dfcols = ['BurnerStatus','UltraLow_Frequency','Low_Frequency','Medium_Frequency','High_Frequency','TempRing1','TempRing2','TempRing3']
    
    for dc in datalist:
        dc.columns = dfcols
    stackdf = pd.DataFrame(pd.concat(datalist).reset_index(drop=True))
    if form == 1:
        return(datalist)
    elif form == 2:
        return(stackdf)
    ###combiner Fucntion

def prob_predict(model,X, threshold=None):
    if threshold == None: # If no threshold passed in, simply call the base class predict, effectively threshold=0.5
        return(model.predict(X))
    else:
        y_scores = model.predict_proba(X)[:, 1]
        y_pred_with_threshold = (y_scores >= threshold).astype(int)
        return(y_pred_with_threshold)
    
##PLot confusiongrid
###returns number of FP FN ,fpr list and tprlist
def plotconfusiongrid(data,model,threshold):
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)
    fig.subplots_adjust(wspace=0.1)
    co = 0
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    f1score = []
    mcc = []
    for i in range(2):
        for j in range(4):
            y_true = np.array(data[co].iloc[:,0].astype('category'))
            y_pred = prob_predict(model,np.array(data[co].iloc[:,1:]),threshold)
            cm = confusion_matrix(y_true,y_pred)
            f1score.append(f1_score(y_true,y_pred,average = 'weighted'))
            mcc.append(matthews_corrcoef(y_true,y_pred))
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
    return(f1score,mcc)
    
   
##PLot confusiongrid_cost
###returns number of FP FN ,fpr list and tprlist
def plotconfusiongrid_cost(data,model,cost = False):
    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)
    fig.subplots_adjust(wspace=0.1)
    co = 0
    fplist = []
    fnlist = []
    tprlist = []
    fprlist = []
    cost_matrix =   np.array([[.15,.5],
     [.3,.15]])
    for i in range(2):
        for j in range(4):
            if cost == True:
                cm = confusion_matrix(data[co].iloc[:,0].astype('category'),
                              model.predict(data[co].iloc[:,1:]))*cost_matrix
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
    plt.show()
    # return(sum(fplist),sum(fnlist),sum(fprlist)+sum(tprlist))
    
    
###Retrive X and Y
##Returns x and y as a list
def TurbineXandY(preprocesseddf):
    y = preprocesseddf.filter(regex = 'BurnerStatus')
    X = preprocesseddf.drop(y,axis = 1)
    return([X,y])


def TurbineXandYStacked(preprocesseddf):
    y = preprocesseddf.filter(regex = 'BurnerStatus') 
    X = np.asarray(preprocesseddf.drop(y,axis = 1))
    return([X,y])


###Feature reduction

# def top_features(df):

####Getting the On/Off switching values from each burner
##Condition:Trained on Wholedf so it takes all the values of all the burners
##Indexing not done so it takes the First 8 columns and finds difference
##This has to be run over the dataset dfbalanced and then we pass it to the valuecounts_allburners
# BurnerXYList = {}
# for i in range(8):
#     BurnerXYList[i]=TurbineXandY(dfBalanced[i])
def valuecounts_allburners(df):
    count_dict = {}
    for i in range(8):
        Values = df[i][1].diff().shift(-1).dropna().value_counts().tolist()[1:]
        count_dict[i]  = sum(Values)
    print(f'The total number of On-Off is{Values[0]},\n The total number of Off-On is {Values[1]}')
    return(count_dict)

##Pass the Burnerstatus column of the combined dataframe
##Returns a single value of the number of switching for the turbine with all the burners valuecounts_stacked
#  def valuecounts_stacked(df):
#         Values = df.diff().dropna().value_counts().tolist()[1:]
#         return(count_dict)

def valuecounts_stackedburner(df):
    values = df.loc[:,'BurnerStatus'].diff().shift(-1).dropna().value_counts().tolist()[1:]
    print(f'The total number of On-Off is {values[0]}')
    print(f'The total number of Off-On is {values[1]}')
    return(values)


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

def XGBoostTuning(dataTrain,model):
    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }
    
##Return False Y and True Y:
def TrueandFalseY_CheckIndex(dataTrain,dataTest,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(dataTrain.iloc[:,1:]),dataTrain.iloc[:,0].astype('category'))
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(dataTrain.iloc[:,1:]),dataTrain.iloc[:,0].astype('category'))
    else:
        model.fit(np.array(dataTrain.iloc[:,1:]),dataTrain.iloc[:,0].astype('category'))
    TrueFalse = {'TrueY':dataTest.iloc[:,0].astype('category'),'FalseY':model.predict(np.array(dataTest.iloc[:,1:]))}
    diffvalTrue = list(np.diff(TrueFalse['TrueY']))
    diffvalFalse = list(np.diff(TrueFalse['FalseY']))
    TrueValues = {}
    for i in range(len(diffvalTrue)):
        if diffvalTrue[i]==1:
            TrueValues['Off/On']=diffvalTrue.index(diffvalTrue[i])
        if diffvalTrue[i]==-1:
            TrueValues['On/Off']=diffvalTrue.index(diffvalTrue[i])
    PredValues = {}
    for i in range(len(diffvalFalse)):
        if diffvalFalse[i]==1:
            PredValues['Off/On']=diffvalFalse.index(diffvalFalse[i])
        if diffvalFalse[i]==-1:
            PredValues['On/Off']=diffvalFalse.index(diffvalFalse[i])
    return([TrueValues,PredValues])

scalermin =  MinMaxScaler()
stdscaler = StandardScaler()

def Model_eval(data,split,model,S = False,MS = 'Standard'):
    X = np.array(data.iloc[:,1:])
    y = data.iloc[:,0].astype('category')
    if S == True:
        if MS == 'Standard':
            X = stdscaler.fit_transform(X)
        elif MS == 'Minmax':
            X = scalermin.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=42)
    model.fit(X_train,y_train)
    print(f"The score using {model} after training on 70% of the data and testing on Validation data is",
              model.score(X_test,y_test)*100)
    return(model.score(X_test,y_test)*100)


def FNRFPR(ytrue,ypred):
    tn,fn,fp,tp = confusion_matrix(ytrue,ypred,labels = [0,1]).ravel()
    fnr = fn/(fn+tp)
    fpr = fp/(fp+tn)
    return((0.9*fpr)+fnr)


##Combined dataframe Confusion matrix
def confumatrixCDF(X_train,y_train,X_test,y_test,model,S = False,MS = 'Standard'):
    ##Confusion Matrix
    X_train = np.array(X_train)
    y_train = y_train.astype('category')
    X_test = np.array(X_test)
    y_test = y_test.astype('category')
    if S == True:
        if MS == 'Standard':
            model.fit(stdscaler.fit_transform(X_train),y_train)
            y_pred = model.predict(stdscaler.fit_transform(X_test))
            cm = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
        elif MS == 'Minmax':
            model.fit(scalermin.fit_transform(X_train),y_train)
            y_pred = model.predict(scalermin.fit_transform(X_test))
            cm = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        model.fit(X_train,y_train)
        # print(f"The score using {model}  after training on Data1 and tesy_trainting on Data2 is",
        # model.score(data2.iloc[:,1:],data2.iloc[:,0].astype('category'))*100)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred )
        f1 = f1_score(y_test, y_pred, average='weighted')
    tn, fp, fn, tp = cm.ravel()
    # ax= plt.subplots(1,2,1)
    fig, ax = plt.subplots(1, 2,figsize=(13,5))
    fig.tight_layout()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax[0]);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax[0].set_xlabel('Predicted labels')
    ax[0].set_ylabel('True labels');
    ax[0].set_title('Confusion Matrix');
    ax[0].xaxis.set_ticklabels(['0', '1'])
    ax[0].yaxis.set_ticklabels(['0', '1'])
    # plt.subplot(1, 2, 2)
    ploty(ax[1],X_test,y_test,model,S,MS)
    plt.tight_layout()
    plt.show()
    fn,fp,tn,tp = int(fn),int(fp),int(tn),int(tp) 
    print("FN :",fn,"\nFP :",fp,"\nTN :",tn,"\nTP :",tp)
    MCC = ((tp*tn) - (fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return(tn, fp, fn, tp,MCC,f1)
    
def ploty(ax,xcomp,ycomp,model,S = False,MS= 'Standard'):
    if S == True:
        if MS == 'Standard':
            ax.plot(model.predict(stdscaler.fit_transform(xcomp)),label="Predicted_y")
        elif MS == 'Minmax':
            ax.plot(model.predict(scalermin.fit_transform(xcomp)),label="Predicted_y")
    else:
        ax.plot(model.predict(xcomp),label="Predicted_y")
    ax.plot(np.array(ycomp),label = "True_y")
    ax.legend(loc="upper left")
    # ax.ylim(0, 1)
    # plt.show()
    
    
def Model_training(Xtrain,ytrain,model):
        Xtrain = np.array(Xtrain)
        ytrain = ytrain.astype('category')
        model.fit(Xtrain,ytrain)
        return(model)
    
def preprocess_combineburner_T_P_sans_SP(df,rm = False,form = 2):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.columns + '_'+df_c.iloc[0,:]+ '_'+df_c.iloc[1,:]
    #drop the date
    SGT750 = df_c.iloc[2:,:]
    #Change the index to sequence
    SGT750.index = range(1,SGT750.shape[0]+1)
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    if rm:
        SGT750 = SGT750[(SGT750.filter(regex = 'Flame_on/off').sum(axis=1)==0) | (SGT750.filter(regex = 'Flame_on/off').sum(axis=1)==8)]
    else:
        pass
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar','High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C','T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','Ultra Low Frequency Pulsation_mbar','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    datalist = [Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8]
    ###combiner Fucntion
    dfcols = ['BurnerStatus','UltraLow_Frequency','Low_Frequency','Medium_Frequency','High_Frequency','TempRing1','TempRing2','TempRing3']
    for dc in datalist:
        dc.columns = dfcols
    stackdf = pd.concat(datalist).reset_index(drop=True)
    if form == 1:
        return(datalist)
    elif form == 2:
        return(stackdf)

def plotconfusiongridXGB(data,model):
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
                              model.predict(data[co].iloc[:,1:].astype('float')))
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
    # return(sum(fplist),sum(fnlist),sum(fprlist)+sum(tprlist))
    
    
def position_ONOFF(yval): ##The y axis 
    offondic = {}
    onoffdic = {}
    for i,j in enumerate(np.diff(yval)):
        if (j==1):
            offondic[i] = j
        elif(j==-1):
            onoffdic[i] = j
    pos_ON = list()        
    for i in offondic.keys():
        pos_ON.append(i)
    pos_OFF = list()
    for i in onoffdic.keys():
        pos_OFF.append(i)
    return(pos_ON,pos_OFF)
    
def Timedifference(pred_y,true_y):
    Predvals = position_ONOFF(pred_y)
    Truevals = position_ONOFF(true_y)
    OfftoOndifference = []
    OntoOffdifference = []
    if len(Predvals[0])==len(Truevals[0]):
        OfftoOnlist = zip(Predvals[0], Truevals[0])
        OntoOfflist = zip(Predvals[1], Truevals[1])
        for list1_i, list2_i in OfftoOnlist:
            OfftoOndifference.append(list1_i-list2_i)    
        for list1_i, list2_i in OntoOfflist:
            OntoOffdifference.append(list1_i-list2_i)
        return(OfftoOndifference,OntoOffdifference)
    else:
        print('Inconsistensies in the classification')
        print(f'The number of Off to On inconsistensies are {len(Predvals[0])-len(Truevals[0])}')
        print(f'The number of On to Off inconsistensies are {len(Predvals[1])-len(Truevals[1])}')
        return(Predvals)
