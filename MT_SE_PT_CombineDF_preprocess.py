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

def preprocess(data):
    datay = data.filter(regex='Flame_on/off')
    # datax = data.filter(regex="^(?!$)")
    # datax = data[data.columns.drop(list(data.filter(regex='Burner')))]
    datax = data.filter(regex='Frequency Pulsation_mbar')
    datax['ExhaustTemp_Ring1'] = data['MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C']
    datax['ExhaustTemp_Ring2'] = data['MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C']
    datax['ExhaustTemp_Ring3'] = data['MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']
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

def preprocess_combineburner_TP(df):
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
    Burner1 = SGT750.loc[:,['MBM10CQ005_XE01_Burner 1 Flame_on/off','MBM10CP005_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP005_XE02_Low Frequency Pulsation_mbar','MBM10CP005_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP005_XE04_High Frequency Pulsation_mbar','MBM10CP005_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar','MBM10CP010_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar','MBM10CP015_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar','MBM10CP020_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar','MBM10CP025_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar','MBM10CP030_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar','MBM10CP035_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar','MBM10CP040_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    datalist = [Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8]
    dfcols = ['BurnerStatus','UltraLow_Frequency','Low_Frequency','Medium_Frequency','High_Frequency','SuperUltraLow_Frequency','TempRing1','TempRing2','TempRing3']
    for dc in datalist:
        dc.columns = dfcols
    stackdf = pd.concat(datalist).reset_index(drop=True)
    return(stackdf)


## Remove SuperUltraLow_Frequency
def preprocess_combineburner_TP_sans_SP(df):
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
    Burner1 = SGT750.loc[:,['MBM10CQ005_XE01_Burner 1 Flame_on/off','MBM10CP005_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP005_XE02_Low Frequency Pulsation_mbar','MBM10CP005_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP005_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    datalist = [Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8]
    dfcols = ['BurnerStatus','UltraLow_Frequency','Low_Frequency','Medium_Frequency','High_Frequency','TempRing1','TempRing2','TempRing3']
    for dc in datalist:
        dc.columns = dfcols
    stackdf = pd.concat(datalist).reset_index(drop=True)
    return(stackdf)


def preprocess_burnerlst_TP_Time(df):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.columns + '_'+df_c.iloc[0,:]+ '_'+df_c.iloc[1,:]
    #drop the date
    SGT750 = df_c.iloc[2:,:]
    #Change the index to sequence
    # SGT750.index = range(1,SGT750.shape[0]+1)
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    Burner1 = SGT750.loc[:,['MBM10CQ005_XE01_Burner 1 Flame_on/off','MBM10CP005_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP005_XE02_Low Frequency Pulsation_mbar','MBM10CP005_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP005_XE04_High Frequency Pulsation_mbar','MBM10CP005_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar','MBM10CP010_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar','MBM10CP015_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar','MBM10CP020_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar','MBM10CP025_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar','MBM10CP030_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar','MBM10CP035_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar','MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar','MBM10CP040_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    datalist = [Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8]
    return(datalist)

def combiner(datalist):
    dfcols = ['BurnerStatus','UltraLow_Frequency','Low_Frequency','Medium_Frequency','High_Frequency','SuperUltraLow_Frequency','TempRing1','TempRing2','TempRing3']
    for dc in datalist:
        dc.columns = dfcols
    newdf = pd.concat(datalist).reset_index(drop=True)
    return(newdf)
