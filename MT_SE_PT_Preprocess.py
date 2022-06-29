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
import warnings

########Datat Retreieval

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

###Combined Dataframe
def Combine_data(df):
    df_c = df.copy(deep=True)
    df_c.columns = df_c.columns + '_'+df_c.iloc[0,:]+ '_'+df_c.iloc[1,:]
    #drop the date
    SGT750 = df_c.iloc[1:,:]
#     check na values
    nalist  = [x for x in SGT750.isna().sum() if x>0]
    if not(nalist):
        print("No na values")
    else:
    #drop na values
        SGT750 = SGT750.dropna()
    return(SGT750)

def preprocess_indiv_burner_PTET(df):
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
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar','MBM10CP010_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar','MBM10CP015_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar','MBM10CP020_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar','MBM10CP025_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar','MBM10CP030_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar','MBM10CP035_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar','MBM10CP040_XE05_Super UL Frequency Pulsation_mbar',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    return([Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8])


def preprocess_indiv_burner_TET(df):
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
    Burner1 = SGT750.loc[:,['MBM10CQ005_XE01_Burner 1 Flame_on/off','MBM10CT005_XE01_RPL Burner 1 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CT010_XE01_RPL Burner 2 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CT015_XE01_RPL Burner 3 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CT020_XE01_RPL Burner 4 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CT025_XE01_RPL Burner 5 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CT030_XE01_RPL Burner 6 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CT035_XE01_RPL Burner 7 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CT040_XE01_RPL Burner 8 Temperature_°C',
                            'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    return([Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8])

def preprocess_indiv_burner_P(df):
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
                            'MBM10CP005_XE04_High Frequency Pulsation_mbar','MBM10CP005_XE05_Super UL Frequency Pulsation_mbar']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar','MBM10CP010_XE05_Super UL Frequency Pulsation_mbar']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar','MBM10CP015_XE05_Super UL Frequency Pulsation_mbar']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar','MBM10CP020_XE05_Super UL Frequency Pulsation_mbar']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar','MBM10CP025_XE05_Super UL Frequency Pulsation_mbar']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar','MBM10CP030_XE05_Super UL Frequency Pulsation_mbar']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar','MBM10CP035_XE05_Super UL Frequency Pulsation_mbar']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar',
                            'MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar','MBM10CP040_XE05_Super UL Frequency Pulsation_mbar']]
    return([Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8])

####Gettting the dataframe for viewing
# data = viewdf(Gazli20210618Balanced)
# data2 = viewdf(Gazli20210618Unbalanced)


# Data with only Correlated variables

def preprocess_indivburn_TV(df):
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
    Burner1 = SGT750.loc[:,['MBM10CQ005_XE01_Burner 1 Flame_on/off',
                            'MBM10CP005_XE02_Low Frequency Pulsation_mbar','MBM10CP005_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP005_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['MBM10CQ010_XE01_Burner 2 Flame_on/off','MBM10CP010_XE02_Low Frequency Pulsation_mbar','MBM10CP010_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP010_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['MBM10CQ015_XE01_Burner 3 Flame_on/off','MBM10CP015_XE02_Low Frequency Pulsation_mbar','MBM10CP015_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP015_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['MBM10CQ020_XE01_Burner 4 Flame_on/off','MBM10CP020_XE02_Low Frequency Pulsation_mbar','MBM10CP020_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP020_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['MBM10CQ025_XE01_Burner 5 Flame_on/off','MBM10CP025_XE02_Low Frequency Pulsation_mbar','MBM10CP025_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP025_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['MBM10CQ030_XE01_Burner 6 Flame_on/off','MBM10CP030_XE02_Low Frequency Pulsation_mbar','MBM10CP030_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP030_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['MBM10CQ035_XE01_Burner 7 Flame_on/off','MBM10CP035_XE02_Low Frequency Pulsation_mbar','MBM10CP035_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP035_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['MBM10CQ040_XE01_Burner 8 Flame_on/off','MBM10CP040_XE02_Low Frequency Pulsation_mbar','MBM10CP040_XE03_Medium Frequency Pulsation_mbar',
                            'MBM10CP040_XE04_High Frequency Pulsation_mbar','MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C','MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C',
                            'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C']]
    return([Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8])
