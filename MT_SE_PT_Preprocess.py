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
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
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
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off','RPL Burner 1 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','RPL Burner 2 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','RPL Burner 3 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','RPL Burner 4 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','RPL Burner 5 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','RPL Burner 6 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','RPL Burner 7 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','RPL Burner 8 Temperature_°C',
                            'T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
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
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','Ultra Low Frequency Pulsation_mbar',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','Super UL Frequency Pulsation_mbar']]
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
    Burner1 = SGT750.loc[:,['Burner 1 Flame_on/off',
                            'Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner2 = SGT750.loc[:,['Burner 2 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner3 = SGT750.loc[:,['Burner 3 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner4 = SGT750.loc[:,['Burner 4 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner5 = SGT750.loc[:,['Burner 5 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner6 = SGT750.loc[:,['Burner 6 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner7 = SGT750.loc[:,['Burner 7 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    Burner8 = SGT750.loc[:,['Burner 8 Flame_on/off','Low Frequency Pulsation_mbar','Medium Frequency Pulsation_mbar',
                            'High Frequency Pulsation_mbar','T7 Exhaust Temp Average Ring1_°C','T7 Exhaust Temp Average Ring 2_°C',
                            'T7 Exhaust Temp Average Ring 3_°C']]
    return([Burner1,Burner2,Burner3,Burner4,Burner5,Burner6,Burner7,Burner8])
