###Checks the heatmap for two of the data out of which one is pressure and the other is all the temperature values
###Dataframe should be preprocessed to have either Exhaust temperature values or should have all the pressure values and burner values of
###each of the 8 burners

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

###Correlation Function
from scipy.stats import pearsonr
from scipy.stats import spearmanr

###Find the correlation for all the burners and make subplots
def heatmap_burner_pressure(data,method = 'pearson'):

    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)

    fig.subplots_adjust(wspace=0.1)

    co = 0

    for i in range(2):
        for j in range(4):
            corrmatrix = data[co].astype('float64').corr(method = c(method))
            corrmatrix.columns = ["Burner Flame_on/off", "Ultra Low Frequency Pulsation_mbar",
                                  "Low Frequency Pulsation_mbar", "Medium Frequency Pulsation_mbar",
                                  "High Frequency Pulsation_mbar", "Super UL Frequency Pulsation_mbar"]
            corrmatrix.index = ["Burner Flame_on/off", "Ultra Low Frequency Pulsation_mbar",
                                  "Low Frequency Pulsation_mbar", "Medium Frequency Pulsation_mbar",
                                  "High Frequency Pulsation_mbar", "Super UL Frequency Pulsation_mbar"]

    #         plt.xlabel(f"Colors")
            plt.ylabel(f"Burner{co}")
            co = co + 1
            sns.heatmap(corrmatrix, ax = ax[i,j], yticklabels=True, xticklabels=True,annot = True)
# heatmap_burner_pressure(dataP)

#Correation for the temperature variables

def heatmap_burner_temperature(data,method):

    fig, ax = plt.subplots(nrows=2,ncols=4, figsize=(15,10),sharey=True,sharex=True)
    fig.subplots_adjust(wspace=0.1)
    co = 0
    for i in range(2):
        for j in range(4):
            corrmatrix = data[co].astype('float64').corr(method = method)
            corrmatrix.columns = ["Burner Flame_on/off", "RPL Burner Temperature_°C",
                                  "T7 Exhaust Temp Average Ring1_°C", "T7 Exhaust Temp Average Ring 2_°C",
                                  "T7 Exhaust Temp Average Ring 3_°C"]
            corrmatrix.index = ["Burner Flame_on/off", "RPL Burner Temperature_°C",
                                  "T7 Exhaust Temp Average Ring1_°C", "T7 Exhaust Temp Average Ring 2_°C",
                                  "T7 Exhaust Temp Average Ring 3_°C"]
    #         plt.xlabel(f"Colors")
            plt.ylabel(f"Burner{co}")
            co = co + 1
            ax.title(co)
            sns.heatmap(corrmatrix, ax = ax[i,j], yticklabels=True, xticklabels=True,annot = True)
# heatmap_burner_temperature(dataT)
