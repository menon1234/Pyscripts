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
###Pass the combined data
def heatmap_burner(data):
        corrmatrix = data.astype('float64').corr()
        mask = np.zeros_like(corrmatrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)]= True
        sns.heatmap(corrmatrix,mask = mask, yticklabels=True, xticklabels=True,annot = True)
# heatmap_burner_pressure(dataP)
