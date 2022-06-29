### MT_SE_PT_Visualisation
# Import Plotly

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.graph_objects as go
from plotly.subplots import make_subplots

###PLots of FlameDetection device giving 8 values of Flame status
###Pass dataframe (data,data2)

def plotFlameStatus(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )

    fig.update_layout(
    title_text="BurnerPredictionMisclassification"
)
    fig.show()




###PLot to show the dependency of Ultralow Frequency pulsations and Flame status
###Pass dataframe (data,data2)

def plot_burner_Ultralow_FP(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP005_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 1P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP010_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 2P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP015_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 3P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP020_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 4P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP025_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 5P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP030_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 6P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP035_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 7P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP040_XE01_Ultra Low Frequency Pulsation_mbar'], name="Burner 8P"),
    secondary_y=True,
)

    fig.update_layout(
    title_text="BurnerPrediction&UltraLowFreqPulsation"
)
    fig.show()


###PLot to show the dependency of Ultralow Frequency pulsations and Flame status
###Pass dataframe (data,data2)
def plot_burner_Low_FP(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP005_XE02_Low Frequency Pulsation_mbar'], name="Burner 1P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP010_XE02_Low Frequency Pulsation_mbar'], name="Burner 2P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP015_XE02_Low Frequency Pulsation_mbar'], name="Burner 3P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP020_XE02_Low Frequency Pulsation_mbar'], name="Burner 4P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP025_XE02_Low Frequency Pulsation_mbar'], name="Burner 5P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP030_XE02_Low Frequency Pulsation_mbar'], name="Burner 6P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP035_XE02_Low Frequency Pulsation_mbar'], name="Burner 7P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP040_XE02_Low Frequency Pulsation_mbar'], name="Burner 8P"),
    secondary_y=True,
)

    fig.update_layout(
    title_text="BurnerPrediction&LowFreqPulsation"
)
    fig.show()

###PLot to show the dependency of Medium Frequency pulsations and Flame status
###Pass dataframe (data,data2)
def plot_burner_Medium_FP(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP005_XE03_Medium Frequency Pulsation_mbar'], name="Burner 1P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP010_XE03_Medium Frequency Pulsation_mbar'], name="Burner 2P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP015_XE03_Medium Frequency Pulsation_mbar'], name="Burner 3P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP020_XE03_Medium Frequency Pulsation_mbar'], name="Burner 4P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP025_XE03_Medium Frequency Pulsation_mbar'], name="Burner 5P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP030_XE03_Medium Frequency Pulsation_mbar'], name="Burner 6P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP035_XE03_Medium Frequency Pulsation_mbar'], name="Burner 7P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP040_XE03_Medium Frequency Pulsation_mbar'], name="Burner 8P"),
    secondary_y=True,
)

    fig.update_layout(
    title_text="BurnerPrediction&MediumFreqPulsation"
)
    fig.show()
###PLot to show the dependency of High Frequency pulsations and Flame status
###Pass dataframe (data,data2)
def plot_burner_High_FP(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP005_XE04_High Frequency Pulsation_mbar'], name="Burner 1P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP010_XE04_High Frequency Pulsation_mbar'], name="Burner 2P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP015_XE04_High Frequency Pulsation_mbar'], name="Burner 3P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP020_XE04_High Frequency Pulsation_mbar'], name="Burner 4P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP025_XE04_High Frequency Pulsation_mbar'], name="Burner 5P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP030_XE04_High Frequency Pulsation_mbar'], name="Burner 6P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP035_XE04_High Frequency Pulsation_mbar'], name="Burner 7P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP040_XE04_High Frequency Pulsation_mbar'], name="Burner 8P"),
    secondary_y=True,
)

    fig.update_layout(
    title_text="BurnerPrediction&HighFreqPulsation"
)
    fig.show()

###PLot to show the dependency of SuperUltralow Frequency pulsations and Flame status
###Pass dataframe (data,data2)
def plot_burner_SULow_FP(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP005_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 1P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP010_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 2P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP015_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 3P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP020_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 4P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP025_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 5P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP030_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 6P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP035_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 7P"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CP040_XE05_Super UL Frequency Pulsation_mbar'], name="Burner 8P"),
    secondary_y=True,
)

    fig.update_layout(
    title_text="BurnerPrediction&SuperUltraLowFreqPulsation"
)
    fig.show()

###PLot to show the dependency of Temperature T7 and Flame status
###Pass dataframe (data,data2)
###This is where we can consider using sequential models

def plot_burner_Temperature_FS_Dependency(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter( y=data.loc[:,'MBM10CQ005_XE01_Burner 1 Flame_on/off'], name="Burner 1"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ010_XE01_Burner 2 Flame_on/off'], name="Burner 2"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ015_XE01_Burner 3 Flame_on/off'], name="Burner 3"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ020_XE01_Burner 4 Flame_on/off'], name="Burner 4"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ025_XE01_Burner 5 Flame_on/off'], name="Burner 5"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ030_XE01_Burner 6 Flame_on/off'], name="Burner 6"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ035_XE01_Burner 7 Flame_on/off'], name="Burner 7"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=data.loc[:,'MBM10CQ040_XE01_Burner 8 Flame_on/off'], name="Burner 8"),
        secondary_y=False,
    )
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBB10FT923_ZE11_T7 Exhaust Temp Average Ring1_°C'], name="T7_Ring1"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBB10FT923_ZE12_T7 Exhaust Temp Average Ring 2_°C'], name="T7_Ring2"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBB10FT923_ZE13_T7 Exhaust Temp Average Ring 3_°C'], name="T7_Ring3"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBB10CP020_XE01_Exhaust Duct Diff Pressure_kPa'], name="Exhaust Duct Diff"),
    secondary_y=False,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT005_XE01_RPL Burner 1 Temperature_°C'], name="Burner 1T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT010_XE01_RPL Burner 2 Temperature_°C'], name="Burner 2T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT015_XE01_RPL Burner 3 Temperature_°C'], name="Burner 3T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT020_XE01_RPL Burner 4 Temperature_°C'], name="Burner 4T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT025_XE01_RPL Burner 5 Temperature_°C'], name="Burner 5T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT030_XE01_RPL Burner 6 Temperature_°C'], name="Burner 6T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT035_XE01_RPL Burner 7 Temperature_°C'], name="Burner 7T"),
    secondary_y=True,
)
    fig.add_trace(
    go.Scatter(y=data.loc[:,'MBM10CT040_XE01_RPL Burner 8 Temperature_°C'], name="Burner 8T"),
    secondary_y=True,
)
    fig.update_layout(
    title_text="BurnerPrediction&SuperUltraLowFreqPulsation"
)
    fig.show()
