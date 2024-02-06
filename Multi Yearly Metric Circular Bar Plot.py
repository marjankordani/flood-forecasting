import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def NRMSE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    SE = np.power(E, 2)
    MSE = np.mean(SE)
    RMSE = np.power(MSE, 0.5)
    Range = np.max(Y) - np.min(Y)
    nrmse = 100* RMSE / Range
    return nrmse

def NMAE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    AE = np.abs(E)
    MAE = np.mean(AE)
    Range = np.max(Y) - np.min(Y)
    nmae = 100* MAE / Range
    return nmae

def MAPE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    PE = np.divide(E, Y)
    APE = np.abs(PE)
    mape = 100* np.mean(APE)
    return mape

def R2(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    SE = np.power(E, 2)
    MSE = np.mean(SE)
    Variance = np.var(Y)
    r2 = 1 - MSE / Variance
    return r2

def R(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    SE = np.power(E, 2)
    MSE = np.mean(SE)
    Variance = np.var(Y)
    R2 = 1 - MSE / Variance
    r = 100* np.power(R2, 0.5)
    return r

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 2
plt.rcParams["axes.linewidth"]=2
plt.rcParams["xtick.direction"]="in"   
plt.rcParams["ytick.direction"]="in" 
plt.rcParams["xtick.labelsize"]=10
plt.rcParams["ytick.labelsize"]=10
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["font.weight"] = "bold"

ys = [ 2019, 2020, 2021]
Metrics = ['NRMSE', 'MAPE', 'R2', 'NMAE','R']
Colors = ['red', 'purple', 'pink','powderblue']
Save = True

ys = np.array(ys)

# plt.style.use('ggplot')

Metric2Function = {i: j for i, j in zip(Metrics, [NRMSE, MAPE, R2, NMAE,R])}

if Save:
    if not os.path.exists('Results'):
        os.mkdir('Results')

Root = '/'.join(os.getcwd().split('\\')[:-1])

s = len(pd.read_csv(Root + '/Linear Fusion/Results/D-DUPAGE/Test Predict.csv', sep=',', header=0, encoding='UTF-8'))

Models = ['Linear Fusion','biLSTM-Dense','Ensemble LSTM-GRU-Dense','Ensemble LSTM-Dense']

Targets = [i for i in os.listdir(f'{Root}/{Models[0]}/Results') if not i.endswith('.csv')]

nYear = len(ys)
nModel = len(Models)
nMetric = len(Metrics)
W = 2 * np.pi / (nYear * nModel + nYear)
xs0 = np.linspace(start=0, stop=2 * np.pi, num=(nYear * nModel + nYear), endpoint=False) + 2.8 * W

xs = np.zeros((nYear, nModel))
i = 0
for j in range(nYear):
    for k in range(nModel):
        xs[j, k] = xs0[i]
        i += 1
    i += 1

TicksX = xs.mean(axis=1)

for Target in Targets:
    for Metric in Metrics:
        MetricValues = {Model: [] for Model in Models}
        Figure, Axes = plt.subplots(figsize=(9, 6), subplot_kw={'projection': 'polar'})
        for i, Model in enumerate(Models):
            DF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Test Predict.csv', sep=',', header=0, encoding='UTF-8')
            Y0 = DF['Y'].to_numpy()[:s]
            P0 = DF['P'].to_numpy()[:s]
            D0 = DF['D'].to_list()[:s]
            Y = {Year: [] for Year in ys}
            P = {Year: [] for Year in ys}
            for y0, p0, d0 in zip(Y0, P0, D0):
                y = int(d0.split('/')[2])
                if y in ys:
                    Y[y].append(y0)
                    P[y].append(p0)
            for Year in ys:
                MetricValues[Model].append(Metric2Function[Metric](Y[Year], P[Year]))
            Axes.bar(xs[:, i], MetricValues[Model], width=W, color=Colors[i], alpha=0.7, label=Model)
        Axes.set_frame_on(False)
        Axes.xaxis.grid(False)
        Axes.yaxis.grid(False)
        # Axes.set_xlabel('Year',fontsize=10)
        Axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
        Axes.set_xticks(TicksX, ys)
        if Metric != 'R2':
            Axes.set_title(f' {Metric} (%)',fontsize=16,fontweight='bold')
        else:
            Axes.set_title(f' $R^2$ ',fontsize=16,fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12)
        if Save:
            plt.savefig(f'Results/{Target}-{Metric}.png', dpi=1536, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
