import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def NRMSE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    SE = np.power(E, 2)
    MSE = np.mean(SE)
    RMSE = np.power(MSE, 0.5)
    Range = np.max(Y) - np.min(Y)
    nrmse =100* RMSE / Range
    return nrmse

def NMAE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    AE = np.abs(E)
    MAE = np.mean(AE)
    Range = np.max(Y) - np.min(Y)
    nmae =100* MAE / Range
    return nmae

def MAPE(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    PE = np.divide(E, Y)
    APE = np.abs(PE)
    mape =100* np.mean(APE)
    return mape

def R2(Y:np.ndarray, P:np.ndarray) -> float:
    E = np.subtract(Y, P)
    SE = np.power(E, 2)
    MSE = np.mean(SE)
    Variance = np.var(Y)
    r2 = 1 - MSE / Variance
    return r2



ys = [2018, 2019, 2020, 2021]
Metrics = ['    NRMSE(%)', 'MAPE(%)', 'R2', 'NMAE(%)']
Colors = ['red', 'blue', 'green', 'hotpink', 'blueviolet', 'coral','gold']
Markers = ['x', 'o', 's', 'v', 'D', '^','h']
Save = True

plt.style.use('ggplot')

Metric2Function = {i: j for i, j in zip(Metrics, [NRMSE, MAPE, R2, NMAE])}

if Save:
    if not os.path.exists('Results'):
        os.mkdir('Results')

Root = '/'.join(os.getcwd().split('\\')[:-1])

Models = []

Targets = [i for i in os.listdir(f'{Root}/{Models[0]}/Results') if not i.endswith('.csv')]

xMetrics = [i if i != 'R2' else '$R^2$' for i in [Metrics[i] if i != 0 else 7 * ' ' + Metrics[i] for i in range(len(Metrics))]]

for Target in Targets:
    Data = np.zeros((len(Models), len(Metrics)))
    for j, Metric in enumerate(Metrics):
        for i, Model in enumerate(Models):
            DF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Total Predict.csv', sep=',', header=0, encoding='UTF-8')
            Y0 = DF['Y'].to_numpy()
            P0 = DF['P'].to_numpy()
            D0 = DF['D'].to_list()
            Y = []
            P = []
            for y0, p0, d0 in zip(Y0, P0, D0):
                y = int(d0.split('/')[2])
                if y in ys:
                    Y.append(y0)
                    P.append(p0)
            Data[i, j] = Metric2Function[Metric](Y, P)
    X = np.linspace(start=0, stop=2 * np.pi, num=len(Metrics) + 1)[:-1]
    plt.figure(figsize=(6, 6))
    plt.subplot(polar=True)
    for i in range(len(Models)):
        a = np.hstack((X, X[0]))
        b = np.hstack((Data[i, :], Data[i, 0]))
        plt.plot(a, b, ls='-', lw=0.9, label=Models[i], c=Colors[i])
    plt.thetagrids(np.degrees(X), labels=xMetrics , fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=8)
    if Save:
        plt.savefig(f'Results/{Target}.png', dpi=1536, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
