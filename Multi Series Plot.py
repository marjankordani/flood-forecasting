import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 1
plt.rcParams["axes.linewidth"]=1
plt.rcParams["xtick.direction"]="in"   
plt.rcParams["ytick.direction"]="in" 
plt.rcParams["xtick.labelsize"]=12
plt.rcParams["ytick.labelsize"]=12
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["font.weight"] = "bold"

ms = [4, 5, 6, 7]
ys = [ 2019, 2020, 2021]
Colors = ['red', 'blueviolet', 'coral', 'hotpink', 'blueviolet', 'coral','gold']
Markers = ['x', 'o', 's', 'v', 'D', '^','h']
Save = True

N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}

# plt.style.use('ggplot')

if Save:
    if not os.path.exists('Results'):
        os.mkdir('Results')

Root = '/'.join(os.getcwd().split('\\')[:-1])

Models = []

s = len(pd.read_csv(Root + '/Linear Fusion/Results/D-DUPAGE/Test Predict.csv', sep=',', header=0, encoding='UTF-8'))

Targets = [i for i in os.listdir(f'{Root}/{Models[0]}/Results') if not i.endswith('.csv')]

for Target in Targets:
    for Year in ys:
        plt.figure(figsize=(9, 4))
        for i, Model in enumerate(Models):
            DF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Test Predict.csv', sep=',', header=0, encoding='UTF-8')
            Y0 = DF['Y'].to_numpy()[:s]
            P0 = DF['P'].to_numpy()[:s]
            D0 = DF['D'].to_list()[:s]
            Y = []
            P = []
            X = []
            D = []
            counter = 0
            for j, d0 in enumerate(D0):
                m = int(d0.split('/')[0])
                y = int(d0.split('/')[2])
                if m in ms and y == Year:
                    Y.append(Y0[j])
                    P.append(P0[j])
                    if len(X) > 0:
                        if m in D[-1]:
                            X[-1].append(counter)
                            D[-1].append(m)
                        else:
                            X.append([counter])
                            D.append([m])
                    else:
                        X.append([counter])
                        D.append([m])
                    counter += 1
            X2 = [sum(j) / len(j) for j in X]
            D2 = [N2Month[j[-1]] for j in D]
            T = np.arange(start=0, stop=counter, step=1)
            plt.scatter(T, P, s=30, c=Colors[i], marker=Markers[i], label=Model)
        plt.plot(T, Y, ls='-', lw=1.2, c='k', label='Target')
        # plt.title(f'Models Prediciton For {Target}')
        plt.xlabel(Year,fontsize=12,fontweight='bold')
        plt.ylabel(r'streamflow ($\frac{ft^3}{s}$)',fontsize=12,fontweight='bold')
        plt.xticks(X2, D2, rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=10)
        if Save:
            plt.savefig(f'Results/{Target}-{Year}.png', dpi=1536, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
