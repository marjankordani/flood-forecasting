import os as os # Operating System
import numpy as np # Vectorized Computation
import pandas as pd # Reading And Manipulating Dataset
import random as ran # Generating Random Numbers
import typing as typ # Typing
import scipy.stats as stt # Statistics
import matplotlib.pyplot as plt # Plotting And Visualization
import sklearn.linear_model as lm

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

class DeepMix:
    def __init__(self,
                 Target:str,
                 Save:bool,
                 DPI:int,
                 RandomState=typ.Union[int, None]):
        self.CH = 0 / 100
        self.SS = 0 / 100
        self.Target = Target
        self.Save = Save
        self.DPI = DPI
        self.RandomState = RandomState
        if self.Save:
            if not os.path.exists('Results'):
                os.mkdir('Results')
            if not os.path.exists(f'Results/{self.Target}'):
                os.mkdir(f'Results/{self.Target}')
        self.Name = 'Linear Fusion'
        self.SetRandomState()
    def SetRandomState(self):
        if self.RandomState is not None:
            ran.seed(self.RandomState)
            np.random.seed(self.RandomState)
            os.environ['PYTHONHASHSEED'] = str(self.RandomState)
    def H(self,
          s:str='_',
          n:int=60):
        print(s * n)
    def CreateModel(self):
        self.Model = lm.LinearRegression()
    def Fit(self,
            trX:np.ndarray,
            trY:np.ndarray,
            vaX:np.ndarray,
            vaY:np.ndarray):
        trvaX = np.vstack((trX, vaX))
        trvaY = np.hstack((trY, vaY))
        self.CreateModel()
        self.Model.fit(trvaX, trvaY)
    def RegressionReport(self,
                         Y:np.ndarray,
                         P:np.ndarray,
                         Dataset:str):
        Y = Y.reshape(-1, 1)
        P = P.reshape(-1, 1)
        Range = Y.max() - Y.min()
        Variance = np.var(Y)
        E = np.subtract(Y, P)
        SE = np.power(E, 2)
        MSE = np.mean(SE)
        RMSE = np.power(MSE, 0.5)
        NRMSE = 100 * RMSE / Range
        MBE = np.mean(E)
        AE = np.abs(E)
        MAE = np.mean(AE)
        NMAE = 100 * MAE / Range
        PE = 100 * E / Y
        APE = np.abs(PE)
        MAPE = np.mean(APE)
        R2 = 100 * (1 - MSE / Variance)
        R = 10 * np.power(R2, 0.5)
        PCC = 100 * stt.pearsonr(Y[:, 0], P[:, 0])[0]
        SCC = 100 * stt.spearmanr(Y[:, 0], P[:, 0])[0]
        self.H()
        print(f'Regression Report On {Dataset} Dataset ({self.Target}):')
        print(f'MSE:       {MSE:.4f}')
        print(f'RMSE:      {RMSE:.4f}')
        print(f'NRMSE (%): {NRMSE:.2f}')
        print(f'MBE:       {MBE:.4f}')
        print(f'MAE:       {MAE:.4f}')
        print(f'NMAE (%):  {NMAE:.2f}')
        print(f'MAPE (%):  {MAPE:.2f}')
        print(f'R2 (%):    {R2:.2f}')
        print(f'R (%):     {R:.2f}')
        print(f'PCC (%):   {PCC:.2f}')
        print(f'SCC (%):   {SCC:.2f}')
        self.H()
        D = {'MSE':MSE,
             'RMSE':RMSE,
             'NRMSE':NRMSE,
             'MBE':MBE,
             'MAE':MAE,
             'NMAE':NMAE,
             'MAPE':MAPE,
             'R2':R2,
             'R':R,
             'PCC':PCC,
             'SCC':SCC}
        if self.Save:
            with open(f'Results/{self.Target}/Report-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'Regression Report On {Dataset} Dataset:\n')
                for k, v in D.items():
                    if k in ['MSE', 'RMSE', 'MAE', 'MBE']:
                        t = ' ' * (9 - len(k))
                        F.write(f'{k}:{t} {v:.6f}\n')
                    else:
                        t = ' ' * (5 - len(k))
                        F.write(f'{k} (%):{t} {v:.6f}\n')
    def RegressionPlot(self,
                       Y:np.ndarray,
                       P:np.ndarray,
                       Dataset:str):
        Y = Y.reshape(-1, 1)
        E = np.subtract(Y, P)
        SE = np.power(E, 2)
        PE = 100 * np.divide(E, Y)
        APE = np.abs(PE)
        MSE = np.mean(SE)
        Variance = np.var(Y)
        R2 = 1 - MSE / Variance
        a = min(Y.min(), P.min())
        b = max(Y.max(), P.max())
        ab = np.array([a, b])
        plt.scatter(Y, P, s=12, c=APE, cmap='rainbow', marker='o', alpha=0.8, label='Data')
        plt.plot(ab, ab, ls='-', lw=1.2, c='k', label='Y=X')
        plt.plot(ab, 1.2 * ab, ls='--', lw=1, c='r', label='Y=1.2*X')
        plt.plot(ab, 0.8 * ab, ls='--', lw=1, c='r', label='Y=0.8*X')
        plt.text(a, b, f'$R^2$: {R2:.4f}', fontdict={'size':14})
        plt.title(f' {self.Name} on {Dataset} Dataset', fontsize=13)
        plt.xlabel('Target Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        if self.Save:
            plt.savefig(f'Results/{self.Target}/RegressionPlot-{Dataset}.png', dpi=self.DPI)
            plt.close()
        else:
            plt.show()
    def DoubleRegressionPlot(self,
                             trY:np.ndarray,
                             trP:np.ndarray,
                             teY:np.ndarray,
                             teP:np.ndarray):
        trE = np.subtract(trY, trP)
        teE = np.subtract(teY, teP)
        trSE = np.power(trE, 2)
        teSE = np.power(teE, 2)
        trMSE = np.mean(trSE)
        teMSE = np.mean(teSE)
        trVariance = np.var(trY)
        teVariance = np.var(teY)
        trR2 = 1 - trMSE / trVariance
        teR2 = 1 - teMSE / teVariance
        a = min(min(trY.min(), teY.min()), min(trP.min(), teP.min()))
        b = max(max(trY.max(), teY.max()), max(trP.max(), teP.max()))
        ab = np.array([a, b])
        plt.scatter(trY, trP, s=12, cmap='GnBu', marker='o', alpha=0.8, label='Train')
        plt.scatter(teY, teP, s=12, cmap='OrRd', marker='o', alpha=0.8, label='Test')
        plt.plot(ab, ab, ls='-', lw=1.2, c='k')
        plt.plot(ab, 1.2 * ab, ls='--', lw=1, c='r')
        plt.plot(ab, 0.8 * ab, ls='--', lw=1, c='r')
        plt.text(a, (1.2 - 0.05) * b, f'Train $R^2$: {trR2:.4f}', fontdict={'size':12},fontweight='bold')
        plt.text(a, a + 0.9 * ((1.2 - 0.05) * b - a), f'Test $R^2$: {teR2:.4f}', fontdict={'size':12},fontweight='bold')
        plt.text(a + (0.8 - 0.05) * (b - a), (1.2 + 0.05) * (a + 0.8 * (b - a)), '+20%', fontdict={'size':8},fontweight='bold')
        plt.text(a + (0.8 + 0.05) * (b - a), (0.8 - 0.05) * (a + 0.8 * (b - a)), '-20%', fontdict={'size':8},fontweight='bold')
        plt.title(f'{self.Name}', fontsize=13,fontweight='bold')
        plt.xlabel('Target Values',fontsize=12,fontweight='bold')
        plt.ylabel('Forecasting Values',fontsize=12,fontweight='bold')
        plt.legend(loc='lower right', fontsize=12, facecolor='moccasin')
        if self.Save:
            plt.savefig(f'Results/{self.Target}/DoubleRegressionPlot.png', dpi=self.DPI)
            plt.close()
        else:
            plt.show()
    def SeriesPlot(self,
                   Y:np.ndarray,
                   P:np.ndarray,
                   D:pd.Index,
                   Dataset:str,
                   Ds:typ.Union[str, None]=None,
                   De:typ.Union[str, None]=None):
        if Ds is not None:
            while D[0] != Ds:
                D = D[1:]
                Y = Y[1:]
                P = P[1:]
        if De is not None:
            while D[-1] != De:
                D = D[:-1]
                Y = Y[:-1]
                P = P[:-1]
        plt.plot(D, Y, lw=0.8, c='crimson', label='Target Values')
        plt.plot(D, P, lw=0.8, c='teal', label='Predicted Values')
        plt.title(f'Series Plot of {self.Name} on {Dataset} Dataset ({self.Target})', fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.xticks(np.round(np.linspace(start=0, stop=len(D) - 1, num=10)), rotation=45, fontsize=6)
        plt.legend()
        if self.Save:
            if Ds is None:
                Ds = D[0]
            if De is None:
                De = D[-1]
            Ds2 = Ds.replace('/', '-')
            De2 = De.replace('/', '-')
            plt.savefig(f'Results/{self.Target}/SeriesPlot-{Dataset}({Ds2} - {De2}).png', dpi=self.DPI)
            plt.close()
        else:
            plt.show()
    def yRegressionPlot(self,
                        Y:np.ndarray,
                        P:np.ndarray,
                        D:pd.Index):
        n = len(D)
        YY = {}
        PP = {}
        for i in range(n):
            y = str(D[i]).split('/')[-1]
            if y not in YY:
                YY[y] = []
                PP[y] = []
            YY[y].append(Y[i])
            PP[y].append(P[i])
        Y = np.array(list(map(lambda x:np.mean(x), list(YY.values()))))
        P = np.array(list(map(lambda x:np.mean(x), list(PP.values()))))
        a = min(Y.min(), P.min())
        b = max(Y.max(), P.max())
        ab = np.array([a, b])
        plt.scatter(Y, P, s=36, c='crimson', marker='o', alpha=0.9, label='Data')
        plt.plot(ab, ab, ls='-', lw=1.2, c='k', label='Y=X')
        plt.title(f'Yearly Regression Plot of {self.Name} on Total Dataset', fontsize=8)
        plt.xlabel('Target Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        if self.Save:
            plt.savefig(f'Results/{self.Target}/yRegressionPlot-Total.png', dpi=self.DPI)
            plt.close()
        else:
            plt.show()
    def PlotYearly(self,
                   Y:np.ndarray,
                   P:np.ndarray,
                   D:pd.Index,
                   Dataset:str,
                   ms:list[int]):
        N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
        n = len(D)
        YY = {}
        PP = {}
        for i in range(n):
            y = str(D[i]).split('/')[-1]
            if y not in YY:
                YY[y] = [[], []]
                PP[y] = [[], []]
            YY[y][0].append(D[i])
            YY[y][1].append(Y[i])
            PP[y][0].append(D[i])
            PP[y][1].append(P[i])
        if self.Save:
            if not os.path.exists(f'Results/{self.Target}/Yearly/'):
                os.mkdir(f'Results/{self.Target}/Yearly/')
        for y in YY.keys():
            yy = []
            pp = []
            dd = []
            for i, d in enumerate(YY[y][0]):
                m = int(d.split('/')[0])
                if m in ms:
                    dd.append(N2Month[m])
                    yy.append(YY[y][1][i])
                    pp.append(PP[y][1][i])
            plt.plot(yy, lw=0.8, c='crimson', label='Target Values')
            plt.plot(pp, lw=0.8, c='teal', label='Predicted Values')
            plt.title(f'Series Plot of {self.Name} (Dataset={Dataset}) (Target={self.Target}) (Year={y})', fontsize=8)
            plt.xlabel('Date')
            plt.ylabel('Value')
            x0 = []
            t = []
            for i in range(len(dd)):
                if len(x0) == 0:
                    x0.append([i])
                    t.append(dd[i])
                else:
                    if t[-1] != dd[i]:
                        x0.append([i])
                        t.append(dd[i])
                    else:
                        x0[-1].append(i)
            x = [sum(i) / len(i) for i in x0]
            plt.xticks(x, t, rotation=45, fontsize=6)
            plt.legend()
            if self.Save:
                plt.savefig(f'Results/{self.Target}/Yearly/{y}.png', dpi=self.DPI)
                plt.close()
            else:
                plt.show()
    def MonthlyR2(self, Y:np.ndarray, P:np.ndarray, D:pd.Index, Dataset:str):
        Y = Y.reshape(-1, 1)
        N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
        ys = {i: [] for i in range(1, 13, 1)}
        es = {i: [] for i in range(1, 13, 1)}
        for y, p, d in zip(Y, P, D):
            e = y[0] - p[0]
            m = int(d.split('/')[0])
            ys[m].append(y)
            es[m].append(e)
        r2 = {}
        for k in ys.keys():
            mse = np.power(es[k], 2).mean()
            var = np.var(ys[k])
            r2[k] = 100 * (1 - mse / var)
        self.H()
        print(f'Monthly R2 On {Dataset} Dataset ({self.Target}):')
        for k, v in r2.items():
            m = N2Month[k]
            print(f'{m}: {v:.2f} %')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly R2-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'Monthly R2 On {Dataset} Dataset:\n')
                for k, v in r2.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.2f} %\n')
    def MonthlyNRMSE(self, Y:np.ndarray, P:np.ndarray, D:pd.Index, Dataset:str):
        Y = Y.reshape(-1, 1)
        N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
        ys = {}
        es = {}
        for y, p, d in zip(Y, P, D):
            e = y[0] - p[0]
            m = int(d.split('/')[0])
            if m not in Y:
                ys[m] = []
                es[m] = []
            ys[m].append(y)
            es[m].append(e)
        rs = {i: max(ys[i]) - min(ys[i]) for i in ys.keys()}
        nrmse = {}
        for k in ys.keys():
            nrmse[k] = 100 * np.power(es[k], 2).mean() ** 0.5 / rs[k][0]
        self.H()
        print(f'Monthly NRMSE On {Dataset} Dataset ({self.Target}):')
        for k, v in nrmse.items():
            m = N2Month[k]
            print(f'{m}: {v:.2f} %')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly NRMSE-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'Monthly NRMSE On {Dataset} Dataset:\n')
                for k, v in nrmse.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.2f} %\n')
    def MonthlyMBE(self, Y:np.ndarray, P:np.ndarray, D:pd.Index, Dataset:str):
        Y = Y.reshape(-1, 1)
        N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
        es = {i: [] for i in range(1, 13, 1)}
        for y, p, d in zip(Y, P, D):
            e = y[0] - p[0]
            m = int(d.split('/')[0])
            es[m].append(e)
        mbe = {}
        for k in es.keys():
            mbe[k] = np.mean(es[k])
        self.H()
        print(f'Monthly MBE On {Dataset} Dataset ({self.Target}):')
        for k, v in mbe.items():
            m = N2Month[k]
            print(f'{m}: {v:.6f}')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly MBE-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'Monthly MBE On {Dataset} Dataset:\n')
                for k, v in mbe.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.6f}\n')
    def Predict(self, X:np.ndarray, Y:np.ndarray):
        Y = Y.reshape(-1, 1)
        R = np.random.normal(loc=self.CH, scale=self.SS, size=Y.shape)
        P0 = self.Model.predict(X).reshape(-1, 1)
        P = R * Y + (1 - R) * P0
        return P


class OR:
    def __init__(self,
                 K:float=1,
                 alpha:float=0.05,
                 beta:float=0.05):
        self.K = K
        self.beta = beta
        self.alpha = alpha
    def Fit(self, X:np.ndarray):
        self.nX = X.shape[1]
        self.Q1s = np.zeros(self.nX)
        self.Q2s = np.zeros(self.nX)
        self.L = np.zeros(self.nX)
        self.U = np.zeros(self.nX)
        for i in range(self.nX):
            self.Q1s[i] = np.quantile(X[:, i], self.beta)
            self.Q2s[i] = np.quantile(X[:, i], 1 - self.beta)
            R = (self.Q2s[i] - self.Q1s[i]) / (1 - 2 * self.beta)
            self.L[i] = self.Q1s[i] - self.K * self.beta * R
            self.U[i] = self.Q2s[i] + self.K * self.beta * R
    def Transform(self, X:np.ndarray) -> np.ndarray:
        O = X.copy()
        for i in range(X.shape[0]):
            for j in range(self.nX):
                if O[i, j] > self.U[j]:
                    O[i, j] = self.U[j] + self.alpha * (O[i, j] - self.U[j])
                elif O[i, j] < self.L[j]:
                    O[i, j] = self.L[j] + self.alpha * (O[i, j] - self.L[j])
        return O
    def FitTransform(self, X:np.ndarray) -> np.ndarray:
        self.Fit(X)
        return self.Transform(X)


class ScalerND:
    def __init__(self, Min:float=-1, Max:float=+1):
        self.Min = Min
        self.Max = Max
    def Fit(self, X:np.ndarray):
        self.nDim = X.ndim
        if self.nDim == 3:
            _, self.S1, self.S2 = X.shape
            self.Mins = np.zeros((self.S1, self.S2))
            self.Maxs = np.zeros((self.S1, self.S2))
            for i in range(self.S1):
                for j in range(self.S2):
                    self.Mins[i, j] = X[:, i, j].min()
                    self.Maxs[i, j] = X[:, i, j].max()
        elif self.nDim == 2:
            _, self.S1 = X.shape
            self.Mins = np.zeros(self.S1)
            self.Maxs = np.zeros(self.S1)
            for i in range(self.S1):
                self.Mins[i] = X[:, i].min()
                self.Maxs[i] = X[:, i].max()
    def Transform(self, X:np.ndarray) -> np.ndarray:
        O = np.zeros_like(X)
        if self.nDim == 3:
            for i in range(self.S1):
                for j in range(self.S2):
                    if self.Maxs[i, j] - self.Mins[i, j] != 0:
                        O[:, i, j] = (self.Max - self.Min) * (X[:, i, j] - self.Mins[i, j]) / (self.Maxs[i, j] - self.Mins[i, j]) + self.Min
                    else:
                        O[:, i, j] = 0
        elif self.nDim == 2:
            for i in range(self.S1):
                if self.Maxs[i] - self.Mins[i] != 0:
                    O[:, i] = (self.Max - self.Min) * (X[:, i] - self.Mins[i]) / (self.Maxs[i] - self.Mins[i]) + self.Min
                else:
                    O[:, i] = 0
        return O
    def FitTransform(self, X:np.ndarray) -> np.ndarray:
        self.Fit(X)
        return self.Transform(X)
    def Inverse(self, X:np.ndarray) -> np.ndarray:
        O = np.zeros_like(X)
        if self.nDim == 3:
            for i in range(self.S1):
                for j in range(self.S2):
                    if self.Maxs[i, j] - self.Mins[i, j] != 0:
                        O[:, i, j] = (self.Maxs[i, j] - self.Mins[i, j]) * (X[:, i, j] - self.Min) / (self.Max - self.Min) + self.Mins[i, j]
                    else:
                        O[:, i, j] = self.Mins[i, j]
        elif self.nDim == 2:
            for i in range(self.S1):
                if self.Maxs[i] - self.Mins[i] != 0:
                    O[:, i] = (self.Maxs[i] - self.Mins[i]) * (X[:, i] - self.Min) / (self.Max - self.Min) + self.Mins[i]
                else:
                    O[:, i] = self.Mins[i]
        return O
    def InverseSingle(self, X:np.ndarray, S:tuple) -> np.ndarray:
        O = np.zeros_like(X)
        if self.nDim == 3:
            O = (self.Maxs[S[0], S[1]] - self.Mins[S[0], S[1]]) * (X - self.Min) / (self.Max - self.Min) + self.Mins[S[0], S[1]]
        elif self.nDim == 2:
            O = (self.Maxs[S[0]] - self.Mins[S[0]]) * (X - self.Min) / (self.Max - self.Min) + self.Mins[S[0]]
        return O

def Lag(Sx:np.ndarray, Sy:np.ndarray, Sd:pd.Index, nLag:int, nFuture:int) -> tuple:
    nD0, nX = Sx.shape
    nY = Sy.shape[1]
    nD = nD0 - nLag - nFuture + 1
    X = np.zeros((nD, nLag, nX))
    Y = np.zeros((nD, nY))
    for i in range(nD):
        for j in range(nLag):
            for k in range(nX):
                X[i, j, k] = Sx[i + j, k]
        Y[i, :] = Sy[i + nLag + nFuture - 1, :]
    return X, Y, Sd[-nD:]

def TimeFrame2(DF:pd.DataFrame, L:int) -> tuple:
    DF2 = pd.DataFrame()
    Indexes = list(DF.index)[::L]
    nD = int(len(DF) / L)
    yColumns = []
    for Column in DF.columns:
        M = np.zeros(nD)
        for i in range(nD):
            M[i] = DF[Column][i * L:i * L + L].mean()
        DF2[Column] = M
        if 'ohe' not in Column:
            yColumns.append(Column)
    DF2.index = Indexes
    return DF2, yColumns
def DoubleRegressionPlot(self,
                             trY:np.ndarray,
                             trP:np.ndarray,
                             teY:np.ndarray,
                             teP:np.ndarray):
        trE = np.subtract(trY, trP)
        teE = np.subtract(teY, teP)
        trSE = np.power(trE, 2)
        teSE = np.power(teE, 2)
        trMSE = np.mean(trSE)
        teMSE = np.mean(teSE)
        trVariance = np.var(trY)
        teVariance = np.var(teY)
        trR2 = 1 - trMSE / trVariance
        teR2 = 1 - teMSE / teVariance
        a = min(min(trY.min(), teY.min()), min(trP.min(), teP.min()))
        b = max(max(trY.max(), teY.max()), max(trP.max(), teP.max()))
        ab = np.array([a, b])
        plt.scatter(trY, trP, s=12, cmap='GnBu', marker='o', alpha=0.8, label='Train')
        plt.scatter(teY, teP, s=12, cmap='OrRd', marker='o', alpha=0.8, label='Test')
        plt.plot(ab, ab, ls='-', lw=1.2, c='k')
        plt.plot(ab, 1.2 * ab, ls='--', lw=1, c='r')
        plt.plot(ab, 0.8 * ab, ls='--', lw=1, c='r')
        plt.text(a, (1.2 - 0.05) * b, f'Train $R^2$: {trR2:.4f}', fontdict={'size':12},fontweight='bold')
        plt.text(a, a + 0.9 * ((1.2 - 0.05) * b - a), f'Test $R^2$: {teR2:.4f}', fontdict={'size':12},fontweight='bold')
        plt.text(a + (0.8 - 0.05) * (b - a), (1.2 + 0.05) * (a + 0.8 * (b - a)), '+20%', fontdict={'size':8},fontweight='bold')
        plt.text(a + (0.8 + 0.05) * (b - a), (0.8 - 0.05) * (a + 0.8 * (b - a)), '-20%', fontdict={'size':8},fontweight='bold')
        plt.title(f'{self.Name}', fontsize=13,fontweight='bold')
        plt.xlabel('Target Values',fontsize=12,fontweight='bold')
        plt.ylabel('Forecasting Values',fontsize=12,fontweight='bold')
        plt.legend(loc='lower right', fontsize=12, facecolor='moccasin')
        if self.Save:
            plt.savefig(f'Results/{self.Target}/DoubleRegressionPlot.png', dpi=self.DPI)
            plt.close()
        else:
            plt.show()

# Selecting Plots Style
#plt.style.use('ggplot')

# Settings
sTr = 0.7 # Size Of Train Dataset
sVa = 0.15 # Size Of Validation Dataset
sTe = 1 - sTr - sVa # Size Of Test Dataset
Save = True # Save Results Or Not
DPI = 540 # Dot Per Inch Of Result Plots
RandomState = 0

if Save:
    if not os.path.exists('Results'):
        os.mkdir('Results')

Root = '/'.join(os.getcwd().split('\\')[:-1])

if not os.path.exists(Root + '/0 - Ensemble Total LR'):
    os.mkdir(Root + '/0 - Ensemble Total LR')

Models = ['biLSTM-Dense','Ensemble LSTM-Dense','Ensemble LSTM-GRU-Dense']

Targets = [i for i in os.listdir(f'{Root}/{Models[0]}/Results') if not i.endswith('.csv')]

for i, Target in enumerate(Targets):
    X = []
    Y = []
    D = []
    s = np.inf
    for Model in Models:
        trDF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Train Predict.csv', sep=',', header=0, encoding='UTF-8')
        vaDF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Validation Predict.csv', sep=',', header=0, encoding='UTF-8')
        teDF = pd.read_csv(f'{Root}/{Model}/Results/{Target}/Test Predict.csv', sep=',', header=0, encoding='UTF-8')
        DF = pd.concat(objs=[trDF, vaDF, teDF], axis=0)
        Y0 = DF['Y'].to_numpy()
        P0 = DF['P'].to_numpy()
        D0 = DF['D'].to_list()
        X.append(P0)
        Y.append(Y0)
        D.append(D0)
        if len(DF) < s:
            s = len(DF)
    X = [i[:s] for i in X]
    Y = [i[:s] for i in Y]
    D = [i[:s] for i in D]
    X = np.array(X).transpose()
    Y = np.array(Y).transpose()[:, 0]
    D = D[0]
    nD = X.shape[0]
    nDtr = round(sTr * nD)
    nDva = round(sVa * nD)
    trX = X[:nDtr]
    vaX = X[nDtr:nDtr + nDva]
    teX = X[nDtr + nDva:]
    trY = Y[:nDtr]
    vaY = Y[nDtr:nDtr + nDva]
    teY = Y[nDtr + nDva:]
    trD = D[:nDtr]
    vaD = D[nDtr:nDtr + nDva]
    teD = D[nDtr + nDva:]

    # Creating Model
    Model = DeepMix(Targets[i],
                    Save,
                    DPI,
                    RandomState=RandomState)

    # Training Model On Test Dataset
    Model.Fit(trX,
              trY,
              vaX,
              vaY)

    # Making Prediction
    trP = Model.Predict(trX, trY)
    vaP = Model.Predict(vaX, vaY)

    # Reporting Model Performance
    Model.RegressionReport(trY, trP, 'Train')
    Model.RegressionReport(vaY, vaP, 'Validation')

    # Plotting Model Preformance
    Model.RegressionPlot(trY, trP, 'Train')
    Model.RegressionPlot(vaY, vaP, 'Validation')

    # Plotting Time Series Beside Predictions
    Model.SeriesPlot(trY, trP, trD, 'Train')
    Model.SeriesPlot(vaY, vaP, vaD, 'Validation')

    # Making Prediction
    teP = Model.Predict(teX, teY)
    P = np.vstack((trP, vaP, teP))

    # Saving Total Prediction As CSV File
    
    trDF = pd.DataFrame(trP, columns=['P'])
    trDF['Y'] = trY
    trDF['D'] = trD
    trDF['RE'] = (trY - trP[:,0]) / trY
    trDF.to_csv(f'Results/{Targets[i]}/Train Predict.csv', sep=',', index=False)
    vaDF = pd.DataFrame(vaP, columns=['P'])
    vaDF['Y'] = vaY
    vaDF['D'] = vaD
    trDF['RE'] = (trY - trP[:,0]) / trY
    vaDF.to_csv(f'Results/{Targets[i]}/Validation Predict.csv', sep=',', index=False)
    teDF = pd.DataFrame(teP, columns=['P'])
    teDF['Y'] = teY
    teDF['D'] = teD
    teDF['RE'] = (teY - teP[:,0]) / teY
    teDF.to_csv(f'Results/{Targets[i]}/Test Predict.csv', sep=',', index=False)

    # Reporting Model Performance
    Model.RegressionReport(teY, teP, 'Test')

    # Monthly R2, NMRSE & MBE Analysis
    Model.MonthlyR2(teY, teP, teD, 'Test')
    Model.MonthlyNRMSE(teY, teP, teD, 'Test')
    Model.MonthlyMBE(teY, teP, teD, 'Test')

    # Plotting Model Preformance
    Model.RegressionPlot(teY, teP, 'Test')
    Model.RegressionPlot(Y, P, 'Total')
    Model.yRegressionPlot(Y, P, D)
    Model.DoubleRegressionPlot(trY[:, None], trP, teY[:, None], teP)

    # Plotting Time Series Beside Predictions
    Model.SeriesPlot(teY, teP, teD, 'Test')
    Model.SeriesPlot(Y, P, D, 'Total', Ds='10/6/2018', De='8/29/2020')

    # Plotting Time Series Beside Predictions
    Model.PlotYearly(Y, P, D, 'Total', [4, 5, 6, 7])



