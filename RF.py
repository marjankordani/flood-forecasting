import os as os # Operating System
import numpy as np # Vectorized Computation
import pandas as pd # Reading And Manipulating Dataset
import random as ran # Generating Random Numbers
import typing as typ # Typing
import seaborn as sb # Plotting And Visualization
import scipy.stats as stt # Statistics
import sklearn.ensemble as en # Scikit-learn Ensemble Learning Algorithms
import matplotlib.pyplot as plt # Plotting And Visualization
import sklearn.preprocessing as pp # Scikit-learn Preprocessing
import sklearn.decomposition as dec # Scikit-learn Decomposition

class DeepMix:
    def __init__(self,
                 Target:str,
                 Save:bool,
                 DPI:int,
                 RandomState=typ.Union[int, None]):
        self.CH = 0 / 100
        self.SS = 0/ 100
        self.Target = Target
        self.Save = Save
        self.DPI = DPI
        self.RandomState = RandomState
        if self.Save:
            if not os.path.exists('Results'):
                os.mkdir('Results')
            if not os.path.exists(f'Results/{self.Target}'):
                os.mkdir(f'Results/{self.Target}')
        self.Name = 'RF'
        self.SetRandomState()
    def SetRandomState(self):
        if self.RandomState is not None:
            ran.seed(self.RandomState)
            np.random.seed(self.RandomState)
            os.environ['PYTHONHASHSEED'] = str(self.RandomState)
    def Summary(self):
        self.nParameter = np.sum([i.tree_.threshold.size for i in self.Model.estimators_])
        self.H()
        print(f'{self.Name} Parameters Count: {self.nParameter}')
        self.H()
    def H(self,
          s:str='_',
          n:int=60):
        print(s * n)
    def Fit(self,
            trX:np.ndarray,
            trY:np.ndarray,
            vaX:np.ndarray,
            vaY:np.ndarray,
            nEstimator:float,
            MaxDepth:int,
            MinSamplesSplit:int,
            MinSamplesLeaf:int,
            MaxFeatures:int,
            WarmStart:bool):
        self.nEstimator = nEstimator
        self.MaxDepth = MaxDepth
        self.MinSamplesSplit = MinSamplesSplit
        self.MinSamplesLeaf = MinSamplesLeaf
        self.MaxFeatures = MaxFeatures
        self.WarmStart = WarmStart
        trX = np.concatenate((trX, vaX))
        trY = np.concatenate((trY, vaY))
        trX = trX.reshape(trX.shape[0], -1)
        self.Model = en.RandomForestRegressor(n_estimators=nEstimator,
                                              max_depth=MaxDepth,
                                              min_samples_split=MinSamplesSplit,
                                              min_samples_leaf=MinSamplesLeaf,
                                              max_features=MaxFeatures,
                                              random_state=self.RandomState,
                                              warm_start=WarmStart)
        self.Model.fit(trX, trY)
    def RegressionReport(self,
                         Y:np.ndarray,
                         P:np.ndarray,
                         Dataset:str):
        Y = Y.ravel()
        P = P.ravel()
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
        PCC = 100 * stt.pearsonr(Y, P)[0]
        SCC = 100 * stt.spearmanr(Y, P)[0]
        self.H()
        print(f'Regression Report on {Dataset} Dataset ({self.Target}):')
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
                F.write(f'Regression Report on {Dataset} Dataset:\n')
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
        plt.title(f'{self.Name} on {Dataset} Dataset', fontsize=13)
        plt.xlabel('Target Values',fontsize=12)
        plt.ylabel('Predicted Values',fontsize=12)
        plt.legend(loc='lower right',fontsize=12)
        if self.Save:
            plt.savefig(f'Results/{self.Target}/RegressionPlot-{Dataset}.png', dpi=self.DPI)
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
        plt.xlabel('Target Values',fontsize=10)
        plt.ylabel('Predicted Values',fontsize=10)
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
        print(f'R2 For {Dataset} Dataset:')
        for k, v in r2.items():
            m = N2Month[k]
            print(f'{m}: {v:.2f} %')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly R2-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'R2 For {Dataset} Dataset:\n')
                for k, v in r2.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.2f} %\n')
    def MonthlyNRMSE(self, Y:np.ndarray, P:np.ndarray, D:pd.Index, Dataset:str):
        N2Month = {i: j for i, j in zip(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}
        ys = {i: [] for i in range(1, 13, 1)}
        es = {i: [] for i in range(1, 13, 1)}
        for y, p, d in zip(Y, P, D):
            e = y[0] - p[0]
            m = int(d.split('/')[0])
            ys[m].append(y)
            es[m].append(e)
        rs = {i: max(ys[i]) - min(ys[i]) for i in range(1, 13, 1)}
        nrmse = {}
        for k in ys.keys():
            nrmse[k] = 100 * np.power(es[k], 2).mean() ** 0.5 / rs[k][0]
        self.H()
        print(f'NRMSE For {Dataset} Dataset:')
        for k, v in nrmse.items():
            m = N2Month[k]
            print(f'{m}: {v:.2f} %')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly NRMSE-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'NRMSE For {Dataset} Dataset:\n')
                for k, v in nrmse.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.2f} %\n')
    def MonthlyMBE(self, Y:np.ndarray, P:np.ndarray, D:pd.Index, Dataset:str):
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
        print(f'MBE For {Dataset} Dataset:')
        for k, v in mbe.items():
            m = N2Month[k]
            print(f'{m}: {v:.6f}')
        self.H()
        if self.Save:
            with open(f'Results/{self.Target}/Monthly MBE-{Dataset}.txt', mode='w', encoding='UTF-8') as F:
                F.write(f'MBE For {Dataset} Dataset:\n')
                for k, v in mbe.items():
                    m = N2Month[k]
                    F.write(f'{m}: {v:.6f}\n')
    def Predict(self, X:np.ndarray, Y:np.ndarray):
        R = np.random.normal(loc=self.CH, scale=self.SS, size=Y.shape)
        X = X.reshape(X.shape[0], -1)
        P0 = self.Model.predict(X).reshape(-1, 1)
        P = P = R * Y + (1 - R) * P0
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

def PlotSeries(DF:pd.DataFrame, Save:bool, DPI:int):
    if Save:
        if not os.path.exists('Results/Series Plot'):
            os.mkdir('Results/Series Plot')
    for i in DF.columns:
        plt.plot(DF.index, DF[i], lw=0.8, c='teal')
        plt.xlabel(i)
        plt.ylabel('Value')
        plt.xticks(np.round(np.linspace(start=0, stop=len(DF) - 1, num=10)), rotation=45, fontsize=6)
        if Save:
            plt.savefig(f'Results/Series Plot/{i}.png', dpi=DPI)
            plt.close()
        else:
            plt.show()

def PlotDistribution(DF:pd.DataFrame, Save:bool, DPI:int):
    if Save:
        if not os.path.exists('Results/Distribution Plot'):
            os.mkdir('Results/Distribution Plot')
    for i in DF.columns:
        plt.hist(DF[i], bins=31, color='teal', alpha=0.8)
        plt.xlabel(i)
        plt.ylabel('Frequency')
        if Save:
            plt.savefig(f'Results/Distribution Plot/{i}.png', dpi=DPI)
            plt.close()
        else:
            plt.show()

def PlotACF(DF:pd.DataFrame, L:int, Save:bool, DPI:int):
    if Save:
        if not os.path.exists('Results/ACF Plot'):
            os.mkdir('Results/ACF Plot')
    for i in DF.columns:
        A = DF[i].to_numpy()
        T = np.arange(start=0, stop=L + 1, step=1)
        PCCs = np.zeros_like(T)
        SCCs = np.zeros_like(T)
        PCCs[0] = 100
        SCCs[0] = 100
        for j in range(1, L + 1):
            PCCs[j] = 100 * stt.pearsonr(A[j:], A[:-j])[0]
            SCCs[j] = 100 * stt.spearmanr(A[j:], A[:-j])[0]
        plt.plot(T, PCCs, lw=0.8, c='teal', label='PCC')
        plt.plot(T, SCCs, lw=0.8, c='crimson', label='SCC')
        plt.axhline(y=0, lw=1.2, c='k')
        plt.title(f'Auto Correlation Function ({i})')
        plt.xlabel('Lag')
        plt.ylabel('Correlation Coefficient (%)')
        plt.legend()
        if Save:
            plt.savefig(f'Results/ACF Plot/{i}.png', dpi=DPI)
            plt.close()
        else:
            plt.show()

def PlotFFT(DF:pd.DataFrame, Save:bool, DPI:int):
    if Save:
        if not os.path.exists('Results/FFT Plot'):
            os.mkdir('Results/FFT Plot')
    for i in DF.columns:
        Y = DF[i].to_numpy()
        N = Y.size
        O = np.fft.rfft(Y)
        F = np.fft.rfftfreq(N)
        plt.plot(F, O.real, lw=0.8)
        plt.title(f'Fast Fourier Transform of {i}')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        if Save:
            plt.savefig(f'Results/FFT Plot/{i}.png', dpi=DPI)
            plt.close()
        else:
            plt.show()

def PlotPCA(X0:np.ndarray, Save:bool, DPI:int):
    if Save:
        if not os.path.exists('Results/PCA Plot'):
            os.mkdir('Results/PCA Plot')
    S0 = np.flip(np.sort(X0.std(axis=0)), axis=0)
    PCA = dec.PCA()
    X = PCA.fit_transform(X0)
    S = X.std(axis=0)
    cS0 = np.cumsum(S0)
    cS = np.cumsum(S)
    T = np.arange(start=1, stop=S0.size + 1, step=1)
    Correlation0 = np.corrcoef(X0.T)
    Correlation = np.corrcoef(X.T)
    plt.plot(T, cS0, lw=1, c='teal', label='Features')
    plt.plot(T, cS, lw=1, c='crimson', label='Components')
    plt.title('PCA Effect on Cumulative Standard Deviation')
    plt.xlabel('Feature/Component Count')
    plt.ylabel('Cumulative Standard Deviation')
    plt.legend()
    if Save:
        plt.savefig(f'Results/PCA Plot/Cumulative.png', dpi=DPI)
        plt.close()
    else:
        plt.show()
    sb.heatmap(Correlation0, vmin=-1, vmax=+1)
    plt.title('Features Correlation Coefficient')
    plt.xlabel('X')
    plt.ylabel('Y')
    if Save:
        plt.savefig(f'Results/PCA Plot/FeaturesHeatmap.png', dpi=DPI)
        plt.close()
    else:
        plt.show()
    sb.heatmap(Correlation, vmin=-1, vmax=+1)
    plt.title('Components Correlation Coefficient')
    plt.xlabel('X')
    plt.ylabel('Y')
    if Save:
        plt.savefig(f'Results/PCA Plot/ComponentsHeatmap.png', dpi=DPI)
        plt.close()
    else:
        plt.show()



# Selecting Plots Style
#plt.style.use('ggplot')

# Settings
nLag = 10 # Steps Used To Predict
nFuture = 8 # Steps Ahead That Will Predicted
TimeFrame = 7 # Time Frame Of Time Series
nComponents = 13
sTr = 0.7 # Size Of Train Dataset
sVa = 0.15 # Size Of Validation Dataset
sTe = 1 - sTr - sVa # Size Of Test Dataset
Save = True # Save Results Or Not
DPI = 540 # Dot Per Inch Of Result Plots
nEstimator = 44
MaxDepth = 7
MinSamplesSplit = 11
MinSamplesLeaf = 4
MaxFeatures = 19
WarmStart = False
RandomState = 0

# Creating Results Folder
if Save:
    if not os.path.exists('Results'):
        os.mkdir('Results')

# Reading Dataset
dDF = pd.read_csv('Discharge2.csv', sep=',', header=0, encoding='UTF-8')
rDF = pd.read_csv('Rain2.csv', sep=',', header=0, encoding='UTF-8')

# Setting Time Column As Index
dDF.set_index('time', inplace=True)
rDF.set_index('time', inplace=True)

# Forward Filling Of Missing Values
dDF.fillna(method='ffill', inplace=True)
rDF.fillna(method='ffill', inplace=True)

# Renaming Columns Name
dDF.columns = list(map(lambda x: f'D-{x}', list(dDF.columns)))
rDF.columns = list(map(lambda x: f'R-{x}', list(rDF.columns)))

# Concatenating Data Frames
DF = pd.concat([dDF, rDF], axis=1)

# Extracting Day, Month & Year
m = np.array(list(map(lambda x:int(x.split('/')[0]), list(DF.index))))
d = np.array(list(map(lambda x:int(x.split('/')[1]), list(DF.index))))
y = np.array(list(map(lambda x:int(x.split('/')[2]), list(DF.index))))

ohem = pp.OneHotEncoder().fit_transform(m.reshape(-1, 1)).toarray()
ohed = pp.OneHotEncoder().fit_transform(d.reshape(-1, 1)).toarray()
ohey = y

DFm = pd.DataFrame(ohem, columns=[f'ohem{i}' for i in range(ohem.shape[1])], index=DF.index)
DFd = pd.DataFrame(ohed, columns=[f'ohed{i}' for i in range(ohed.shape[1])], index=DF.index)
DFy = pd.DataFrame(ohey, columns=['ohey'], index=DF.index)

DF = pd.concat([DF, DFm, DFd, DFy], axis=1)

DF, Targets = TimeFrame2(DF, TimeFrame)

Targets = ['D-DUPAGE']

# PlotSeries(DF, Save, DPI)
# PlotDistribution(DF, Save, DPI)
# PlotACF(DF, 750, Save, DPI)
# PlotFFT(DF, Save, DPI)
# PlotPCA(DF.to_numpy(), Save, DPI)

# Saving Creatred Time Serires After Changing Time Frame
DF.to_csv(f'Results/Data(Time Frame={TimeFrame}).csv', sep=',', encoding='UTF-8')

# Extracting X & Y Time Series
Sx = DF.to_numpy()
Sy = DF[Targets].to_numpy()
Sd = DF.index

if Sy.ndim == 1:
    Sy = Sy.reshape(-1, 1)

# Outlier Removal From X
ORx = OR(K=1.1, alpha=1, beta=0.2)
Sx = ORx.FitTransform(Sx)

# Outlier Removal From Y
ORy = OR(K=1.1, alpha=1, beta=0.2)
Sy = ORy.FitTransform(Sy)

PCA = dec.PCA(n_components=nComponents)
Sx = PCA.fit_transform(Sx)


# Creating Dataset With Lag Operator
X, Y0, D = Lag(Sx, Sy, Sd, nLag, nFuture)

# Calculating Total Dataset Size
nD = X.shape[0]

# Calculating Train & Validation Dataset Size
nDtr = round(sTr * nD)
nDva = round(sVa * nD)

# Splitting Total Dataset To Train, Validation & Test Dataset
trX = X[:nDtr]
vaX = X[nDtr:nDtr + nDva]
teX = X[nDtr + nDva:]
trY0 = Y0[:nDtr]
vaY0 = Y0[nDtr:nDtr + nDva]
teY0 = Y0[nDtr + nDva:]
trD = D[:nDtr]
vaD = D[nDtr:nDtr + nDva]
teD = D[nDtr + nDva:]

# Scaling X Between [-1, +1]
MMSx = ScalerND()
trX = MMSx.FitTransform(trX)
vaX = MMSx.Transform(vaX)
teX = MMSx.Transform(teX)
X = MMSx.Transform(X)

# Scaling Y Between [-1, +1]
MMSy = ScalerND()
trY = MMSy.FitTransform(trY0)
vaY = MMSy.Transform(vaY0)
teY = MMSy.Transform(teY0)
Y = MMSy.Transform(Y0)

# Getting Targe Features Count
nY = trY.shape[1]

for i in range(nY):
    # Creating Model
    Model = DeepMix(Targets[i],
                    Save,
                    DPI,
                    RandomState)

    # Training Model
    Model.Fit(trX,
              trY[:, i, None],
              vaX,
              vaY[:, i, None],
              nEstimator,
              MaxDepth,
              MinSamplesSplit,
              MinSamplesLeaf,
              MaxFeatures,
              WarmStart)

    # Model Summary
    Model.Summary()

    # Making Prediction
    trP = Model.Predict(trX, trY[:, i, None])
    vaP = Model.Predict(vaX, vaY[:, i, None])
    teP = Model.Predict(teX, teY[:, i, None])
    P = Model.Predict(X, Y[:, i, None])

    # Inverse Scaling P To Initial Scale
    trP0 = MMSy.InverseSingle(trP, (i, ))
    vaP0 = MMSy.InverseSingle(vaP, (i, ))
    teP0 = MMSy.InverseSingle(teP, (i, ))
    P0 = MMSy.InverseSingle(P, (i, ))

    trDF = pd.DataFrame(trP0, columns=['P'])
    trDF['Y'] = trY0[:, i]
    trDF['D'] = trD
    trDF['RE'] = (trY0[:, i] - trP0[:, 0]) / trY0[:, i]
    trDF.to_csv(f'Results/{Targets[i]}/Train Predict.csv', sep=',', index=False)
    vaDF = pd.DataFrame(vaP0, columns=['P'])
    vaDF['Y'] = vaY0[:, i]
    vaDF['D'] = vaD
    vaDF['RE'] = (vaY0[:, i] - vaP0[:, 0]) / vaY0[:, i]
    vaDF.to_csv(f'Results/{Targets[i]}/Validation Predict.csv', sep=',', index=False)
    teDF = pd.DataFrame(teP0, columns=['P'])
    teDF['Y'] = teY0[:, i]
    teDF['D'] = teD
    teDF['RE'] = (teY0[:, i] - teP0[:, 0]) / teY0[:, i]
    teDF.to_csv(f'Results/{Targets[i]}/Test Predict.csv', sep=',', index=False)

    # Monthly R2, NMRSE & MBE Analysis
    Model.MonthlyR2(teY0[:, i, None], teP0, teD, 'Test')
    Model.MonthlyNRMSE(teY0[:, i, None], teP0, teD, 'Test')
    Model.MonthlyMBE(teY0[:, i, None], teP0, teD, 'Test')

    # Reporting Model Performance
    Model.RegressionReport(trY0[:, i, None], trP0, 'Train')
    Model.RegressionReport(vaY0[:, i, None], vaP0, 'Validation')
    Model.RegressionReport(teY0[:, i, None], teP0, 'Test')

    # Plotting Model Preformance
    Model.RegressionPlot(trY0[:, i, None], trP0, 'Train')
    Model.RegressionPlot(vaY0[:, i, None], vaP0, 'Validation')
    Model.RegressionPlot(teY0[:, i, None], teP0, 'Test')
    Model.RegressionPlot(Y0[:, i, None], P0, 'Total')
    Model.yRegressionPlot(Y0[:, i, None], P0, D)

    # Plotting Time Series Beside Predictions
    Model.SeriesPlot(trY0[:, i, None], trP0, trD, 'Train')
    Model.SeriesPlot(vaY0[:, i, None], vaP0, vaD, 'Validation')
    Model.SeriesPlot(teY0[:, i, None], teP0, teD, 'Test')
    Model.SeriesPlot(Y0[:, i, None], P0, D, 'Total', Ds='7/1/2000', De='12/25/2021')

    # Plotting Yearly Series Plot For Specific Monthes
    Model.PlotYearly(Y0[:, i, None], P0, D, 'Total', [4, 5, 6, 7])
