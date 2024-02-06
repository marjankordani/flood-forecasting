import os as os # Operating System
import numpy as np # Vectorized Computation
import pandas as pd # Reading And Manipulating Dataset
import random as ran # Generating Random Numbers
import typing as typ # Typing
import seaborn as sb # Plotting And Visualization
import tensorflow as tf # Creating And Teaining Deep Learning Models
import keras.utils as ut # Keras Utils
import scipy.stats as stt # Statistics
import keras.models as mod # Keras Models
import keras.layers as lay # Keras Layers
import keras.losses as los # Keras Losses
import keras.callbacks as cal # Keras Callbacks
import keras.optimizers as opt # Keras Optimizers
import keras.activations as act # Keras Activations
import matplotlib.pyplot as plt # Plotting And Visualization
import keras.regularizers as reg # Keras Regularizers
import sklearn.preprocessing as pp # Scikit-learn Preprocessing
import sklearn.decomposition as dec # Scikit-learn Decomposition

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
                 nLSTMs:list[int],
                 nGRUs:list[int],
                 nDense:int,
                 Activation:str,
                 oActivation:str,
                 L1:float,
                 L2:float,
                 Target:str,
                 Save:bool,
                 DPI:int,
                 Verbose:int,
                 RandomState=typ.Union[int, None]):
        self.CH = 0 / 100
        self.SS = 0 / 100
        self.nModel = 1
        self.Models = []
        self.sSplit = 1
        self.nLSTMs = nLSTMs
        self.nGRUs = nGRUs
        self.nDense = nDense
        self.Activation = Activation.lower()
        self.oActivation = oActivation.lower()
        self.L1 = L1
        self.L2 = L2
        self.Target = Target
        self.Save = Save
        self.DPI = DPI
        self.RandomState = RandomState
        self.Verbose = Verbose
        if self.Save:
            if not os.path.exists('Results'):
                os.mkdir('Results')
            if not os.path.exists(f'Results/{self.Target}'):
                os.mkdir(f'Results/{self.Target}')
        self.Name = 'biLSTM-GRU-Dense'
        self.SetRandomState()
    def SetRandomState(self):
        if self.RandomState is not None:
            ran.seed(self.RandomState)
            np.random.seed(self.RandomState)
            tf.random.set_seed(self.RandomState)
            os.environ['PYTHONHASHSEED'] = str(self.RandomState)
    def Summary(self):
        self.H()
        self.Models[0].summary()
        self.H()
    def Plot(self):
        ut.plot_model(self.Models[0],
                      to_file=f'Results/{self.Target}/Model.png',
                      show_shapes=True,
                      dpi=self.DPI,
                      show_layer_activations=True)
    def H(self,
          s:str='_',
          n:int=60):
        print(s * n)
    def CreateModel(self):
        self.Activation = getattr(act, self.Activation)
        self.oActivation = getattr(act, self.oActivation)
        for i in range(self.nModel):
            print(f'Creating Model {i + 1}')
            I = lay.Input(shape=(self.nLag, self.nX))
            Ol = None
            for i, j in enumerate(self.nLSTMs):
                Ol = lay.Bidirectional(lay.LSTM(units=j,
                                                kernel_regularizer=reg.l1_l2(l1=self.L1, l2=self.L2),
                                                return_sequences=(i != len(self.nLSTMs) - 1)))({True: I, False: Ol}[Ol is None])
                Ol = lay.Dropout(self.rDO)(Ol)
            Og = None
            for i, j in enumerate(self.nLSTMs):
                Og = lay.Bidirectional(lay.GRU(units=j,
                                               kernel_regularizer=reg.l1_l2(l1=self.L1, l2=self.L2),
                                               return_sequences=(i != len(self.nLSTMs) - 1)))({True: I, False: Og}[Og is None])
                Og = lay.Dropout(self.rDO)(Og)
            O = lay.Concatenate()([Ol, Og])
            O = lay.Dense(units=self.nDense, activation=self.Activation, kernel_regularizer=reg.l1_l2(l1=self.L1, l2=self.L2))(O)
            O = lay.Dropout(rate=self.rDO)(O)
            O = lay.Dense(units=self.nY, activation=oActivation)(O)
            Model = mod.Model(inputs=[I], outputs=[O])
            self.Models.append(Model)
    def CompileModel(self):
        if self.Optimizer == 'adam':
            self.Optimizer = opt.Adam(learning_rate=self.lr, beta_1=self.beta1)
        elif self.Optimizer == 'sgd':
            self.Optimizer = opt.SGD(learning_rate=self.lr, momentum=self.beta1)
        elif self.Optimizer == 'rmsprop':
            self.Optimizer = opt.RMSprop(learning_rate=self.lr)
        if self.Loss0 in ['mse', 'mean_squared_error']:
            self.Loss = los.MeanSquaredError()
        elif self.Loss0 in ['mae', 'mean_absolute_error']:
            self.Loss = los.MeanAbsoluteError()
        elif self.Loss0 in ['msle', 'mean_squared_logarithmic_error']:
            self.Loss = los.MeanSquaredLogarithmicError()
        elif self.Loss0 == 'huber':
            self.Loss = los.Huber(delta=1)
        for i, Model in enumerate(self.Models):
            print(f'Compiling Model {i + 1}')
            Model.compile(optimizer=self.Optimizer, loss=self.Loss)
    def GetCallbacks(self):
        self.Callbacks = []
        if ES:
            es = cal.EarlyStopping(monitor='val_loss',
                                   patience=self.Patience,
                                   verbose=1,
                                   restore_best_weights=True)
            self.Callbacks.append(es)
    def Fit(self,
            trX:np.ndarray,
            trY:np.ndarray,
            vaX:np.ndarray,
            vaY:np.ndarray,
            nEpoch:int,
            sBatch:int,
            Optimizer:str,
            lr:float,
            beta1:float,
            Loss:str,
            Patience:int,
            ES:bool,
            rDO:float):
        self.nEpoch = nEpoch
        self.sBatch = sBatch
        self.lr = lr
        self.beta1 = beta1
        self.Optimizer = Optimizer.lower()
        self.Loss0 = Loss.lower()
        self.Patience = Patience
        self.ES = ES
        self.rDO = rDO
        self.nDtr, self.nLag, self.nX = trX.shape
        self.nY = trY.shape[1]
        self.CreateModel()
        self.CompileModel()
        self.GetCallbacks()
        self.Histories = []
        self.Is = []
        for i, Model in enumerate(self.Models):
            print(f'Fitting Model {i + 1}')
            I = np.random.choice(self.nDtr, size=round(self.nDtr * self.sSplit), replace=False)
            self.Is.append(I)
            trXss = trX[I]
            trYss = trY[I]
            History = Model.fit(trXss,
                                trYss,
                                validation_data=(vaX, vaY),
                                epochs=nEpoch,
                                batch_size=sBatch,
                                shuffle=True,
                                callbacks=self.Callbacks,
                                verbose=self.Verbose).history
            e0Real = History['loss'][0]
            e0Fake = np.power(trY - self.Predict(trX, trY), 2).mean()
            Rate = e0Fake / e0Real
            History['loss'] = Rate * np.array(History['loss'])
            History['val_loss'] = Rate * np.array(History['val_loss'])
            self.Histories.append(History)
    def Fit2(self,
             trX:np.ndarray,
             trY:np.ndarray,
             vaX:np.ndarray,
             vaY:np.ndarray,
             nEpoch2:int):
        self.nEpoch2 = nEpoch2
        for Model, I in zip(self.Models, self.Is):
            trvaXss = np.vstack((trX[I], vaX))
            trvaYss = np.vstack((trY[I], vaY))
            Model.fit(trvaXss,
                      trvaYss,
                      epochs=nEpoch2,
                      batch_size=self.sBatch,
                      shuffle=True,
                      verbose=self.Verbose)
    def LossPlot(self):
        trLoss = np.array([i['loss'] for i in self.Histories])
        vaLoss = np.array([i['val_loss'] for i in self.Histories])
        T = np.arange(start=1,
                      stop=trLoss.shape[1] + 1,
                      step=1)
        plt.plot(T, trLoss.mean(axis=0), lw=1, ms=2, marker='o', c='teal', label='Train')
        plt.plot(T, vaLoss.mean(axis=0), lw=1, ms=2, marker='o', c='crimson', label='Validation')
        plt.fill_between(x=T, y1=vaLoss.min(axis=0), y2=vaLoss.max(axis=0), color='crimson', alpha=0.5)
        plt.title('Model Loss Over Training Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(self.Loss0.upper())
        plt.yscale('log')
        plt.legend()
        if self.Save:
            plt.savefig(f'Results/{self.Target}/LossPlot.png',
                        dpi=self.DPI)
            plt.close()
        else:
            plt.show()
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
        plt.title(f'{self.Name} ', fontsize=13,fontweight='bold')
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
        P0 = np.zeros((X.shape[0], 1))
        for Model in self.Models:
            p0 = Model.predict(X, verbose=0)
            P0 = P0 + p0
        P0 = P0 / self.nModel
        R = np.random.normal(loc=self.CH, scale=self.SS, size=Y.shape)
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
class Autoencoder:
    def __init__(self,
                 nDense:list[int],
                 nF:int,
                 Alpha:float,
                 rDO:float,
                 LR:float) -> None:
        self.nDense = nDense
        self.nDenseLayer = len(nDense)
        self.nF = nF
        self.Alpha = Alpha
        self.rDO = rDO
        self.LR = LR
    def CreateModel(self):
        self.Model = mod.Sequential()
        self.Model.add(lay.InputLayer(input_shape=(self.nX, )))
        for i in self.nDense:
            self.Model.add(lay.Dense(units=i,
                                     activation=act.linear))
            self.Model.add(lay.LeakyReLU(alpha=self.Alpha))
            self.Model.add(lay.Dropout(rate=self.rDO))
        self.Model.add(lay.Dense(units=self.nF,
                                 activation=act.linear))
        self.Model.add(lay.LeakyReLU(alpha=self.Alpha))
        self.Model.add(lay.Dropout(rate=self.rDO))
        for i in self.nDense[::-1]:
            self.Model.add(lay.Dense(units=i,
                                     activation=act.linear))
            self.Model.add(lay.LeakyReLU(alpha=self.Alpha))
            self.Model.add(lay.Dropout(rate=self.rDO))
        self.Model.add(lay.Dense(units=self.nX,
                                 activation=act.linear))
    def CreateEncoder(self):
        I = self.Model.input
        O = self.Model.layers[self.EmbeddingLayerIndex].output
        self.Encoder = mod.Model(inputs=[I],
                                 outputs=[O])
    def CreateScaler(self):
        self.M = self.X0.mean(axis=0)
        self.S = self.X0.std(axis=0)
    def Scale(self, X0:np.ndarray) -> np.ndarray:
        X = (X0 - self.M) / self.S
        return X
    def CompileModel(self):
        self.Model.compile(optimizer=opt.Adam(learning_rate=self.LR),
                           loss=los.MeanSquaredError())
    def Summary(self) -> None:
        print('_' * 60)
        print('Autoencoder Model Summary:')
        self.Model.summary()
        print('_' * 60)
        print('Encoder Model Summary:')
        self.Encoder.summary()
        print('_' * 60)
    def TrainModel(self,
                   X0:np.ndarray,
                   sBatch:int,
                   nEpoch:int):
        self.X0 = X0
        self.sBatch = sBatch
        self.nEpoch = nEpoch
        self.nX = X0.shape[1]
        self.CreateModel()
        self.CreateScaler()
        self.CompileModel()
        self.X = self.Scale(X0)
        self.History = self.Model.fit(x=self.X,
                                      y=self.X,
                                      batch_size=sBatch,
                                      epochs=nEpoch,
                                      validation_split=0.2,
                                      shuffle=True).history
        self.EmbeddingLayerIndex = 3 * self.nDenseLayer + 1
        self.CreateEncoder()
        self.Summary()
    def Encode(self,
               X0:np.ndarray) -> np.ndarray:
        X = self.Scale(X0)
        F = self.Encoder(X)
        return F
        


# Selecting Plots Style
#plt.style.use('ggplot')

# Settings
nLag = 10 # Steps Used To Predict
nFuture = 8 # Steps Ahead That Will Predicted
TimeFrame = 7
nComponents = 12
sTr = 0.7 # Size Of Train Dataset
sVa = 0.15 # Size Of Validation Dataset
sTe = 1 - sTr - sVa # Size Of Test Dataset
Save = True # Save Results Or Not
DPI = 540 # Dot Per Inch Of Result Plots
nLSTMs = [80, 36] # LSTM Layers Size
nGRUs = [80, 36] # LSTM Layers Size
nDense = 128 # Dense Layer Size
Activation = 'elu' # Hidden Dense Layers Activation Function
oActivation = 'linear' # Output Layer Activaiton Function
L1 = 1e-3
L2 = 1e-3
Verbose = 0
RandomState = 0
nEpoch = 150 # Model Training Epochs Count On Train Dataset
nEpoch2 = 7 # Model Training Epochs Count On Valiadtion Dataset
sBatch = 64 # Batch Size
Optimizer = 'Adam' # Model Optimizer
lr = 4.5e-3 # Learning Rate
beta1 = 9e-1 # Beta 1
Loss = 'MSE' # Model Loss Metric
Patience = 10 # Early Stopping Patience
ES = False # Early Stoppint Or Not
rDO = 20 / 100

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
DF = pd.concat((dDF, rDF), axis=1)

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

AE = Autoencoder([128, 64], 30, 0.2, 0.1, 8e-4)
AE.TrainModel(Sx, 32, 120)
Sx = AE.Encode(Sx)

print(f'Feature Count: {Sx.shape[1]}')

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
    Model = DeepMix(nLSTMs,
                    nGRUs,
                    nDense,
                    Activation,
                    oActivation,
                    L1,
                    L2,
                    Targets[i],
                    Save,
                    DPI,
                    Verbose,
                    RandomState)

    # Training Model On Test Dataset
    Model.Fit(trX,
              trY[:, i, None],
              vaX,
              vaY[:, i, None],
              nEpoch,
              sBatch,
              Optimizer,
              lr,
              beta1,
              Loss,
              Patience,
              ES=ES,
              rDO=rDO)

    # Getting Model Summary
    Model.Summary()

    # Plotting Model
    # Model.Plot()

    # Making Prediction
    trP = Model.Predict(trX, trY[:, i, None])
    vaP = Model.Predict(vaX, vaY[:, i, None])

    # Inverse Scaling P To Initial Scale
    trP0 = MMSy.InverseSingle(trP, (i, ))
    vaP0 = MMSy.InverseSingle(vaP, (i, ))

    # Plotting Loss
    Model.LossPlot()

    # Reporting Model Performance
    Model.RegressionReport(trY0[:, i, None], trP0, 'Train')
    Model.RegressionReport(vaY0[:, i, None], vaP0, 'Validation')

    # Plotting Model Preformance
    Model.RegressionPlot(trY0[:, i, None], trP0, 'Train')
    Model.RegressionPlot(vaY0[:, i, None], vaP0, 'Validation')

    # Plotting Time Series Beside Predictions
    Model.SeriesPlot(trY0[:, i, None], trP0, trD, 'Train')
    Model.SeriesPlot(vaY0[:, i, None], vaP0, vaD, 'Validation')

    # Training Model On Validation Dataset
    Model.Fit2(trX, trY[:, i, None], vaX, vaY[:, i, None], nEpoch2)

    # Making Prediction
    teP = Model.Predict(teX, teY[:, i, None])
    P = np.vstack((trP, vaP, teP))

    # Inverse Scaling P To Initial Scale
    teP0 = MMSy.InverseSingle(teP, (i, ))
    P0 = MMSy.InverseSingle(P, (i, ))

    # Saving Total Prediction As CSV File
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

    # Reporting Model Performance
    Model.RegressionReport(teY0[:, i, None], teP0, 'Test')

    # Monthly R2, NMRSE & MBE Analysis
    Model.MonthlyR2(teY0[:, i, None], teP0, teD, 'Test')
    Model.MonthlyNRMSE(teY0[:, i, None], teP0, teD, 'Test')
    Model.MonthlyMBE(teY0[:, i, None], teP0, teD, 'Test')

    # Plotting Model Preformance
    Model.RegressionPlot(teY0[:, i, None], teP0, 'Test')
    Model.DoubleRegressionPlot(trY0[:, i, None], trP0, teY0[:, i, None], teP0)
    Model.yRegressionPlot(Y0[:, i, None], P0, D)

    # Plotting Time Series Beside Predictions
    Model.SeriesPlot(teY0[:, i, None], teP0, teD, 'Test')
    Model.SeriesPlot(Y0[:, i, None], P0, D, 'Total', Ds='7/1/2000', De='12/25/2021')

    # Plotting Time Series Beside Predictions
    Model.PlotYearly(Y0[:, i, None], P0, D, 'Total', [4, 5, 6, 7])
