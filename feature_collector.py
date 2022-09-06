import pandas as pd
import numpy as np
from dask.dataframe.shuffle import set_index
from datetime import datetime, timedelta
from time import sleep
import sys
sys.setrecursionlimit(10000)
from sklearn.preprocessing import StandardScaler
import joblib
import time
import datetime as dt

from feature_functions import *
from dwx_client import dwx_client

# Load our CSV Data

data = pd.read_csv('Data/EURUSD_2018-2022.csv')

pd.set_option('mode.chained_assignment', None)

data.columns = ['date','open','high','low','close','volume']
data['date']=pd.to_datetime(data.date)


for i in range(0,len(data)):
    if data.date.dt.minute.iloc[i] == 0:
        break
data = data[i:]

data = data.set_index(pd.to_datetime(data.date))
data = data[['open','high','low','close','volume']]
prices = data.drop_duplicates(keep=False)

prices = resamble(prices)

momentumKey = [3,4,5,8,9,10]
stochasticKey = [3,4,5,8,9,10]
williamsKey = [6,7,8,9,10]
procKey = [12,13,14,15]
wadlKey = [15]
adoscKey = [2,3,4,5]
macdKey = [15,30]
cciKey = [15]
bollingerKey = [15]
heikenashiKey = [15]
paverageKey = [2]
slopeKey = [3,4,5,10,20,30]
fourierKey = [10,20,30]
sineKey = [5,6]
marketKey = [0]


keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
           ,paverageKey,slopeKey,fourierKey,sineKey,marketKey]


momentumDict = momentum(prices,momentumKey)
print('1')
stochasticDict = stochastic(prices,stochasticKey)
print('2')
williamsDict = williams(prices,williamsKey)
print('3')
procDict = proc(prices,procKey)
print('4')
wadlDict = wadl(prices,wadlKey)
print('5')
adoscDict = adosc(prices,adoscKey)
print('6')
macdDict = macd(prices,macdKey)
print('7')
cciDict = cci(prices,cciKey)
print('8')
bollingerDict = bollinger(prices,bollingerKey,2)
print('9')
'''
hkaprices = prices.copy()
hkaprices['Symbol']='SYMB'
HKA = OHLCresample(hkaprices,'1H')

heikenDict = Heiken_Ashi(HKA,heikenashiKey) 
print('10') 
'''
paverageDict = paverage(prices,paverageKey)
print('11')
slopeDict = slopes(prices,slopeKey)
print('12')
fourierDict = fourier(prices,fourierKey)
print('13')
sineDict = sine(prices,sineKey)
print('14')
marketDict = Market(prices,marketKey)
print('15')
# Create list of dictionaries

dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
            ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
            ,cciDict.cci,bollingerDict.bands,paverageDict.avs
            ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs,marketDict.slope]

# list of column name on csv

colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
           'cci','bollinger','paverage','slope','fourier','sine','market']

masterFrame = pd.DataFrame(index = prices.index)
for i in range(0,len(dictlist)):
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1])
        masterFrame[colID] = dictlist[i]
    else:
        for j in keylist[i]:
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + str(k)
                masterFrame[colID] = dictlist[i][j][k]

threshold = round(0.7*len(masterFrame))
masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]
'''
masterFrame.heiken150 = masterFrame.heiken150.fillna(method='bfill') 
masterFrame.heiken151 = masterFrame.heiken151.fillna(method='bfill')
masterFrame.heiken152 = masterFrame.heiken152.fillna(method='bfill')
masterFrame.heiken153 = masterFrame.heiken153.fillna(method='bfill') 
# Drop columns that have 30% or more NAN data 
'''
masterFrameCleaned = masterFrame.copy()

masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)
masterFrameCleaned.to_csv('Data/calculatedData.csv')
print('completed')

columns = ['momentum3close', 'momentum4close'
    , 'momentum5close', 'momentum8close', 'momentum9close', 'momentum10close'
    , 'stoch3K', 'stoch3D', 'stoch4K', 'stoch4D'
    , 'stoch5K', 'stoch5D', 'stoch8K', 'stoch8D'
    , 'stoch9K', 'stoch9D', 'stoch10K'
    , 'stoch10D', 'will6R', 'will7R', 'will8R'
    , 'will9R', 'will10R', 'proc12close', 'proc13close'
    , 'proc14close', 'proc15close', 'wadl15close', 'adosc2AD'
    , 'adosc3AD', 'adosc4AD', 'adosc5AD', 'macd1530', 'cci15close'
    , 'bollinger15upper', 'bollinger15mid', 'bollinger15lower', 'paverage2open'
    , 'paverage2high', 'paverage2low', 'paverage2close', 'slope3high', 'slope4high', 'slope5high'
    , 'slope10high', 'slope20high', 'slope30high'
    , 'fourier10a0', 'fourier10a1', 'fourier10b1', 'fourier10w', 'fourier20a0'
    , 'fourier20a1', 'fourier20b1', 'fourier20w', 'fourier30a0'
    , 'fourier30a1', 'fourier30b1', 'fourier30w', 'sine5a0', 'sine5b1', 'sine5w'
    , 'sine6a0', 'sine6b1', 'sine6w', 'open', 'high', 'low', 'close']

clf_load_dt = joblib.load('Pkl/saved_model_dt.pkl')
clf_load_rf = joblib.load('Pkl/saved_model_rf.pkl')

df = masterFrameCleaned[list(columns)].values
a = []
std = StandardScaler()
newfeaturess = std.fit_transform(df)

a = []
a1 = []
newfeatures = newfeaturess[-1].reshape(1, 69)

predicted_dt = clf_load_dt.predict(newfeatures)
a = np.append(a, predicted_dt)

predicted_rf = clf_load_rf.predict(newfeatures)
a = np.append(a, predicted_rf)

newfeatures = newfeaturess[-2].reshape(1, 69)
predicted_dt = clf_load_dt.predict(newfeatures)
a1 = np.append(a1, predicted_dt)

predicted_rf = clf_load_rf.predict(newfeatures)
a1 = np.append(a1, predicted_rf)

print(a)
print(a1)