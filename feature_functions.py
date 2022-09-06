import threading

import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import mplfinance
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mplfinance.original_flavor import candlestick_ochl
from matplotlib.dates import date2num
from datetime import datetime
import sys
sys.setrecursionlimit(100000)
threading.stack_size(200000000)



class holder:
    1

# Heiken Ashi Candle
def heikenashi(prices,periods):

    """

    :param prices: dataframe of OHLC & volume data
    :param periods: periods for which to create the candles
    :return: heiken ashi OHLC candles

    """
    results = holder()

    dict = {}

    HAclose = prices[['open','high','close','low']].sum(axis=1)/4

    HAopen = HAclose.copy()

    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()

    HAlow = HAclose.copy()

    for i in range(1,len(prices)):

        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([prices.high.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = [['open','high','close','low']]

    df.index = df.index.droplevel(0)

    dict[periods[0]] = df
    results.candles = dict

    return results

# Detrender

def detrend(prices,method='difference'):

    """

    :param prices: dataframe of OHLC currency data
    :param method: method by which to detrend 'Linear' or 'Difference'
    :return: the detrended price series

    """
    if method == 'difference':
        detrended = prices.close[1:]-prices.close[:-1].values

    elif method == 'linear':

        x = np.arange(0,len(prices))
        y = prices.close.values

        model = LinearRegression()
        model.fit(x.reshape(-1,1),y.reshape(-1,1))

        trend = model.predict(x.reshape(-1,1))
        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend

    else:

        print('You did not input a valid method for detrending! Options are linear or difference')

    return detrended

# Fourier Series Expansion Fitting Function

def fseries(x,a0,a1,b1,w):

    """

    :param x: the hours(independent variable)
    :param a0: first Fourier series coefficient
    :param a1: second fourier series coefficient
    :param b1: third fourier series coefficient
    :param w: fourier series frequency
    :return: the value of the fourier function
    """
    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)

    return f

#Sine Series Expansion Fitting function

def sseries(x,a0,b1,w):

    """

    :param x: the hours(independent variable)
    :param a0: first sine series coefficient
    :param b1: third sine series coefficient
    :param w: sine series frequency
    :return: the value of the sine function
    """
    s = a0 + b1*np.sin(w*x)

    return s

# Fourier Serries Coefficient Calculator Function

def fourier(price,period,method='difference'):

    """

    :param price: OHLC
    :param period: List of periods for which to compute coefficients
    :param method: method by which to detrend the date
    :return: dict of dataframes containing coefficients for said periods
    """
    results = holder()
    dict = {}
    plot = False
    detrended = detrend(price, method)

    for i in range(0, len(period)):
        coeffs = []
        for j in range(period[i], len(price)):
            x = np.arange(0, period[i])
            y = detrended.iloc[j - period[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries, x, y)

                except (RuntimeError, OptimizeWarning):
                    res = np.empty((1, 4))
                    res[0, :] = np.NAN

            if plot == True:
                xt = np.linspace(0, period[i], 100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs) // 4, 4)))
        df = pd.DataFrame(coeffs, index=price.iloc[period[i]:len(price)].index)
        df.columns = ['a0', 'a1', 'b1', 'w']
        df = df.fillna(method='bfill')
        dict[period[i]] = df
    results.coeffs = dict

    return results

# Sine Serries Coefficient Calculator Function

def sine(price,period,method='difference'):

    """

    :param price: OHLC
    :param period: List of periods for which to compute coefficients
    :param method: method by which to detrend the date
    :return: dict of dataframes containing coefficients for said periods
    """
    results = holder()
    dict = {}
    plot = False
    detrended = detrend(price, method)

    for i in range(0, len(period)):
        coeffs = []
        for j in range(period[i], len(price)):
            x = np.arange(0, period[i])
            y = detrended.iloc[j - period[i]:j]
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sseries, x, y)

                except (RuntimeError, OptimizeWarning):
                    res = np.empty((1, 3))
                    res[0, :] = np.NAN

            if plot == True:
                xt = np.linspace(0, period[i], 100)
                yt = sseries(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs) // 3, 3)))
        df = pd.DataFrame(coeffs, index=price.iloc[period[i]:len(price)].index)
        df.columns = ['a0', 'b1', 'w']
        df = df.fillna(method='bfill')
        dict[period[i]] = df
    results.coeffs = dict

    return results

# Williams Accumulatuion Distribution Function

def wadl(price,period):

    """

    :param price: dataframe of OHLC prices
    :param period: (list) periods for which to calculate the function
    :return: williams accumulation distribution lines for each periods
    """

    results = holder()
    dict = {}

    for i in range(0, len(period)):

        WAD = []

        for j in range(period[i], len(price)):

            TRH = np.array([price.high.iloc[j], price.close.iloc[j - 1]]).max()
            TRL = np.array([price.low.iloc[j], price.close.iloc[j - 1]]).min()

            if price.close.iloc[j] > price.close.iloc[j - 1]:
                PM = price.close.iloc[j] - TRL
            elif price.close.iloc[j] < price.close.iloc[j - 1]:
                PM = price.close.iloc[j] - TRH
            elif price.close.iloc[j] == price.close.iloc[j - 1]:
                PM = 0
            else:
                print("error in wadl")

            AD = PM * price.volume.iloc[j]
            WAD = np.append(WAD, AD)

        WAD = WAD.cumsum()
        WAD = pd.DataFrame(WAD, index=price.iloc[period[i]:len(price)].index)
        WAD.columns = ['close']
        dict[period[i]] = WAD

    results.wadl = dict
    return results

# data resampling function

def OHLCresample(DataFrame,TimeFrame,column='ask'):

    """

    :param DataFrame: dataframe containing data that we want to resample
    :param TimeFrame: timeframe that we want for resampling
    :param column: which column we are resampling (bid or ask) default='ask'
    :return: resampled OHLC data for the given timeframe
    """
    grouped = DataFrame.groupby('Symbol')

    if np.any(DataFrame.columns == 'Ask'):

        if column == 'ask':
            ask = grouped['Ask'].resample(TimeFrame).ohlc()
            askVol = grouped['AskVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = askVol

        elif column == 'bid':
            bid = grouped['Bid'].resample(TimeFrame).ohlc()
            bidVol = grouped['BidVol'].resample(TimeFrame).count()
            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bidVol

        else:
            raise ValueError('error OHLCresample')

    elif np.any(DataFrame.columns == 'close'):
        open = grouped['open'].resample(TimeFrame).ohlc()
        close = grouped['close'].resample(TimeFrame).ohlc()
        high = grouped['high'].resample(TimeFrame).ohlc()
        low = grouped['low'].resample(TimeFrame).ohlc()
        askVol = grouped['volume'].resample(TimeFrame).count()

        resampled = pd.DataFrame(open)
        resampled['high'] = high
        resampled['low'] = low
        resampled['close'] = close
        resampled['volume'] = askVol

    resampled = resampled.dropna()

    return resampled


# Momentum Function

def momentum(prices,periods):

    """

    :param prices: dataframe of OHLC data
    :param periods: list of periods to calculate values
    :return: momentum indicator
    """
    results = holder()
    open = {}
    close = {}
    for i in range(0,len(periods)):

        open[periods[i]] = pd.DataFrame(prices.open.iloc[periods[i]:]-prices.open.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)
        close[periods[i]] = pd.DataFrame(prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values, index=prices.iloc[periods[i]:].index)

        open[periods[i]].columns = ['open']
        close[periods[i]].columns = ['close']

    results.open = open
    results.close = close

    return results

# Stochastic Oscillator function

def stochastic(prices,periods):

    """

    :param prices: OHLC dataframe
    :param periods: periods to calculate function value
    :return: Oscillator function values
    """
    results = holder()
    close = {}

    for i in range(0, len(periods)):
        Ks = []
        for j in range(periods[i], len(prices)):
            C = prices.close.iloc[j]
            H = prices.high.iloc[j - periods[i]:j - 1].max()
            L = prices.low.iloc[j - periods[i]:j - 1].min()
            if H == L:
                K = 0
            else:
                K = 100 * (C - L) / (H - L)
            Ks = np.append(Ks, K)

        df = pd.DataFrame(Ks, index=prices.iloc[periods[i]:len(prices)].index)
        df.columns = ['K']
        df['D'] = df.K.rolling(3).mean()
        df = df.dropna()
        close[periods[i]] = df
    results.close = close

    return results

# Williams Oscillator Function

def williams(prices,periods):

    """

    :param prices: OHLC price Data
    :param periods: list of periods to calculate function values
    :return: values of williams osc function
    """
    results = holder()
    close = {}
    for i in range(0, len(periods)):
        Rs = []
        for j in range(periods[i], len(prices)):

            C = prices.close.iloc[j]
            H = prices.high.iloc[j - periods[i]:j - 1].max()
            L = prices.low.iloc[j - periods[i]:j - 1].min()
            if H == L:
                R = 0
            else:
                R = 100 * (H - C) / (H - L)
            Rs = np.append(Rs, R)

        df = pd.DataFrame(Rs, index=prices.iloc[periods[i]:len(prices)].index)
        df.columns = ['R']
        df = df.dropna()
        close[periods[i]] = df

    results.close = close

    return results

# PROC function (price rate of change)
def proc(prices,periods):

    """

    :param prices: dataframe containing prices
    :param periods: periods for which to calculate proc
    :return: PROC for indicated periods
    """
    results = holder()
    proc = {}
    for i in range(0, len(periods)):
        proc[periods[i]] = pd.DataFrame(
            (prices.close.iloc[periods[i]:] - prices.close.iloc[:-periods[i]].values) / prices.close.iloc[
                                                                                        :-periods[i]].values)
        proc[periods[i]].columns = ['close']

    results.proc = proc
    return results

# Accumulation Distribution Oscillator

def adosc(prices,periods):

    """

    :param prices: OHLC DataFrame
    :param periods: periods for which to compute indicator
    :return: indicator values for indicated periods
    """
    results = holder()
    accdist = {}
    for i in range(0, len(periods)):
        AD = []
        for j in range(periods[i], len(prices)):

            C = prices.close.iloc[j]
            H = prices.high.iloc[j - periods[i]:j - 1].max()
            L = prices.low.iloc[j - periods[i]:j - 1].min()
            V = prices.volume.iloc[j]
            if H == L:
                CLV = 0
            else:
                CLV = ((C - L) - (H - C)) / (H - L)
            AD = np.append(AD, CLV * V)
        AD = AD.cumsum()
        AD = pd.DataFrame(AD, index=prices.iloc[periods[i]:len(prices)].index)
        AD.columns = ['AD']
        accdist[periods[i]] = AD

    results.AD = accdist

    return results

# MACD (Moving Average Convergence Divergence)

def macd(prices,periods):

    """

    :param prices: OHLC dataframe of prices
    :param periods: 1x2 array containing values for the EMAs
    :return: MACD for given periods
    """
    results = holder()

    EMA1 = prices.close.ewm(span=periods[0]).mean()
    EMA2 = prices.close.ewm(span=periods[1]).mean()

    MACD = pd.DataFrame(EMA1 - EMA2)
    MACD.columns = ['L']

    sigMACD = MACD.rolling(3).mean()
    sigMACD.columns = ['SL']

    results.line = MACD
    results.signal = sigMACD

    return results

# CCI (Commodity Channel Index)

def cci(prices,periods):

    """

    :param prices: OHLC dataframe of price data
    :param periods: periods for which to compute the indicator
    :return: CCI for the given periods
    """
    results = holder()
    cci = {}

    for hours in periods:
        # Moving Average 
        ma = prices.close.rolling(hours).mean()
        std = prices.close.rolling(hours).std()

        # Mean Deviation
        md = (prices.close - ma) / std

        cci[hours] = pd.DataFrame((prices.close - ma) / (0.015 * md))
        cci[hours].columns = ['close']

    results.cci = cci

    return results

# Bollinger Bands

def bollinger(prices,periods,deviations):

    """

    :param prices: OHLC data
    :param periods: periods for which to compute the bollinger bands
    :param deviations: deviations to use when calculating bands (upper & lower)
    :return: bollinger bands
    """
    results = holder()
    boll = {}
    for i in range(0, len(periods)):
        mid = prices.close.rolling(periods[i]).mean()
        std = prices.close.rolling(periods[i]).std()

        upper = mid + deviations * std
        lower = mid - deviations * std

        df = pd.concat((upper, mid, lower), axis=1)
        df.columns = ['upper', 'mid', 'lower']

        boll[periods[i]] = df

    results.bands = boll
    return results

# Price Averages

def paverage(prices,periods):

    """

    :param prices: OHLC data
    :param periods: list of periods for which to calculate indicator values
    :return: averages over the given periods
    """
    results = holder()

    avs = {}

    for i in range(0,len(periods)):

        avs[periods[i]] = pd.DataFrame(prices[['open','high','close','low']].rolling(periods[i]).mean())

    results.avs = avs

    return results

# Slope Functions

def slopes(prices,periods):

    """

    :param prices: OHLC price DataFrame
    :param periods: periods to get the indicator values
    :return: slopes over given periods
    """
    results = holder()

    slope = {}

    for i in range(0, len(periods)):
        ms = []
        for j in range(periods[i], len(prices)):
            y = prices.high.iloc[j - periods[i]:j].values
            x = np.arange(0, len(y))

            res = stats.linregress(x, y=y)
            m = res.slope
            ms = np.append(ms, m)

        ms = pd.DataFrame(ms, index=prices.iloc[periods[i]:len(prices)].index)
        ms.columns = ['high']
        slope[periods[i]] = ms

    results.slope = slope

    return results


def Market(prices, periods):
    results = holder()
    slope = {}
    m = 3
    prices.index = pd.to_datetime(prices.index)
    prices['B'] = prices.index.minute

    for i in range(0, len(periods)):
        ms = []
        for j in range(periods[i], len(prices) - 60):
            if prices.B.iloc[j] == 0:
                if prices.open[j] < prices.close[j + 60]:
                    m = 1
                elif prices.open[j] >= prices.close[j + 60]:
                    m = 0
            ms.append(m)
        ms = pd.DataFrame(ms, index=prices.iloc[periods[i]:len(prices) - 60].index)
        ms.columns = ['Market']
        slope[periods[i]] = ms

    results.slope = slope

    return results


# ------------------------------------------------------------------------------
def resamble(prices):
    results = holder()
    maxHigh = 0
    maxLow = 0
    msOpen = []
    msHigh = []
    msLow = []
    msClose = []
    msVolume = []
    prices.index = pd.to_datetime(prices.index)
    prices['B'] = prices.index.minute
    Open = prices.open.iloc[0]
    High = prices.high.iloc[0]
    Low = prices.low.iloc[0]
    Close = prices.close.iloc[0]
    Volume = prices.volume.iloc[0]

    for i in range(0, len(prices)):
        x = 0
        if prices.B.iloc[i] == 0:
            Openn = prices.open.iloc[i]
            maxHigh = prices.open.iloc[i]
            maxLow = prices.open.iloc[i]
            Volume = prices.volume.iloc[i]
            x = 1
        if maxHigh < prices.high.iloc[i]:
            maxHigh = prices.high.iloc[i]
        if maxLow > prices.low.iloc[i]:
            maxLow = prices.low.iloc[i]

        Open = Openn
        High = maxHigh
        Low = maxLow
        Close = prices.close.iloc[i]
        if x != 1:
            Volume = Volume + prices.volume.iloc[i]

        msOpen = np.append(msOpen, Open)
        msHigh = np.append(msHigh, High)
        msLow = np.append(msLow, Low)
        msClose = np.append(msClose, Close)
        msVolume = np.append(msVolume, Volume)

    df1 = pd.DataFrame(msOpen, index=prices.iloc[0:len(prices)].index)
    df1.columns = ['open']
    df2 = pd.DataFrame(msHigh, index=prices.iloc[0:len(prices)].index)
    df2.columns = ['high']
    df3 = pd.DataFrame(msLow, index=prices.iloc[0:len(prices)].index)
    df3.columns = ['low']
    df4 = pd.DataFrame(msClose, index=prices.iloc[0:len(prices)].index)
    df4.columns = ['close']
    df5 = pd.DataFrame(msVolume, index=prices.iloc[0:len(prices)].index)
    df5.columns = ['volume']
    result = pd.concat([df1, df2, df3, df4, df5], axis=1, sort=False)

    results = result

    return results







