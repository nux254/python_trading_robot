import pandas as pd
import plotly as py
from plotly import tools
from plotly import subplots
import plotly.graph_objs as go

# load up data and create moving average
df = pd.read_csv('EURUSDhours.csv')
df.columns = ['date','open','high','low','close','volume']
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df = df.drop_duplicates(keep=False)

ma = df.close.rolling(center=False,window=30).mean()

# Get function data from selected functions:


# Plot
trace0 = go.Ohlc(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name='Currency Quote')
trace1 = go.Scatter(x=df.index,y=ma)
trace2 = go.Bar(x=df.index,y=df.volume)

data = [trace0,trace1,trace2]

fig = subplots.make_subplots(rows=2,cols=1,shared_xaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)

py.offline.plot(fig,filename='tutorial.html')