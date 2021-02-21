import numpy as np

class Zizag:

    def __init__(self, InpDepth=12, InpDeviation=5, InpBackstep=3, Point=0.00001):
        self.InpDepth = InpDepth
        self.InpDeviation = InpDeviation
        self.InpBackstep = InpBackstep
        self.Point = Point
        self.ExtLevel = 3
        self.ExtZigzagBuffer = []
        self.ExtHighBuffer = []
        self.ExtLowBuffer = []
        if (InpBackstep >= InpDepth):
            print("Backstep cannot be greater or equal to Depth")

    def InitializeAll(self, Bars):
        self.ExtZigzagBuffer = [0.0 for _ in range(Bars)]
        self.ExtHighBuffer = [0.0 for _ in range(Bars)]
        self.ExtLowBuffer = [0.0 for _ in range(Bars)]
        return Bars-self.InpDepth

    def iHighest(self, data, start, count):
        return np.max(data[start:(start+count)])

    def iLowest(self, data, start, count):
        return np.min(data[start:(start+count)])

    def on_calculation(self, prev_calculated, high, low):
        rates_total = high.size
        i, limit, counterZ, whatlookfor = 0, 0, 0, 0
        back, pos, lasthighpos = 0, 0, 0
        lastlowpos = 0
        extremum = 0.0
        curlow = 0.0
        curhigh = 0.0
        lasthigh = 0.0
        lastlow = 0.0

        # --- check for history and inputs
        if rates_total < self.InpDepth or self.InpBackstep >= self.InpDepth:
            return 0
        # --- first calculations
        if prev_calculated == 0:
            limit = self.InitializeAll(high.size)
        else:
            # --- find first extremum in the depth ExtLevel or 100 last bars
            i, counterZ = 0, 0
            while counterZ < self.ExtLevel and i < 100:
                if self.ExtZigzagBuffer[i] != 0.0:
                    counterZ += 1
                i+=1
            # --- no extremum found - recounting all from begin
            if counterZ == 0:
                limit = self.InitializeAll(high.size)
            else:
                # --- set start position to found extremum position
                limit = i - 1
                # --- what kind of extremum?
                if self.ExtLowBuffer[i] != 0.0:
                    # --- low extremum
                    curlow = self.ExtLowBuffer[i]
                    # --- will look for the next high extremum
                    whatlookfor = 1
                else:
                    # --- high extremum
                    curhigh = self.ExtHighBuffer[i]
                    # --- will look for the next low extremum
                    whatlookfor = -1
                # --- clear the rest data
                for i in range(limit-1, -1, -1):
                    self.ExtZigzagBuffer[i] = 0.0
                    self.ExtLowBuffer[i] = 0.0
                    self.ExtHighBuffer[i] = 0.0

        # --- main loop
        for i in range(limit, -1, -1):
            # --- find lowest low in depth of bars
            extremum = self.iLowest(low, i, self.InpDepth)
            # --- this lowest has been found previously
            if extremum == lastlow:
                extremum = 0.0
            else:
                # --- new last low
                lastlow = extremum
                # --- discard extremum if current low is too high
                if low[i] - extremum > self.InpDeviation * self.Point:
                    extremum = 0.0
                else:
                    # --- clear previous extremums in backstep bars
                    for back in range(1, self.InpBackstep):
                        pos = i + back
                        if self.ExtLowBuffer[pos] != 0 and self.ExtLowBuffer[pos] > extremum:
                            self.ExtLowBuffer[pos]=0.0
            # --- found extremum is current low
            if low[i] == extremum:
                self.ExtLowBuffer[i] = extremum
            else:
                self.ExtLowBuffer[i] = 0.0
            # --- find highest high in depth of bars
            extremum = self.iHighest(high, i, self.InpDepth)
            # --- this highest has been found previously
            if extremum == lasthigh:
                extremum = 0.0
            else:
                # --- new last high
                lasthigh = extremum
                # --- discard extremum if current high is too low
                if extremum - high[i] > self.InpDeviation * self.Point:
                    extremum = 0.0
                else:
                    # --- clear previous extremums in backstep bars
                    for back in range(1, self.InpBackstep):
                        pos=i+back
                        if self.ExtHighBuffer[pos] != 0 or self.ExtHighBuffer[pos] < extremum:
                            self.ExtHighBuffer[pos]=0.0

            # --- found extremum is current high
            if high[i] == extremum:
                self.ExtHighBuffer[i] = extremum
            else:
                self.ExtHighBuffer[i] = 0.0

        if whatlookfor == 0:
            lastlow = 0.0
            lasthigh = 0.0
        else:
            lastlow = curlow
            lasthigh = curhigh

        for i in range(limit, -1, -1):
            if whatlookfor == 0: # look for peak or lawn
                if lastlow == 0.0 and lasthigh == 0.0:
                    if self.ExtHighBuffer[i] != 0.0:
                        lasthigh = high[i] #TODO: High[i]
                        lasthighpos = i
                        whatlookfor = -1
                        self.ExtZigzagBuffer[i] = lasthigh

                    if self.ExtLowBuffer[i] != 0.0:
                        lastlow=low[i] #TODO: Low[i]
                        lastlowpos=i
                        whatlookfor=1
                        self.ExtZigzagBuffer[i]=lastlow

            if whatlookfor == 1: # look for peak
                if self.ExtLowBuffer[i] != 0.0 and self.ExtLowBuffer[i] < lastlow and self.ExtHighBuffer[i] == 0.0:
                    self.ExtZigzagBuffer[lastlowpos] = 0.0
                    lastlowpos = i
                    lastlow = self.ExtLowBuffer[i]
                    self.ExtZigzagBuffer[i] = lastlow
                if self.ExtHighBuffer[i] != 0.0 and self.ExtLowBuffer[i] == 0.0:
                    lasthigh = self.ExtHighBuffer[i]
                    lasthighpos = i
                    self.ExtZigzagBuffer[i] = lasthigh
                    whatlookfor = -1

            if whatlookfor == -1: # look for lawn
                if self.ExtHighBuffer[i] != 0.0 and self.ExtHighBuffer[i] > lasthigh and self.ExtLowBuffer[i] == 0.0:
                    self.ExtZigzagBuffer[lasthighpos] = 0.0
                    lasthighpos = i
                    lasthigh = self.ExtHighBuffer[i]
                    self.ExtZigzagBuffer[i] = lasthigh
                if self.ExtLowBuffer[i] != 0.0 and self.ExtHighBuffer[i] == 0.0:
                    lastlow = self.ExtLowBuffer[i]
                    lastlowpos = i
                    self.ExtZigzagBuffer[i] = lastlow
                    whatlookfor = 1
        return rates_total


# import plotly as py
# import plotly.graph_objs as go
#
# import pandas_datareader as web
# from datetime import datetime
#
# df = web.DataReader("aapl", 'morningstar').reset_index()
#
# trace = go.Candlestick(x=df.Date,
#                        open=df.Open,
#                        high=df.High,
#                        low=df.Low,
#                        close=df.Close)
# data = [trace]
#
# zizag = Zizag()
# zizag.on_calculation(0, df.High.values[::-1], df.Low.values[::-1])
# print('ALLBuffer', zizag.ExtZigzagBuffer)
# print('high', df.High.values[::-1])
# print('low', df.Low.values[::-1])
# zizag.ExtZigzagBuffer = np.array(zizag.ExtZigzagBuffer)[::-1]
# x = df.Date[zizag.ExtZigzagBuffer>0].values
# y = zizag.ExtZigzagBuffer[zizag.ExtZigzagBuffer>0]
# print(x)
# print(y)
#
# layout = {
#     'title': 'The Great Recession',
#     'yaxis': {'title': 'AAPL Stock'},
#     'shapes': [],
#     # 'shapes': [{
#     #     'type': 'line',
#     #     'x0': '2016-12-09', 'x1': '2012-12-9',
#     #     'y0': 0, 'y1': .5, 'xref': 'x', 'yref': 'paper',
#     #     'line': {'color': 'rgb(30,30,30)', 'width': 1}
#     # }],
#     # 'annotations': [{
#     #     'x': '2016-12-09', 'y': 0.05, 'xref': 'x', 'yref': 'paper',
#     #     'showarrow': False, 'xanchor': 'left',
#     #     'text': 'Increase Period Begins'
#     # }]
# }
# for i in range(x.size-1):
#     layout['shapes'].append({
#         'type': 'line',
#         'x0': str(x[i])[:10],
#         'x1': str(x[i+1])[:10],
#         'y0': y[i],
#         'y1': y[i+1],
#         'xref': 'x', 'yref': 'y',
#         'line': {'color': 'rgb(10,10,10)', 'width': 1}
#     })
# print(layout)
# fig = dict(data=data, layout=layout)
# py.offline.plot(fig)
#print(zizag.ExtLowBuffer)