from codes.price_action.zizag_indicator import Zizag
import numpy as np

class Trendline:

    def __init__(self, InpDepth=12, InpDeviation=5, InpBackstep=3, Point=0.00001, Min_dist=0, fibo=25, tolerance=200, Intersection_ab=1, Intersection_bc=1, Score_tolerance=30, Score_limit=3):
        self.InpDepth = InpDepth
        self.InpDeviation = InpDeviation
        self.InpBackstep = InpBackstep
        self.Point = Point

        self.zizag = Zizag(InpDepth, InpDeviation, InpBackstep, Point)
        self.Min_dist = Min_dist
        self.fibo = fibo
        self.tolerance = tolerance
        self.Intersection_ab = Intersection_ab
        self.Intersection_bc = Intersection_bc
        self.Score_tolerance = Score_tolerance
        self.Score_limit = Score_limit
        # self.line0 = []
        self.UpTrend = []
        self.DownTrend = []
        # self.a, self.b, self.cross_ab, self.cross_bc = 0,0,0,0
        pass


    def onCalculate(self, prev_calculated, prev_i, window_size, open, high, low, close):
        self.remove_old_lines(prev_i, window_size)
        self.zizag.on_calculation(prev_calculated, high, low)
        mass = []
        rates_total = len(close)
        max_zizag = close[0]
        min_zizag = close[0]
        z = 0
        for shift in range(rates_total):
            zizag = self.zizag.ExtZigzagBuffer[shift]

            if zizag > 0:
                if zizag >= max_zizag and zizag == high[shift]:
                    mass.append({
                        'price': zizag,
                        'pos': shift,
                        'hpoint': True,
                        'lpoint': False,
                    })
                    z += 1
                    max_zizag = zizag

                if zizag <= min_zizag and zizag == low[shift]:
                    mass.append({
                        'price': zizag,
                        'pos': shift,
                        'hpoint': False,
                        'lpoint': True,
                    })
                    z += 1
                    min_zizag = zizag

        # +------------------------------------------------------------------+

        # for i in range(z):
        #     if mass[i]['hpoint']:
        #         self.line0[mass[i]['pos']] = mass[i]['price']
        #     if mass[i]['lpoint']:
        #         self.line0[mass[i]['pos']] = mass[i]['price']

        # +------------------------------------------------------------------+

        for j in range(z-1, -1, -1):
            if mass[j]['hpoint']:
                for i in range(j-1, -1, -1):
                    if mass[i]['hpoint']:
                        if i < j:
                            ax = mass[j]['pos']
                            bx = mass[i]['pos']
                            ratio = ((ax-bx) * 100.0 / ax)

                            if self.fibo < ratio < 100 - self.fibo:
                                if bx > self.Min_dist and (ax - bx) > self.Min_dist:
                                    ay = mass[j]['price']
                                    by = mass[i]['price']

                                    coef = (ay - by) / (ax - bx)

                                    price = close[0]

                                    # deviation = (ay + coef * by) - price
                                    beta = ay - coef*ax
                                    deviation = 1*coef + beta - price
                                    cross_bc = 0
                                    cross_ab = 0

                                    if abs(deviation) < self.tolerance * self.Point:
                                        last_n = ax
                                        # number of crossings from point ax to point bx
                                        for n in range(ax, bx, -1):
                                            if (coef * n + beta) >= min(open[n], close[n]) and (coef * (n+1) + beta) <= max(open[n], close[n]) and last_n-n>2:
                                                last_n = n
                                                cross_ab += 1
                                        # number of crossings from point bx to the end
                                        last_n = bx

                                        for n in range(bx-1, -1, -1):
                                            if (coef * n + beta) >= min(open[n], close[n]) and (coef * (n + 1) + beta) <= max(open[n], close[n]) and last_n-n>2:
                                                last_n = n
                                                cross_bc += 1

                                        score = 0
                                        for i in range(len(self.zizag.ExtZigzagBuffer)):
                                            zizag = self.zizag.ExtZigzagBuffer[i]
                                            if zizag > 0 and abs(zizag - (coef * i + beta)) <= self.Score_tolerance * self.Point:
                                                score += 1
                                        if cross_bc <= self.Intersection_bc and cross_ab <= self.Intersection_ab and score >= self.Score_limit:

                                            self.DownTrend.append({
                                                'prev_i': prev_i,
                                                'ax': ax,
                                                'bx': bx,
                                                'ay': ay,
                                                'by': by,
                                                'dst': abs(deviation),
                                                'coef': coef,
                                                'beta': beta,
                                                'score': score
                                            })

        # +------------------------------------------------------------------+

        for j in range(z-1, -1, -1):
            if mass[j]['lpoint']:
                for i in range(j-1, -1, -1):
                    if mass[i]['lpoint']:
                        if i < j:
                            ax = mass[j]['pos']
                            bx = mass[i]['pos']
                            ratio = ((ax-bx) * 100.0 / ax)

                            if self.fibo < ratio < 100 - self.fibo:
                                if bx > self.Min_dist and (ax - bx) > self.Min_dist:
                                    ay = mass[j]['price']
                                    by = mass[i]['price']

                                    coef = (ay - by) / (ax - bx)

                                    price = close[0]

                                    # deviation = (ay + coef * by) - price
                                    beta = ay - coef*ax
                                    deviation = 1*coef + beta - price

                                    cross_bc = 0
                                    cross_ab = 0

                                    if abs(deviation) < self.tolerance * self.Point:
                                        last_n = ax
                                        # number of crossings from point ax to point bx
                                        for n in range(ax, bx, -1):
                                            if (coef * n + beta) >= min(open[n], close[n]) and (coef * (n+1) + beta) <= max(open[n], close[n]) and last_n-n>2:
                                                last_n = n
                                                cross_ab += 1
                                        # number of crossings from point bx to the end
                                        last_n = bx

                                        for n in range(bx-1, -1, -1):
                                            if (coef * n + beta) >= min(open[n], close[n]) and (coef * (n + 1) + beta) <= max(open[n], close[n]) and last_n-n>2:
                                                last_n = n
                                                cross_bc += 1

                                        score = 0
                                        for i in range(len(self.zizag.ExtZigzagBuffer)):
                                            zizag = self.zizag.ExtZigzagBuffer[i]
                                            if zizag > 0 and abs(zizag - (coef * i + beta)) <= self.Score_tolerance * self.Point:
                                                score += 1
                                        if cross_bc <= self.Intersection_bc and cross_ab <= self.Intersection_ab and score >= self.Score_limit:

                                            self.UpTrend.append({
                                                'prev_i': prev_i,
                                                'ax': ax,
                                                'bx': bx,
                                                'ay': ay,
                                                'by': by,
                                                'dst': abs(deviation),
                                                'coef': coef,
                                                'beta': beta,
                                                'score': score,
                                            })
        self.UpTrend = self.remove_duplicate_line(self.UpTrend)
        self.DownTrend = self.remove_duplicate_line(self.DownTrend)
        return rates_total

    def remove_duplicate_line(self, lst):
        should_remove = []
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if abs(lst[i]['coef'] - lst[j]['coef']) <= 1/1e4 and abs(lst[i]['beta']-lst[j]['beta']) <= 1/1e4 and (lst[i]['ax']-lst[i]['prev_i'] == lst[j]['ax']-lst[j]['prev_i'] or lst[i]['prev_i'] == lst[j]['prev_i']):
                    should_remove.append(i)
                    break
        return [v for i, v in enumerate(lst) if i not in should_remove]

    def remove_old_lines(self, prev_i, window_size):
        self.DownTrend = list(filter(lambda x: prev_i - x['prev_i'] < 1, self.DownTrend))
        self.UpTrend = list(filter(lambda x: prev_i - x['prev_i'] < 1, self.UpTrend))

    @staticmethod
    def get_price_line_on_pos(line, i):
        return line['beta'] + line['coef'] * (line['prev_i'] - i)

    def is_trendline_down(self):
        for i in range(len(self.zizag.ExtHighBuffer)):
            if self.zizag.ExtHighBuffer[i]:
                for j in range(i+1, len(self.zizag.ExtHighBuffer)):
                    if self.zizag.ExtHighBuffer[j]:
                        if self.zizag.ExtHighBuffer[i] <= self.zizag.ExtHighBuffer[j]:
                            for a in range(len(self.zizag.ExtLowBuffer)):
                                if self.zizag.ExtLowBuffer[a]:
                                    for b in range(i + 1, len(self.zizag.ExtLowBuffer)):
                                        if self.zizag.ExtLowBuffer[b]:
                                            if self.zizag.ExtLowBuffer[a] <= self.zizag.ExtLowBuffer[b]:
                                                return True
                                            else:
                                                return False
                        else:
                            return False
        return False

    def is_trendline_up(self):
        for i in range(len(self.zizag.ExtHighBuffer)):
            if self.zizag.ExtHighBuffer[i]:
                for j in range(i+1, len(self.zizag.ExtHighBuffer)):
                    if self.zizag.ExtHighBuffer[j]:
                        if self.zizag.ExtHighBuffer[i] >= self.zizag.ExtHighBuffer[j]:
                            for a in range(len(self.zizag.ExtLowBuffer)):
                                if self.zizag.ExtLowBuffer[a]:
                                    for b in range(i + 1, len(self.zizag.ExtLowBuffer)):
                                        if self.zizag.ExtLowBuffer[b]:
                                            if self.zizag.ExtLowBuffer[a] >= self.zizag.ExtLowBuffer[b]:
                                                return True
                                            else:
                                                return False
                        else:
                            return False
        return False

    def get_last_low(self):
        for low in self.zizag.ExtLowBuffer:
            if low:
                return low
        return None

    def get_last_high(self):
        for high in self.zizag.ExtHighBuffer:
            if high:
                return high
        return None

if __name__ == '__main__':
    import plotly as py
    import plotly.graph_objs as go

    py.tools.set_credentials_file(username='mbehroozikhah', api_key='evr0jZbxNMUeP5Hysifc')
    import pandas_datareader as web
    from datetime import datetime
    import pandas

    # df = web.DataReader("aapl", 'morningstar').reset_index()

    df = pandas.read_csv("../../data/USDJPY_M5.csv")
    df = df[-900:-800].reset_index()
    trace = go.Candlestick(x=df.index,
                           open=df.Open,
                           high=df.High,
                           low=df.Low,
                           close=df.Close)
    data = [trace]

    trendline = Trendline(tolerance=100, Point=0.001, Intersection_ab=5, Intersection_bc=5, fibo=25, InpDepth=5, InpDeviation=5, InpBackstep=3, Score_limit=2)
    trendline.onCalculate(0, 0, df.Open.values[::-1], df.High.values[::-1], df.Low.values[::-1], df.Close.values[::-1])

    trendline.UpTrend = np.array(trendline.UpTrend)[::-1]
    trendline.DownTrend = np.array(trendline.DownTrend)[::-1]
    print(df)
    print(trendline.UpTrend)

    # x = df.Date[zizag.ExtZigzagBuffer>0].values
    # y = zizag.ExtZigzagBuffer[zizag.ExtZigzagBuffer>0]
    # print(x)
    # print(y)
    #
    layout = {
        'title': 'The Great Recession',
        'yaxis': {'title': 'AAPL Stock'},
        'shapes': [],
    }
    for i in range(trendline.DownTrend.size):
        layout['shapes'].append({
            'type': 'line',
            'x0': df.index[df.DateTime.size - trendline.DownTrend[i]['ax'] - 1],#df.DateTime[df.DateTime.size - trendline.DownTrend[i]['bx']],
            'x1': df.index[df.DateTime.size-1],#df.DateTime[df.DateTime.size - trendline.DownTrend[i]['ax']],
            'y0': trendline.DownTrend[i]['ay'],
            'y1': trendline.DownTrend[i]['beta'],
            'xref': 'x', 'yref': 'y',
            'line': {'color': 'rgb(255,0,0)', 'width': trendline.DownTrend[i]['score']/3}
        })
    for i in range(trendline.UpTrend.size):
        layout['shapes'].append({
            'type': 'line',
            'x0': df.index[df.DateTime.size - trendline.UpTrend[i]['ax'] - 1],#df.DateTime[df.DateTime.size - trendline.DownTrend[i]['bx']],
            'x1': df.index[df.DateTime.size-1],#df.DateTime[df.DateTime.size - trendline.DownTrend[i]['ax']],
            'y0': trendline.UpTrend[i]['ay'],
            'y1': trendline.UpTrend[i]['beta'],
            'xref': 'x', 'yref': 'y',
            'line': {'color': 'rgb(0,255,0)', 'width': trendline.UpTrend[i]['score']/3}
        })

    # for i in range(trendline.UpTrend.size):
    #     layout['shapes'].append({
    #         'type': 'line',
    #         'x0': df.index[df.DateTime.size - trendline.UpTrend[i]['bx'] - 1],
    #         'x1': df.index[df.DateTime.size - trendline.UpTrend[i]['ax'] - 1],
    #         'y0': trendline.UpTrend[i]['bx']*trendline.UpTrend[i]['coef'] + trendline.UpTrend[i]['beta'],
    #         'y1': trendline.UpTrend[i]['ax']*trendline.UpTrend[i]['coef'] + trendline.UpTrend[i]['beta'],
    #         'xref': 'x', 'yref': 'y',
    #         'line': {'color': 'rgb(0,255,0)', 'width': 1}
    #     })

    print(layout)
    fig = dict(data=data, layout=layout)
    py.offline.plot(fig)