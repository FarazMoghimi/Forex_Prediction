def single_candle(high, low, close):
    res = []
    for i in range(len(close)):
        if close[i] is None:
            res.append(None)
            continue
        diff = (high[i] - low[i]) / 3
        if close[i] < diff + low[i]:
            res.append(-1)
        elif close[i] <= 2*diff + low[i]:
            res.append(0)
        else:
            res.append(+1)
    return res