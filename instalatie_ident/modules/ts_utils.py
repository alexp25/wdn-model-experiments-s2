import numpy as np
from typing import List
from modules.graph import Timeseries
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def remove_outliers(data, maxval):
    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    for j in range(cols):
        for i in range(rows):
            if i > 1:
                if data[i][j] > maxval:
                    data[i][j] = data[i-1][j]

    return data


def reorder(x, order):
    x_ord = []

    for (i, ord) in enumerate(order):
        x_ord.append(x[ord])

    return np.array(x_ord)


def reorder2d(x, order):
    sdata = np.shape(x)
    rows = sdata[0]
    cols = sdata[1]

    x_ord = []
    for i in range(rows):
        new_row = []
        for (j, ord) in enumerate(order):
            new_row.append(x[i, ord])
        x_ord.append(new_row)

    return np.array(x_ord)


def create_timeseries(data, header, datax=None, colorspec=None):
    tss: List[Timeseries] = []
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    print(sdata)

    if colorspec is None:
        colors = cm.rainbow(np.linspace(0, 1, cols))
        # colors = cm.viridis(np.linspace(0, 1, cols))
    else:
        colors = colorspec

    for j in range(cols):
        ts: Timeseries = Timeseries()
        try:
            ts.label = header[j]
        except:
            ts.label = "unknown"
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        for i in range(rows):
            if datax is not None:
                ts.x.append(datax[i][j])
            else:
                ts.x.append(i)
            ts.y.append(data[i][j])
        tss.append(ts)
        ts = None

    return tss

def normalize(_d, to_sum=True, copy=True):
    # d is a (n x dimension) np array
    d = _d if not copy else np.copy(_d)
    d -= np.min(d, axis=0)
    div = (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
    if div != 0:
        d /= div
    else:
        d /= 1000000000
    return d
    
def order_data(data, header, order):
    header1 = reorder(header, order)
    data1 = reorder2d(data, order)
    return data1, header1