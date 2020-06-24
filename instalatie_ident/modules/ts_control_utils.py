import numpy as np
from typing import List
from modules.graph import Timeseries
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_flow_metrics(data2):
    err = [abs(e[0] - e[1]) for e in data2]
    mean = np.mean([e[0] for e in data2])
    stdev = np.std(err)
    # print(mean, stdev)
    return mean, stdev


def get_flow_metrics_noref(data2):
    err = [e for e in data2]
    mean = np.mean([e[0] for e in data2])
    stdev = np.std(err)
    # print(mean, stdev)
    return mean, stdev


def get_control_metrics(data3):
    d = [e[0] for e in data3]
    mean = np.mean(d)
    stdev = np.std(d)
    # print(mean, stdev)
    return mean, stdev


def split_data_bookmarks(data):
    bookmarks = []
    b1 = data[0]
    i1 = 0
    for i, d in enumerate(data):
        if d != b1 and i > 0:
            b1 = data[i-1]
            i2 = i-1
            if i1 != i2:
                bookmarks.append([i1, i2])
            i1 = i
    return bookmarks


def run_multiple_bookmarks(data, bm, fn):
    res = []
    if len(bm) == 0:
        res = fn(data)
    else:
        for b in bm:
            d = data[b[0]:b[1]]
            res.append(fn(d))
    return res

def extract_bookmarks_multitrack(data):
    bm = []
    prev_sample = data[0]
    s = np.shape(data)
    bm.append(0)
    for i in range(s[0]):
        if not np.array_equal(data[i], prev_sample):
            bm.append(i)
            prev_sample = data[i]   
    return bm


def extract_bookmarks(data, stretch=0):
    bm = []
    reset_val = data[0]
    ts = 0
    cue_vect = []
    final_peak = reset_val

    for (i, d) in enumerate(data):

        if i == 0:
            pass
        else:
            if data[i-1] != d:
                cue_vect.append(i)
            if data[i-1] < d:
                if data[i-1] == reset_val:
                    bm.append(i)
            if data[i-1] > d:
                final_peak = data[i-1]
                if d == reset_val:
                    bm.append(i)
            if i == len(data) - 1 and d == final_peak:
                bm.append(i)

    ts = cue_vect[1] - cue_vect[0]
    # print(cue_vect)
    for i in range(len(bm)):
        if i % 2 == 0:
            bm[i] -= stretch*ts
        else:
            bm[i] += stretch*ts
    # quit()
    return bm
