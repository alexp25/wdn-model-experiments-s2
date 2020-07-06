import numpy as np
import pandas as pd

# import our modules
from modules import loader, graph
from modules.graph import Timeseries
import time
import os
import yaml
from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm


root_data_folder = "./data"
filenames = ["exp_245"]
title = "PID"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/mmc"
# filenames = ["exp_397"]
# title = "MMC"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/mmc_weights"
# filenames = ["exp_399"]
# title = "MMC-W"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/mmc_ai"
# filenames = ["exp_400"]
# title = "MMC AI"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/mmc_ai_weights"
# filenames = ["exp_401"]
# title = "MMC-W AI"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/single"
# filenames = ["exp_402"]
# title = "PID M0"

# root_data_folder = "./data"
# root_data_folder += "/mmc_eval/single"
# filenames = ["exp_405"]
# title = "PID M7"


# title = "MMC AI"
# title = "MMC-W"
# title = "MMC-W AI"
# title = "PID"

bookmarks = [188, 282, 375, 469, 563, 657]
bookmarks = []

show_extra = True
show_extra = False

frame_size = 8

start_index_bm = 1
stop_index_bm = start_index_bm + frame_size

start_index_bm = 0
stop_index_bm = None

start_index_override = 5000
stop_index_override = 12500


def remove_outliers(data, maxval):
    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    for j in range(cols):
        for i in range(rows):
            if i > 1:
                if data[i][j] - data[i-1][j] > maxval:
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


def create_timeseries(data, header):
    tss: List[Timeseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    # colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

    for j in range(cols):
        ts: Timeseries = Timeseries()
        ts.label = header[j]
        ts.color = colors[ck]
        ck += 1
        if ck >= len(colors):
            ck = 0

        for i in range(rows):
            ts.x.append(i)
            ts.y.append(data[i][j])

        tss.append(ts)
        ts = None

    return tss


def trim_ts(tss: List[Timeseries], start, n):
    for ts in tss:
        ts.x = ts.x[start:n]
        ts.y = ts.y[start:n]


def order_data(data, header, order):
    header_valves = reorder(header, order)
    data_valves = reorder2d(data, order)
    return data_valves, header_valves


def get_flow_metrics(data_flow):
    err = [abs(e[0] - e[1]) for e in data_flow]
    mean = np.mean([e[0] for e in data_flow])
    stdev = np.std(err)
    # print(mean, stdev)
    return mean, stdev


def get_control_metrics(data_pump):
    d = [e[0] for e in data_pump]
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


def split_data_bookmarks_2d(data):
    bookmarks = []
    s = np.shape(data)
    n_rows = s[0]
    n_cols = s[1]
    b1 = data[0, :]
    i1 = 0
    for i in range(n_rows):
        if not np.array_equal(b1, data[i, :]) and i > 0:
            b1 = data[i-1, :]
            i2 = i - 1
            if i1 != i2:
                bookmarks.append([i1, i2])
            i1 = i
    return bookmarks


def run_multiple_bookmarks(data, bm, fn):
    res = []
    for b in bm:
        d = data[b[0]:b[1]]
        res.append(fn(d))
    return res


# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    data, header = loader.load_dataset_full_with_header(data_file)

    print(header)
    print(len(header))
    for i, h in enumerate(header):
        print(i, h)

    # order = [2,3,5,6,7,8,9,10,11,12,4]
    # order = [10, 20]

    # data = data[200:]

    # data_valves, header_valves = order_data(data, header, range(14, 19))

    # valves: 14-19
    data_valves, header_valves = order_data(data, header, range(14, 19+1))
    # data_valves, header_valves = order_data(data, header, [14])
    # flow: 2-12
    # data_flow, header_flow = order_data(data, header, range(2, 12))
    # [flow, ref=20]
    data_flow, header_flow = order_data(data, header, [10, 20])
    # pump: 13
    data_pump, header_pump = order_data(data, header, [13])

    print(np.shape(data_valves))
    print(np.shape(data_flow))

    # tss = create_timeseries(np.concatenate((x,y), axis=1), np.concatenate((xheader,yheader)))

    # data_valves = remove_outliers(data_valves, 100)
    data_flow = remove_outliers(data_flow, 300)

    tss1 = create_timeseries(data_valves, header_valves)
    tss2 = create_timeseries(data_flow, header_flow)
    tss3 = create_timeseries(data_pump, header_pump)

    # print(json.dumps(acc, indent=2))

    # fig, _ = graph.plot_timeseries_multi(tss, "valve sequence", "samples [x0.1s]", "position [%]", False)

    gmean_flow, gstdev_flow = get_flow_metrics(data_flow)
    print(gmean_flow, gstdev_flow)
    gmean_pump, gstdev_pump = get_control_metrics(data_pump)
    print(gmean_pump, gstdev_pump)
    # print(data_flow)

    bm = split_data_bookmarks_2d(data_valves)
    print(bm)
    # quit()

    if stop_index_bm is None:
        stop_index_bm = len(bm) - 1

    stop_index = bm[stop_index_bm][1]
    # stop_index = bm[5][1]
    # stop_index = len(data_flow)

    start_index = bm[start_index_bm][1]

    if start_index_override is not None:
        start_index = start_index_override
    if stop_index_override is not None:
        stop_index = stop_index_override

    res = run_multiple_bookmarks(data_flow, bm, get_flow_metrics)

    flow_mean_bm = [e[0] for e in res]
    print(flow_mean_bm)
    flow_min_bm = np.min(flow_mean_bm)
    flow_max_bm = np.max(flow_mean_bm)
    gmean_flow_bm = np.mean(flow_mean_bm[start_index_bm: stop_index_bm])
    print(gmean_flow_bm)

    # quit()

    res1 = [[e[0], gmean_flow_bm] for e in res]
    print(res1)

    print(flow_min_bm)
    print(flow_max_bm)
    print(gmean_flow_bm)
    print(gmean_flow)

    print(abs(flow_max_bm - gmean_flow_bm)/gmean_flow_bm * 100)
    print(abs(flow_min_bm - gmean_flow_bm)/gmean_flow_bm * 100)

    print(abs(flow_max_bm - gmean_flow)/gmean_flow * 100)
    print(abs(flow_min_bm - gmean_flow)/gmean_flow * 100)

    gstdev_flow_bm = np.mean(
        [e[1] for e in res[start_index_bm: stop_index_bm]])

    header_aux = ["crt", "mean: " + "%0.1f" % gmean_flow_bm]

    tss4 = create_timeseries(res1, header_aux)
    res2 = [[e[1], gstdev_flow_bm] for e in res]

    header_aux = ["crt", "mean: " + "%0.1f" % gstdev_flow_bm]
    tss5 = create_timeseries(res2, header_aux)

    res = run_multiple_bookmarks(data_pump, bm, get_control_metrics)

    gstdev_pump_bm = np.mean(
        [e[1] for e in res[start_index_bm: stop_index_bm]])
    gmean_pump_bm = np.mean([e[0] for e in res[start_index_bm: stop_index_bm]])
    print(gstdev_pump_bm)

    res1 = [[e[0], gmean_pump_bm] for e in res]
    header_aux = ["crt", "mean: " + "%0.1f" % gmean_pump_bm]

    tss6 = create_timeseries(res1, header_aux)
    res3 = [[e[1], gstdev_pump_bm] for e in res]
    header_aux = ["crt", "mean: " + "%0.1f" % gstdev_pump_bm]

    tss7 = create_timeseries(res3, header_aux)

    print("shape:")
    print(np.shape(data_flow))
    print(np.shape(res1))

    # head = ["", "", "", ""]
    # axhead = ["yk (avg)", "yk (stdev)", "uk (avg)", "uk (stdev)"]
    # fig, _ = graph.plot_timeseries_multi_sub2([tss4, tss5, tss6, tss7], head, "samples [x0.1s]", axhead)

    # quit()

    # fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3], [
    #                                         "valve sequence", "sensor output", "pump output"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]"], (16,16))

    # fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3, tss4, tss5], [
    #                                           "valve sequence", "sensor output", "pump output", "mean", "stdev"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]", "mean [L/h]", "stdev"], (16,16))

    trim_ts(tss1, start_index, stop_index)
    trim_ts(tss2, start_index, stop_index)
    trim_ts(tss3, start_index, stop_index)
    trim_ts(tss4, start_index_bm, stop_index_bm)
    trim_ts(tss5, start_index_bm, stop_index_bm)
    trim_ts(tss6, start_index_bm, stop_index_bm)
    trim_ts(tss7, start_index_bm, stop_index_bm)

    head = ["control metrics (" + title + ")", "", "", "", "", "", ""]
    axhead = ["valve sequence", "flow [L/h]", "pump [%]"]
    fig, _ = graph.plot_timeseries_multi_sub2(
        [tss1, tss2, tss3], head, "sample [x0.1 s]", axhead, (16, 16))
    graph.save_figure(fig, "./figs/control_" + filename)

    if show_extra:
        head = [
            "control metrics (" + title + " - local averages)", "", "", "", "", "", ""]
        axhead = ["valve sequence", "sensor output", "pump output",
                  "flow [L/h]", "stdev [L/h]", "pump [%]", "stdev [%]"]
        fig, _ = graph.plot_timeseries_multi_sub2(
            [tss1, tss2, tss3, tss4, tss5, tss6, tss7], head, "sequence step", axhead, (16, 16))
    else:
        head = ["control metrics (" + title + " - local averages)", "", "", ""]
        axhead = ["flow [L/h]", "stdev [L/h]", "pump [%]", "stdev [%]"]
        fig, _ = graph.plot_timeseries_multi_sub2(
            [tss4, tss5, tss6, tss7], head, "sequence step", axhead, (16, 16))

    graph.save_figure(fig, "./figs/control_metrics_" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()
