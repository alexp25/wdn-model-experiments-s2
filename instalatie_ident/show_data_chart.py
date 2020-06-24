import numpy as np
import pandas as pd

# import our modules
from modules import loader, graph
from modules.graph import Timeseries
from modules import clustering
import time
import os
import yaml
from typing import List
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

root_data_folder = "./data/cstatic"
# read the data from the csv file

filenames = ["exp_246"]

root_data_folder = "./data/cdynamic"
# read the data from the csv file

filenames = ["exp_252"]


def run_clustering(x, times, xheader):
    # for each node, split the data into days (2400 samples) => 2D

    # for each node get 2 clusters
    # then combine all clusters to get 2 final clusters
    # print([t.timestamp() for t in times[:10]])
    # quit()

    # print(xheader)

    # seconds, time based
    # t_day = 240
    t_day = 480
    # n_day = 2400
    # n_day = 4800
    n_day = 9600
    # n_skip = 200
    # n_skip = 400
    n_skip = 1

    sx = np.shape(x)
    sd = int(math.ceil(sx[0]/n_day))
    print(sd)
    n3 = int(n_day / n_skip)
    xd = np.zeros((sx[1], sd+1, n3))

    xheader1 = ["day " + str(d) for d in range(sd+1)]
    print(np.shape(xd))
    print(sx)

    xlabels = []

    day = 0

    for k in range(sx[1]):
        # for each sample
        t0 = times[0].timestamp()
        sample = 0
        day = 0
        xk = x[:, k]
        xk = normalize(xk, False)

        for j in range(sx[0]):
            try:
                xd[k][day][sample] = xk[j]
            except:
                print("exception at [" + str(k) + "," +
                      str(day) + "," + str(sample) + "]")

            sample += 1

            # check days
            t1 = times[j].timestamp()
            if t1 - t0 >= t_day:
                t0 = t1
                sample = 0
                day += 1

    print(np.shape(xd[0]))
    print(xd[0])

    for row in xd[0][1:]:
        print(row)

    # quit()
    ncons = sx[1]
    # ncons = 1

    cons = range(ncons)
    # cons = [2, 3, 4, 7, 8, 9]
    cons = [4, 5, 7]

    trim = False
    trim = True

    # for each node

    plot_each = False
    # plot_each = True

    xc_vect = []

    for k in cons:

        print(k)

        # plot daily demands

        if trim:
            xt = np.transpose(xd[k][1:-1])
        else:
            xt = np.transpose(xd[k])

        title = "consumer #" + str(k+1) + " patterns"

        if plot_each:
            tss = create_timeseries(xt, xheader1)
            fig, _ = graph.plot_timeseries_multi_sub2(
                [tss], [title], "samples [x0.1s]", ["flow [L/h]"], None)

        xt = np.transpose(xt)
        nc = 2
        X, kmeans, _, _ = clustering.clustering_kmeans(xt, nc)
        xc = np.transpose(kmeans.cluster_centers_)
        # print(xc)
        xheader2 = [str(e+1) for e in range(nc)]
        # print(xheader2)

        # x = x[:10000]
        # xc = remove_outliers(xc)

        xc_vect.append(xc)

        # x = [list(x[:,0])]
        # x = np.transpose([x[:,0]])
        # xheader = [xheader[0]]
        # # print(x)
        # print(xheader)

        if plot_each:
            tss = create_timeseries(xc, xheader2)
            fig, _ = graph.plot_timeseries_multi_sub2(
                [tss], [title], "samples [x0.1s]", ["flow [L/h]"], None)

    xc_vect = np.array(xc_vect)

    xcs = np.shape(xc_vect)
    xc_vect = xc_vect.reshape(xcs[0]*xcs[2], xcs[1])
    print(xc_vect)
    print(np.shape(xc_vect))

    # xc_vect = np.transpose(xc_vect)
    # quit()
    # nc = 12
    nc = 2
    X, kmeans, _, _ = clustering.clustering_kmeans(xc_vect, nc)
    xc = np.transpose(kmeans.cluster_centers_)
    print(xc)
    xheader2 = [str(e+1) for e in range(nc)]

    hours = np.linspace(0, t_day/2, (np.shape(xc))[0])
    xlabels=[[e for e in hours] for i in range(nc)]
    xlabels=np.array(xlabels)
    xlabels=np.transpose(xlabels)

    print(xlabels)

    # quit()

    title="consumer patterns"

    tss=create_timeseries(xc, xheader2, xlabels)
    fig, _=graph.plot_timeseries_multi_sub2(
        [tss], [title], "day [240s => 24h]", ["flow [0-1]"], None)

    graph.save_figure(fig, "./figs/consumer_patterns_all_2")


def plot_data(x, y, xheader, yheader):
    tss=create_timeseries(y, yheader)

    tss2=create_timeseries(x, xheader)

    # print(json.dumps(acc, indent=2))

    # fig, _ = graph.plot_timeseries_multi(tss, "valve sequence", "samples [x0.1s]", "position [%]", False)

    fig, _=graph.plot_timeseries_multi_sub2([tss, tss2], [
                                              "valve sequence", "sensor output"], "samples [x0.1s]", ["position [%]", "flow [L/h]"], (9, 16))

    graph.save_figure(fig, "./figs/valve_sequence_" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()


# create separate models for each data file
for filename in filenames:
    data_file=root_data_folder + "/" + filename + ".csv"
    x, y, xheader, yheader, times=loader.load_dataset(data_file)

    # tss = create_timeseries(x, xheader)

    # TODO: sort by chan number 0 - 10
    # TODO: show as subplot

    print(xheader)
    print(yheader)

    print(len(xheader))

    order=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2]

    xheader=reorder(xheader, order)

    x=reorder2d(x, order)

    print(x)

    # xheader[2], xheader[10] = xheader[10], xheader[2]
    # x[:, 2], x[:, 10] = x[:, 10], x[:, 2].copy()

    print("sorted")
    print(xheader)
    print(yheader)

    start_index=3
    end_index=None
    # end_index = 9600
    # end_index = 4800 * 5

    x=x[start_index:, :]
    y=y[start_index:, :]

    if end_index is not None:
        x=x[:end_index, :]
        y=y[:end_index, :]

    print(x)

    sx=np.shape(x)
    sy=np.shape(y)

    print(np.shape(x))
    print(np.shape(y))

    print(x)

    print(y)

    x=remove_outliers(x)

    # run_clustering(x, times, xheader)
    # run_clustering(y, times, yheader)
    plot_data(x, y, xheader, yheader)
