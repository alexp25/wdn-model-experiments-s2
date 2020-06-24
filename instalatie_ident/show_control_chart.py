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

from modules.ts_utils import *
from modules.ts_control_utils import *


root_data_folder = "./data/cstatic"
# read the data from the csv file

filenames = ["exp_246"]

root_data_folder = "./data/cdynamic"
# read the data from the csv file

filenames = ["exp_252"]

bookmarks = [188, 282, 375, 469, 563, 657]
bookmarks = []


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

    # data1, header1 = order_data(data, header, range(14, 19))

    # valves: 14-19
    data1, header1 = order_data(data, header, range(14, 19+1))
    # data1, header1 = order_data(data, header, [14])
    # flow: 2-12
    # data2, header2 = order_data(data, header, range(2, 12))
    # [flow, ref=20]
    data2, header2 = order_data(data, header, [10, 20])
    # pump: 13
    data3, header3 = order_data(data, header, [13])

    print(np.shape(data1))
    print(np.shape(data2))

    # tss = create_timeseries(np.concatenate((x,y), axis=1), np.concatenate((xheader,yheader)))

    data1 = remove_outliers(data1, 100)
    data2 = remove_outliers(data2, 300)

    tss1 = create_timeseries(data1, header1)
    tss2 = create_timeseries(data2, header2)
    tss3 = create_timeseries(data3, header3)

    # print(json.dumps(acc, indent=2))

    # fig, _ = graph.plot_timeseries_multi(tss, "valve sequence", "samples [x0.1s]", "position [%]", False)

    gmean_flow, gstdev_flow = get_flow_metrics(data2)
    print(gmean_flow, gstdev_flow)
    gmean_pump, gstdev_pump = get_control_metrics(data3)
    print(gmean_pump, gstdev_pump)
    # print(data2)
    bm = split_data_bookmarks(data1[:,0])
    print(bm)
    # quit()

    header_aux = ["crt", "mean"]
    res = run_multiple_bookmarks(data2, bm, get_flow_metrics)
    res1 = [[e[0], gmean_flow] for e in res]
    flow_mean_bm = [e[0] for e in res]
    flow_min_bm = np.min(flow_mean_bm)
    flow_max_bm = np.max(flow_mean_bm)
    gmean_flow_bm = np.mean(flow_mean_bm)

    print(flow_min_bm)
    print(flow_max_bm)
    print(gmean_flow_bm)
    print(gmean_flow)

    print(abs(flow_max_bm - gmean_flow_bm)/gmean_flow_bm * 100)
    print(abs(flow_min_bm - gmean_flow_bm)/gmean_flow_bm * 100)

    print(abs(flow_max_bm - gmean_flow)/gmean_flow * 100)
    print(abs(flow_min_bm - gmean_flow)/gmean_flow * 100)


    gstdev_pump_bm = np.mean(gstdev_pump)
    print(gstdev_pump_bm)

    # quit()


    tss4 = create_timeseries(res1, header_aux)
    res2 = [[e[1], gstdev_flow] for e in res]
    tss5 = create_timeseries(res2, header_aux)

    res = run_multiple_bookmarks(data3, bm, get_control_metrics)
    res1 = [[e[0], gmean_pump] for e in res]
    tss6 = create_timeseries(res1, header_aux)
    res2 = [[e[1], gstdev_pump] for e in res]
    tss7 = create_timeseries(res2, header_aux)

    print("shape:")
    print(np.shape(data2))
    print(np.shape(res1))


    # head = ["", "", "", ""]
    # axhead = ["yk (avg)", "yk (stdev)", "uk (avg)", "uk (stdev)"]
    # fig, _ = graph.plot_timeseries_multi_sub2([tss4, tss5, tss6, tss7], head, "samples [x0.1s]", axhead)


    # quit()
    fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3], [
                                              "valve sequence", "sensor output", "pump output"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]"], (16,16))

    # fig, _ = graph.plot_timeseries_multi_sub2([tss1, tss2, tss3, tss4, tss5], [
    #                                           "valve sequence", "sensor output", "pump output", "mean", "stdev"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]", "mean [L/h]", "stdev"])


    graph.save_figure(fig, "./figs/control" + filename)

    # x = remove_outliers(x)
    # tss = create_timeseries(x, xheader)

    # fig, _ = graph.plot_timeseries_multi(tss, "sensor output", "samples [x0.1s]", "flow [L/h]", False)

    # graph.save_figure(fig, "./figs/sensor_output")
    # graph.plot_timeseries(ts, "title", "x", "y")

    # quit()
