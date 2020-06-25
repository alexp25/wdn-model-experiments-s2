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

use_pump = 1
use_valves = 2

root_data_folder = "./data/cstatic"
filenames = ["exp_246"]
title = "static"
skip2 = True
mode = use_pump

# root_data_folder = "./data/cdynamic"
# filenames = ["exp_252"]
# title = "dynamic"
# skip2 = True
# mode = use_pump

# root_data_folder = "./data/prbs_1"
# filenames = ["exp_258"]
# title = "prbs_1"
# skip2 = False
# mode = use_valves

root_data_folder = "./data/prbs_2"
filenames = ["exp_259"]
title = "prbs_2"
skip2 = False
mode = use_valves
append_last_bookmark = True

full_chart = True
full_chart = False



# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    data, header = loader.load_dataset_full_with_header(data_file)

    print(header)
    print(len(header))
    for i, h in enumerate(header):
        print(i, h)


    # valves: 14-19
    data_valves, header_valves = order_data(data, header, range(14, 19+1))
    # data_valves, header_valves = order_data(data, header, [14])
    # flow: 2-12
    # data_flow, header_flow = order_data(data, header, range(2, 12))
    # [flow, ref=20]
    data_flow, header_flow = order_data(data, header, [10])
    # pump: 13
    data_pump, header_pump = order_data(data, header, [13])

    if mode == use_pump:
        bm_pump = extract_bookmarks(data_pump, 1)
    elif mode == use_valves:
        bm_pump = extract_bookmarks_multitrack(data_valves, append_last_bookmark)

    print(bm_pump)
    bookmarks = bm_pump

    # quit()

    if full_chart:
        bookmarks = [0, len(data_pump)-1]
    # quit()

    print(np.shape(data_valves))
    print(np.shape(data_flow))

    gmean_flow, gstdev_flow = get_flow_metrics_noref(data_flow)
    print(gmean_flow, gstdev_flow)
    gmean_pump, gstdev_pump = get_control_metrics(data_pump)
    print(gmean_pump, gstdev_pump)
    # print(data_flow)
    bm = split_data_bookmarks(data_valves[:,0])
    print(bm)


    header_aux = ["crt", "mean"]
    res = run_multiple_bookmarks(data_flow, bm, get_flow_metrics_noref)
    res1 = [[e, gmean_flow] for e in res]
    flow_mean_bm = [e for e in res]

    print(bookmarks)

    for i, b in enumerate(bookmarks):
        if i > 0:
            data_valves_b = data_valves[bookmarks[i-1]:b,:]
            data_flow_b = data_flow[bookmarks[i-1]:b,:]
            data_pump_b = data_pump[bookmarks[i-1]:b,:]

            tss_valves = create_timeseries(data_valves_b, header_valves)
            tss_flow = create_timeseries(data_flow_b, header_flow)
            tss_pump = create_timeseries(data_pump_b, header_pump)

            if len(bookmarks) == 2:
                fig, _ = graph.plot_timeseries_multi_sub2([tss_valves, tss_flow, tss_pump], [
                                                        "valve sequence", "sensor output", "pump output"], "samples [x0.1s]", ["model", "flow [L/h]", "pump [%]"], (16,16), False, i)
            else:
                fig, _ = graph.plot_timeseries_multi_sub2([tss_flow, tss_pump], ["sensor output", "pump output"], "samples [x0.1s]", ["flow [L/h]", "pump [%]"], (16,16), False, i)


            if len(bookmarks) == 2:
                graph.save_figure(fig, "./figs/ident_" + title + "_" + filename + "_full")
                loader.save_csv(data_flow_b, "./data/output/ident_flow_" + title + "_" + filename + "_full.csv")
                loader.save_csv(data_pump_b, "./data/output/ident_pump_" + title + "_" + filename + "_full.csv")
            else:
                if i%2==1 or not skip2:
                    graph.save_figure(fig, "./figs/ident_" + title + "_" + filename + "_" + str(i))
                    loader.save_csv(data_flow_b, "./data/output/ident_flow_" + title + "_" + filename + "_" + str(i) + ".csv")
                    loader.save_csv(data_pump_b, "./data/output/ident_pump_" + title + "_" + filename + "_" + str(i) + ".csv")
