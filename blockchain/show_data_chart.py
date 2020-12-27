import numpy as np
import pandas as pd

# import our modules
from modules import loader, graph
from modules.graph import Timeseries
import time
import os
import yaml
from typing import List
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import operator

root_data_folder = "./"
# read the data from the csv file

# filenames = ["d1", "d2"]
# titles = ["Add new block time", "Get row by id time"]
# scales = [None, (None, (2000,2200))]

filenames = ["d3", "d4", "d3_1", "d4_1"]
titles = ["Write average latency", "Batch write average latency", "Read average latency", "Batch read average latency"]
scales = [None, None, None, None]

# filenames = ["d2"]
# titles = ["Get row by id time"]

add_mean = False

def fit_model(x, y, degree):
    # create the polynomal features to be used for training the regression model
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    # create a linear regression model using scikit-learn
    model = LinearRegression()
    # fit the model to the data points (x,y)
    model.fit(x_poly, y)
    # use the model to predict the data points
    y_poly_pred = model.predict(x_poly)

    print(model.coef_)
    # get the prediction error as RMSE
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    # get the regression score
    r2 = r2_score(y, y_poly_pred)
    return x_poly, y_poly_pred, rmse, r2


def plot_line(x, y_poly_pred):
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred)

def create_timeseries(data, header, datax=None):
    tss: List[Timeseries] = []
    colors = ['blue', 'red', 'green', 'orange']
    # colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo']

    ck = 0

    sdata = np.shape(data)
    rows = sdata[0]
    cols = sdata[1]

    print(sdata)

    # colors = cm.rainbow(np.linspace(0, 1, cols))
    # colors = cm.viridis(np.linspace(0, 1, cols))

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



# create separate models for each data file
for i, filename in enumerate(filenames):
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, xheader, yheader = loader.load_dataset(data_file)

    print(xheader)
    print(yheader)

    start_index = 0
    end_index = None

    # end_index = 9600
    # end_index = 4800 * 5

    x = x[start_index:, :]
    y = y[start_index:, :]

    if end_index is not None:
        x = x[:end_index, :]
        y = y[:end_index, :]

    sx = np.shape(x)
    sy = np.shape(y)

    print(np.shape(x))
    print(np.shape(y))

    # plot_data(x, y, xheader, yheader)

    x_poly, y_poly_pred, rmse, r2 = fit_model(x, y, 0)

    tss_xy = create_timeseries(y, ["data"], x)
    plot_tss = [tss_xy[0]]
    if add_mean:
        tss_xy_poly = create_timeseries(y_poly_pred, ["mean: " + str(int(y_poly_pred[0]))], x)
        tss_xy_poly[0].color = "orange"
        plot_tss.append(tss_xy_poly[0])
    
    fig, _ = graph.plot_timeseries_multi(i, plot_tss, titles[i], xheader[0], yheader[0], scales[i], (10, 8))
    graph.save_figure(fig, titles[i])
