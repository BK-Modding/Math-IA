import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

useMeasuredTemp = True

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func2(x, a, b):
    return a * np.exp(-b * x) + 27.7

data = pd.read_csv('brominewater-glycerol.csv')
data2 = pd.read_csv('glycerol-ethanol.csv')

option = 'pair1'
regression = True

if option == 'pair1':
    time = data['Run one: Time (s)']

    temp1 = data['Run one: Temperature (°C)']
    temp2 = data['Run one: Temperature 2 (°C)']
    plt.plot(time, temp1, label='glycerol')
    plt.plot(time, temp2, label='bromine water')
    plt.xlabel('Time elapsed (s)')
    plt.ylabel('Temperature (°C)')
    if regression:
        x = np.array(time, dtype=float)
        y1 = np.array(temp1, dtype=float)
        y2 = np.array(temp2, dtype=float)

        if useMeasuredTemp:
            popt1, pcov1 = curve_fit(func2, x, y1)
            print("Glycerol: params = {}, R^2 = {}".format(popt1, r2_score(y1, func2(x, *popt1))))

            popt2, pcov2 = curve_fit(func2, x, y2)
            print("Bromine water: params = {}, R^2 = {}".format(popt2, r2_score(y2, func2(x, *popt2))))

            plt.plot(x, func2(x, *popt1), 'r-', label="glycerol fitted curve")
            plt.plot(x, func2(x, *popt2), 'g-', label="bromine water fitted curve")
        else:
            popt1, pcov1 = curve_fit(func, x, y1)
            print("Glycerol: params = {}, R^2 = {}".format(popt1, r2_score(y1, func(x, *popt1))))

            popt2, pcov2 = curve_fit(func, x, y2)
            print("Bromine water: params = {}, R^2 = {}".format(popt2, r2_score(y2, func(x, *popt2))))

            plt.plot(x, func(x, *popt1), 'r-', label="glycerol fitted curve")
            plt.plot(x, func(x, *popt2), 'g-', label="bromine water fitted curve")


    plt.legend(loc='upper right')
    plt.show()
elif option == 'pair2':
    time = data2['Remote Data: Time (s)']

    temp1 = data2['Remote Data: Temperature (°C)']
    temp2 = data2['Remote Data: Temperature 2 (°C)']
    plt.plot(time, temp1, label='ethanol')
    plt.plot(time, temp2, label='glycerol')
    plt.xlabel('Time elapsed (s)')
    plt.ylabel('Temperature (°C)')
    if regression:
        x = np.array(time, dtype=float)
        y1 = np.array(temp1, dtype=float)
        y2 = np.array(temp2, dtype=float)

        if useMeasuredTemp:
            popt1, pcov1 = curve_fit(func2, x, y1)
            print("Ethanol: params = {}, R^2 = {}".format(popt1, r2_score(y1, func2(x, *popt1))))

            popt2, pcov2 = curve_fit(func2, x, y2)
            print("Glycerol: params = {}, R^2 = {}".format(popt2, r2_score(y2, func2(x, *popt2))))

            plt.plot(x, func2(x, *popt1), 'r-', label="ethanol fitted curve")
            plt.plot(x, func2(x, *popt2), 'g-', label="glycerol fitted curve")
        else:
            popt1, pcov1 = curve_fit(func, x, y1)
            print("Ethanol: params = {}, R^2 = {}".format(popt1, r2_score(y1, func(x, *popt1))))

            popt2, pcov2 = curve_fit(func, x, y2)
            print("Glycerol: params = {}, R^2 = {}".format(popt2, r2_score(y2, func(x, *popt2))))

            plt.plot(x, func(x, *popt1), 'r-', label="ethanol fitted curve")
            plt.plot(x, func(x, *popt2), 'g-', label="glycerol fitted curve")

    plt.legend(loc='upper right')
    plt.show()


