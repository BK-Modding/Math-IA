import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

data = pd.read_csv('brominewater-glycerol.csv')

time = data['Run one: Time (s)']

temp1 = data['Run one: Temperature (°C)']

x = np.array(time, dtype=float)
y = np.array(temp1, dtype=float)

popt = [11.775, 0.00201981, 39.099]

print(r2_score(y, func(x, *popt)))

plt.figure()
plt.plot(x, y, 'ko', label="Original glycerol data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted curve using regression by hand")
plt.xlabel('Time elapsed (s)')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.show()
