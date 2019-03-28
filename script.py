import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import sympy as sym

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

data = pd.read_csv('brominewater-glycerol.csv')
time = data['Run one: Time (s)']

temp1 = data['Run one: Temperature (°C)']
temp2 = data['Run one: Temperature 2 (°C)']

plt.plot(time, temp1)
plt.plot(time, temp2)

x = np.array(time, dtype=float)
y = np.array(temp1, dtype=float)

popt, pcov = curve_fit(func, x, y)
print(popt)
print(r2_score(y, func(x, *popt)))

plt.figure()
plt.plot(x, y, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()
