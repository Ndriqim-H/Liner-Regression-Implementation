import pandas as pd
import numpy as np

data = pd.read_csv('ex1data1.csv')
print(data.head())
#Plot the data

import matplotlib.pyplot as plt
# plt.scatter(data['X'], data['Y'])
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# plt.show()


import Linear_Regression as lr
import matplotlib.markers as markers
model = lr.LinearRegression()
model.fit(data['X'], data['Y'])
#Show the line on the plot
plt.scatter(data['X'], data['Y'], marker=markers.MarkerStyle(marker='x', fillstyle=None), color='red')
plt.plot(data['X'], model.line(data['X']))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()