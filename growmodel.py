import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import scipy.optimize
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort

# Now, (06/10/2021) I will modify the code to run growth curve modelling
# Pull in data from the database
conn = sqlite3.connect('database3.db')
c = conn.cursor()
c.execute('SELECT * FROM posts WHERE id = 3')
rex = c.fetchone()

Pre_years = rex[3]
Pre_datah = rex[4]
Pre_parameters = rex[5]

first_years = list(map(int,Pre_years.split(",")))
first_datah = list(map(float,Pre_datah.split(",")))
first_parameters = list(map(float,Pre_parameters.split(", ")))
# print(first_parameters)

# Separate the columns of data into their own arrays. These are panda arrays and will need to be changed to numpy arrays
# to do this, just call the variables as 'years.values' and 'datah.values' when used in a function
years = pd.DataFrame(data=first_years)
datah = pd.DataFrame(data=first_datah)
parameters = pd.DataFrame(data=first_parameters)


# Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array
def ackley(x):
    arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
    arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
    arg3 = np.add(arg1, arg2)                      # add curves together to get total model curve
    arg4 = np.subtract(datah.values,arg3)          # find difference of this array and the actual data
    arg5 = np.square(arg4)                         # Square the difference
    return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)
    

# Method of optimization - minimize sum of squares by changing values in 'x' array - works great!
x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user
  
res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
res.x
print(res.x)                                                   # print the 'x' array (growth curve parameters)

# Define variables used in logicstic model - res.x is the array of parameters
L1 = res.x[0]
H1 = res.x[1]
W1 = res.x[2]
M1 = res.x[3]

L2 = res.x[4]
H2 = res.x[5]
W2 = res.x[6]
M2 = res.x[7]

# Use the Growth Curve Model to create curve1, curve2, sumcurves arrays
curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
sumcurves=curve1+curve2
finalparametri = [L1, H1, W1, M1, L2, H2, W2, M2]

sqle = "UPDATE posts SET finalparameters = %s WHERE address = %s"
vale = (finalparametri, "finalparameters")
c.execute(sqle, vale)

# Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
allyears = years[0]
alldatah = datah[0]

plt.xkcd()
plt.title("Growth Curve Modelling") 
plt.xlabel("Years") 
plt.ylabel("Efficiency") 
plt.plot(allyears.values,curve1)
plt.plot(allyears.values,curve2) 
plt.plot(allyears.values,sumcurves)
plt.plot(allyears.values,alldatah.values) 
plt.legend(['Curve1', 'Curve2', 'SumCurves', 'Data'])  
plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project2\\myplot.png', format='png')
plt.show()

# Output the optimized parameter values to the database and then show them to the user

conn.commit()

conn.close()