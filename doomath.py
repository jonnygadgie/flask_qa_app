# This is an attempt at more-or-less full automation of the growth model 
# initial guesses will be estimated based off of raw data and the code run
# I'm interested in how well the model will optimize parameters based on this approach

import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt 
from matplotlib import pyplot as plt
import scipy.optimize
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

# Now, (06/10/2021) I will modify the code to run growth curve modelling
conn = sqlite3.connect('database3.db')
c = conn.cursor()

rex = conn.execute('SELECT * FROM posts WHERE id = ?',
                        (3,)).fetchone()                # post_id is 3: the 3rd (or 4th?) input from users

Pre_years = rex[3]
# print('Pre_years is :' , Pre_years)
Pre_datah = rex[4]
Pre_parameters = rex[5]
first_years = list(map(float,Pre_years.split(",")))
       # getting 'RuntimeWarning: overflow encountered in exp' for years array
       # apparently this is due to exp(too_big_a_number) so I might end up using float128 to get more space (comes with other potential issues)
first_datah = list(map(float,Pre_datah.split(",")))
first_parameters = list(map(float,Pre_parameters.split(",")))

# The last few lines were just getting the database values in a format I can use
# Separate the columns of data into their own arrays. These are panda arrays and will need to be changed to numpy arrays
# to do this, pull the second column from each dataframe and call the variables with '.values' when used in a function


# print('years is: ', years)
datah = pd.DataFrame(data=first_datah)
parameters = pd.DataFrame(data=first_parameters)
years = pd.DataFrame(data=first_years)

# Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array    ***2-curves***
def ackley(x):
    arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
    arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
    arg3 = np.add(arg1, arg2)                      # add curves together to get total model curve
    arg4 = np.subtract(datah.values,arg3)          # find difference of this array and the actual data
    arg5 = np.square(arg4)                         # Square the difference
    return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)
    
#x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user
#firstparameterslength = len(first_parameters)   # only 
#if firstparameterslength == 12:                                              # If user inputs 3 sets of parameters (3-curve fitting)
#    x0 = np.delete(first_parameters, [8,9,10,11]) 
#if firstparameterslength == 16:                                              # If user inputs 4 sets of parameters (4-curve fitting)
#    x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15]) 
#if firstparameterslength == 20:                                              # If user inputs 5 sets of parameters (5-curve fitting)
#    x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15,16,17,18,19])    # we only need 2 sets: index number 0-7 of the array
x0 = np.delete(first_parameters, 0)

datah_array_length = len(first_datah)
datah_last_element = first_datah[datah_array_length - 1]
datah_first_element = first_datah[0]
years_array_length = len(first_years)
years_last_element = first_years[years_array_length - 1]
years_first_element = first_years[0]
print('The datah_last_element is: ', datah_last_element)
print('The years_last_element is: ', years_last_element)

L1g = 0
L2g = 0
datah_range = (datah_last_element)-(datah_first_element)
years_range = (years_last_element)-(years_first_element)       # The range of years 

H1g = (datah_range*0.33)+(datah_first_element)      # The first H parameter guess is 1/3 of the way between first and last datah values
H2g = (datah_range*0.67)+(datah_first_element)      # The second H parameter guess is 2/3 of the way between first and last datah values
W1g = 1.0
W2g = 1.0
M1g = (years_range/3)+(years_first_element)         # The first M parameter guess is 1/3 of the way between first and last years values
M2g = (years_range*.67)+(years_first_element)       # The second M parameter guess is 2/3 of the way between first and last years values

                               # Now put all these revised guesses into the x0 array
                               # in the final version, the user won't even enter parameter guesses, so x0 won't need to be updated
                               # but it will need to be created in the first place
                               # This is a 2-curve attempt at modelling
np.put(x0, [0, 1, 2, 3, 4, 5, 6, 7], [L1g, H1g, W1g, M1g, L2g, H2g, W2g, M2g])

print('the new parameters are: ', x0)
res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)                   # setup the algorithm
res.x                                                                        # Return the 'x' array of optimized parameters

# Define variables used in logistic model - res.x is the array of parameters (This example has 2 curves)
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

array_length = len(first_years)
last_year = first_years[array_length - 1]
futurecurveyears = np.array([last_year + 1, last_year + 2, last_year + 3, last_year + 4, last_year + 5, last_year + 6, last_year + 7, last_year + 8, last_year + 9, last_year + 10])
futureyears = pd.DataFrame(data=futurecurveyears)
futurecurvedata = (H1+((L1-H1)*np.reciprocal(1+np.exp((futureyears.values-M1)/W1))))+(H2+((L2-H2)*np.reciprocal(1+np.exp((futureyears.values-M2)/W2))))
# print("Future curve data is: ", futurecurvedata) # Data is in column form, just like curve1 and curve2

sumcurves=curve1+curve2
print("Sum of curves data is: ", sumcurves)
sumcurves2b = np.transpose(sumcurves)
sumcurves2 = np.array(sumcurves2b[0])
theyears = np.array(first_years)

# print('sumcurves2 is: ', sumcurves2, 'theyears is: ' , theyears)
p = np.poly1d(np.polyfit(theyears, sumcurves2, deg=6))
print(' The fit is:', p)

ty = theyears   # abbreviate for the next set of calculations
# determine the values of p for every year
trendline = p[6]*(ty**6) + p[5]*(ty**5) + p[4]*(ty**4) + p[3]*(ty**3) + p[2]*(ty**2) + p[1]*ty + p[0] 
# print ('trendline values are: ', trendline) # numpy array
# Now take the derivative of p
d1 = p[1]*1
d2 = p[2]*2
d3 = p[3]*3
d4 = p[4]*4
d5 = p[5]*5
d6 = p[6]*6

derivative = d6*(ty**5) + d5*(ty**4) + d4*(ty**3) + d3*(ty**2) + d2*(ty) + d1
print('The derivate is: ', derivative)

Li1 = str(L1)
Hi1 = str(H1)
Wi1 = str(W1)
Mi1 = str(M1)
Li2 = str(L2)
Hi2 = str(H2)
Wi2 = str(W2)
Mi2 = str(M2)
# Get the R-squared for the curve fitting. First, we need the f(x) definition (model fit curve). This is the sum of all curves or simply 'sumcurves'.
# Now we need to sum the residuals. First calculate the residuals array
first_sumcurves = list(map(float,sumcurves))                # This is our fitted curve (sumcurves) as a manipulable array o' data     
x_values = first_datah                                      # The next 6 lines of code I got from online
y_values = first_sumcurves                                  # Process for calculating R-squared (Matches excel)
correlation_matrix = np.corrcoef(x_values, y_values)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2.
print("The R-squared is :", r_squared)


finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2])
conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, 3])
# Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
# Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
alldatah = datah[0]
print('allyears.values = ', allyears.values)

plt.subplot(1, 2, 1)                               
plt.title("Growth Curve Modelling") 
plt.xlabel("Years") 
plt.ylabel("Efficiency") 
plt.plot(allyears.values,curve1,',')
plt.plot(allyears.values,curve2,',') 
plt.plot(allyears.values,sumcurves,'*')
plt.plot(allyears.values,alldatah.values, '.') 
plt.plot(allyears.values,trendline, '*')
# plt.plot(futurecurveyears, futurecurvedata)

plt.legend(['Curve1', 'Curve2', 'SumCurves', 'Data', 'Trendline', 'Futurecurve'])  
plt.subplot(1, 2, 2)
plt.title("First Derivative") 
plt.plot(ty,derivative)
# plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
# save the .png of the graph - this is future functionality. Currently, the next line of code will show the graph to the user
# Once the user closes this graph, the optimized parameters will be showed
plt.show()

conn.commit()
conn.close()   