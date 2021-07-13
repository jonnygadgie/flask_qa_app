# As of 6/26/21, this is my latest attempt at a data processing section for app.py
# The plan is to perform a 2-curve fit to the incoming data and determine the R-squared of the fit
# If this R-squared is below a threshold value, a 3-curve fit will be attempted and so on (up to a 5-curve fit)
# This way, the user won't have to input a guess at the number of curves. 
# It also paves the way for a program that is entirely automatic (the math section takes initial guesses at fitting parameters -- no user input besides raw data)

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
                        (post_id,)).fetchone()

Pre_years = rex[3]
Pre_datah = rex[4]
Pre_parameters = rex[5]
first_years = list(map(int,Pre_years.split(",")))
first_datah = list(map(float,Pre_datah.split(",")))
first_parameters = list(map(float,Pre_parameters.split(",")))
# The last few lines were just getting the database values in a format I can use
# Separate the columns of data into their own arrays. These are panda arrays and will need to be changed to numpy arrays
# to do this, pull the second column from each dataframe and call the variables with '.values' when used in a function
years = pd.DataFrame(data=first_years)
datah = pd.DataFrame(data=first_datah)
parameters = pd.DataFrame(data=first_parameters)

# Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array    ***2-curves***
def ackley(x):
    arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
    arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
    arg3 = np.add(arg1, arg2)                      # add curves together to get total model curve
    arg4 = np.subtract(datah.values,arg3)          # find difference of this array and the actual data
    arg5 = np.square(arg4)                         # Square the difference
    return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)
    
#x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user
firstparameterslength = len(first_parameters)   # only 
if firstparameterslength == 12:                                              # If user inputs 3 sets of parameters (3-curve fitting)
    x0 = np.delete(first_parameters, [8,9,10,11]) 
if firstparameterslength == 16:                                              # If user inputs 4 sets of parameters (4-curve fitting)
    x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15]) 
if firstparameterslength == 20:                                              # If user inputs 5 sets of parameters (5-curve fitting)
    x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15,16,17,18,19])    # we only need 2 sets: index number 0-7 of the array
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
sumcurves=curve1+curve2

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

if (r_squared > 0.93):
    finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2])
    conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])
    # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
    # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
    allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
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
    # plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
    # save the .png of the graph - this is future functionality. Currently, the next line of code will show the graph to the user
    # Once the user closes this graph, the optimized parameters will be showed
    plt.show()
    
else: 
    # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array     *** 3-curve ***
        def ackley(x):
            arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
            arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
            arg3 = x[9]+((x[8]-x[9])*np.reciprocal(1+np.exp(((years.values)-x[11])/x[10])))  # Curve 3 growth curve model
            arg4b = np.add(arg1, arg2)
            arg4 = np.add(arg4b,arg3)                      # add curves together to get total model curve
            arg5 = np.subtract(datah.values,arg4)          # find difference of this array and the actual data
            arg6 = np.square(arg5)                         # Square the difference
            return np.sum(arg6)                            # add up all elements in the squared difference (sum of squares)

        x0 = np.delete(first_parameters, 0)  # initial guesses at values. The first value in 'first_parameters' is the number of curves from the user 
        if firstparameterslength == 12:                                              # If user inputs 3 sets of parameters (3-curve fitting)
            x0 = first_parameters
        if firstparameterslength == 16:                                              # If user inputs 4 sets of parameters (4-curve fitting)
            x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15]) 
        if firstparameterslength == 20:                                              # If user inputs 5 sets of parameters (5-curve fitting)
            x0 = np.delete(first_parameters, [8,9,10,11,12,13,14,15,16,17,18,19])    # we only need 2 sets: index number 0-7 of the array
        res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
        res.x

        # Define variables used in logistic model - res.x is the array of parameters (This example has 3 curves)
        L1 = res.x[0]
        H1 = res.x[1]
        W1 = res.x[2]
        M1 = res.x[3]
        L2 = res.x[4]
        H2 = res.x[5]
        W2 = res.x[6]
        M2 = res.x[7]
        L3 = res.x[8]
        H3 = res.x[9]
        W3 = res.x[10]
        M3 = res.x[11]

        # Use the Growth Curve Model to create curve1, curve2, curve 3, sumcurves arrays
        curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
        curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
        curve3 = H3+((L3-H3)*np.reciprocal(1+np.exp((years.values-M3)/W3)))
        sumcurvesb=curve1+curve2
        sumcurves = sumcurvesb+curve3
        Li1 = str(L1)
        Hi1 = str(H1)
        Wi1 = str(W1)
        Mi1 = str(M1)
        Li2 = str(L2)
        Hi2 = str(H2)
        Wi2 = str(W2)
        Mi2 = str(M2)
        Li3 = str(L3)
        Hi3 = str(H3)
        Wi3 = str(W3)
        Mi3 = str(M3)

        first_sumcurves = list(map(float,sumcurves))                # This is our fitted curve (sumcurves) as a manipulable array o' data     
        x_values = first_datah                                      # The next 6 lines of code I got from online
        y_values = first_sumcurves                                  # Process for calculating R-squared (Matches excel)
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2.
        print("The R-squared is :", r_squared)      
        
        if(r_squared > 0.93):
            finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2, Li3, Hi3, Wi3, Mi3])
            conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])

            # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
            # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
            allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
            alldatah = datah[0]                                
            plt.title("Growth Curve Modelling") 
            plt.xlabel("Years") 
            plt.ylabel("Efficiency") 
            plt.plot(allyears.values,curve1)
            plt.plot(allyears.values,curve2) 
            plt.plot(allyears.values,curve3) 
            plt.plot(allyears.values,sumcurves)
            plt.plot(allyears.values,alldatah.values) 
            plt.legend(['Curve1', 'Curve2', 'Curve3', 'SumCurves', 'Data'])  
            # plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
            # Future functionality will save the plot to the sqlite database. Currently the graph is just displayed to the user. 
            plt.show()
        
        else:
            # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array        *** 4-curve ***
            def ackley(x):
                arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
                arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
                arg3 = x[9]+((x[8]-x[9])*np.reciprocal(1+np.exp(((years.values)-x[11])/x[10])))  # Curve 3 growth curve model
                arg3b = x[13]+((x[12]-x[13])*np.reciprocal(1+np.exp(((years.values)-x[15])/x[14])))  # Curve 4 growth curve model
                arg3c = np.add(arg1, arg2)                     # add curves together to get total model curve
                arg3d = np.add(arg3,arg3b)
                arg3e = np.add(arg3c,arg3d)
                arg4 = np.subtract(datah.values,arg3e)         # find difference of this array and the actual data
                arg5 = np.square(arg4)                         # Square the difference
                return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)

            res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
            res.x                                                          # returns the value of 'x' = optimized parameter array

            # Define variables used in logistic model - res.x is the array of parameters (This example has 4 curves)
            L1 = res.x[0]
            H1 = res.x[1]
            W1 = res.x[2]
            M1 = res.x[3]
            L2 = res.x[4]
            H2 = res.x[5]
            W2 = res.x[6]
            M2 = res.x[7]
            L3 = res.x[8]
            H3 = res.x[9]
            W3 = res.x[10]
            M3 = res.x[11]
            L4 = res.x[12]
            H4 = res.x[13]
            W4 = res.x[14]
            M4 = res.x[15]

            # Use the Growth Curve Model to create curve1, curve2, curve3, curve4, sumcurves arrays
            curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
            curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
            curve3 = H3+((L3-H3)*np.reciprocal(1+np.exp((years.values-M3)/W3)))
            curve4 = H4+((L4-H4)*np.reciprocal(1+np.exp((years.values-M4)/W4)))

            sumcurvesa=curve1+curve2
            sumcurvesb=curve3+curve4
            sumcurves=sumcurvesa+sumcurvesb
            Li1 = str(L1)
            Hi1 = str(H1)
            Wi1 = str(W1)
            Mi1 = str(M1)
            Li2 = str(L2)
            Hi2 = str(H2)
            Wi2 = str(W2)
            Mi2 = str(M2)
            Li3 = str(L3)
            Hi3 = str(H3)
            Wi3 = str(W3)
            Mi3 = str(M3)
            Li4 = str(L4)
            Hi4 = str(H4)
            Wi4 = str(W4)
            Mi4 = str(M4)

            # Determine the R-squared for a 4-curve fit
            first_sumcurves = list(map(float,sumcurves))                # This is our fitted curve (sumcurves) as a manipulable array o' data     
            x_values = first_datah                                      # The next 6 lines of code I got from online
            y_values = first_sumcurves                                  # Process for calculating R-squared (Matches excel)
            correlation_matrix = np.corrcoef(x_values, y_values)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2.
            print("The R-squared is :", r_squared)   
            
            if(r_squared > 0.93):
                finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2, Li3, Hi3, Wi3, Mi3, Li4, Hi4, Wi4, Mi4])
                print(finalparametri)
                conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])
                # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
                # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
                allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
                alldatah = datah[0]                                
                plt.title("Growth Curve Modelling") 
                plt.xlabel("Years") 
                plt.ylabel("Efficiency") 
                plt.plot(allyears.values,curve1)
                plt.plot(allyears.values,curve2) 
                plt.plot(allyears.values,curve3) 
                plt.plot(allyears.values,curve4) 
                plt.plot(allyears.values,sumcurves)
                plt.plot(allyears.values,alldatah.values) 
                plt.legend(['Curve1', 'Curve2', 'Curve3', 'Curve4', 'SumCurves', 'Data'])  
                plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
                plt.show()
                
            else:
                # Setup a minimization for sum of squares (like in excel) - parameters listed as part of the 'x' array
                # This one is for a 5-curve fit

                def ackley(x):
                    arg1 = x[1]+((x[0]-x[1])*np.reciprocal(1+np.exp(((years.values)-x[3])/x[2])))  # Curve 1 growth curve model
                    arg2 = x[5]+((x[4]-x[5])*np.reciprocal(1+np.exp(((years.values)-x[7])/x[6])))  # Curve 2 growth curve model
                    arg3 = x[9]+((x[8]-x[9])*np.reciprocal(1+np.exp(((years.values)-x[11])/x[10])))  # Curve 3 growth curve model
                    arg3b = x[13]+((x[12]-x[13])*np.reciprocal(1+np.exp(((years.values)-x[15])/x[14])))  # Curve 4 growth curve model
                    arg4a = x[17]+((x[16]-x[17])*np.reciprocal(1+np.exp(((years.values)-x[19])/x[18])))  # Curve 5 growth curve model
                    arg3c = np.add(arg1, arg2)                     # add curves together to get total model curve
                    arg3d = np.add(arg3,arg3b)
                    arg3e = np.add(arg3c,arg3d)
                    arg3f = np.add(arg3e,arg4a)
                    arg4 = np.subtract(datah.values,arg3f)         # find difference of this array and the actual data
                    arg5 = np.square(arg4)                         # Square the difference
                    return np.sum(arg5)                            # add up all elements in the squared difference (sum of squares)

                res = minimize(ackley, x0, method='Nelder-Mead', tol=1e-6)     # setup the algorithm
                res.x                                                          # Return the value of 'x' = array of optimized parameters

                # Define variables used in logistic model - res.x is the array of parameters (This example has 5 curves)
                L1 = res.x[0]
                H1 = res.x[1]
                W1 = res.x[2]
                M1 = res.x[3]
                L2 = res.x[4]
                H2 = res.x[5]
                W2 = res.x[6]
                M2 = res.x[7]
                L3 = res.x[8]
                H3 = res.x[9]
                W3 = res.x[10]
                M3 = res.x[11]
                L4 = res.x[12]
                H4 = res.x[13]
                W4 = res.x[14]
                M4 = res.x[15]
                L5 = res.x[16]
                H5 = res.x[17]
                W5 = res.x[18]
                M5 = res.x[19]

                # Use the Growth Curve Model to create curve1, curve2, curve3, curve4, curve5, sumcurves arrays
                curve1 = H1+((L1-H1)*np.reciprocal(1+np.exp((years.values-M1)/W1)))
                curve2 = H2+((L2-H2)*np.reciprocal(1+np.exp((years.values-M2)/W2)))
                curve3 = H3+((L3-H3)*np.reciprocal(1+np.exp((years.values-M3)/W3)))
                curve4 = H4+((L4-H4)*np.reciprocal(1+np.exp((years.values-M4)/W4)))
                curve5 = H5+((L5-H5)*np.reciprocal(1+np.exp((years.values-M5)/W5)))

                sumcurvesa=curve1+curve2
                sumcurvesb=curve3+curve4
                sumcurvesc=sumcurvesb+curve5
                sumcurves=sumcurvesa+sumcurvesc
                Li1 = str(L1)
                Hi1 = str(H1)
                Wi1 = str(W1)
                Mi1 = str(M1)
                Li2 = str(L2)
                Hi2 = str(H2)
                Wi2 = str(W2)
                Mi2 = str(M2)
                Li3 = str(L3)
                Hi3 = str(H3)
                Wi3 = str(W3)
                Mi3 = str(M3)
                Li4 = str(L4)
                Hi4 = str(H4)
                Wi4 = str(W4)
                Mi4 = str(M4)
                Li5 = str(L5)
                Hi5 = str(H5)
                Wi5 = str(W5)
                Mi5 = str(M5)

                # Determine the R-squared for a 5-curve fit
                first_sumcurves = list(map(float,sumcurves))                # This is our fitted curve (sumcurves) as a manipulable array o' data     
                x_values = first_datah                                      # The next 6 lines of code I got from online
                y_values = first_sumcurves                                  # Process for calculating R-squared (Matches excel)
                correlation_matrix = np.corrcoef(x_values, y_values)
                correlation_xy = correlation_matrix[0,1]
                r_squared = correlation_xy**2.
                print("The R-squared is :", r_squared)   
                
                # At this point, I'm done with the Russian doll system of curve fitting. If more than five curves are needed, let me know. 
                finalparametri = ", ".join([Li1, Hi1, Wi1, Mi1, Li2, Hi2, Wi2, Mi2, Li3, Hi3, Wi3, Mi3, Li4, Hi4, Wi4, Mi4, Li5, Hi5, Wi5, Mi5])
                conn.execute('UPDATE posts SET finalparameters = ?' ' WHERE id = ?', [finalparametri, post_id])
                # Use the MATLAB-style plotting library to visualize curves and original data (in this case, efficiency data)
                # Like I said earlier, split the years dataframe into a single column (has the header '0' for some reason)
                allyears = years[0]     # Comment all this out for now. We'll add the .png to the db later...
                alldatah = datah[0]                                
                plt.title("Growth Curve Modelling") 
                plt.xlabel("Years") 
                plt.ylabel("Efficiency") 
                plt.plot(allyears.values,curve1)
                plt.plot(allyears.values,curve2) 
                plt.plot(allyears.values,curve3) 
                plt.plot(allyears.values,curve4) 
                plt.plot(allyears.values,curve5) 
                plt.plot(allyears.values,sumcurves)
                plt.plot(allyears.values,alldatah.values) 
                plt.legend(['Curve1', 'Curve2', 'Curve3', 'Curve4', 'Curve5', 'SumCurves', 'Data'])  
                #plt.savefig('C:\\Users\\coun427\\OneDrive - PNNL\\Desktop\\Project3\\myplot.png', format='png')
                # I'd like to save this .png to sqlite database (probably as a 'blob' datatype) but that's for another day
                plt.show()
        

# Close the database and commit update(s)
conn.commit()
conn.close()   
